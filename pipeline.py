#!/usr/bin/env python3
"""ProProp pipeline: live prop odds + ESPN box scores -> data.json (run every day).

Setup:   pip install pandas requests        (already installed in Colab)
Run:     ODDS_API_KEY=your_key python pipeline.py
Optional: injuries.json  e.g. {"Player Name": {"status": "OUT"}, "Other Name": {"mins": 30}}

Projection = per-minute rate x projected minutes, then adjusted for the opponent's
defense against that stat (per 100 possessions) and for the expected game pace.
"""
import os, re, json, sys, time, unicodedata, datetime as dt
from statistics import NormalDist, pstdev
from zoneinfo import ZoneInfo
import pandas as pd
import requests

KEY = os.environ.get("ODDS_API_KEY")
BOOKS = "draftkings,fanduel,betmgm"
API = "https://api.the-odds-api.com/v4/sports/basketball_nba"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE = "logs_cache_v2.csv"
MARKETS = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST",
           "player_threes": "3PM", "player_points_rebounds_assists": "PRA"}
_only = os.environ.get("ODDS_MARKETS")  # e.g. "PTS,REB,AST" to save odds-API credits
if _only:
    MARKETS = {k: v for k, v in MARKETS.items() if v in _only.split(",")}
DEF_KEY = {"PTS": "pts", "REB": "reb", "AST": "ast", "3PM": "tpm", "PRA": "pra"}
KEYS = ["pace", "pts", "reb", "ast", "tpm", "pra", "fg", "tp", "oreb", "tov", "stl", "blk"]
MIN_GAMES = 5
SUM_COLS = ["PTS", "REB", "AST", "FG3M", "FG3A", "FGM", "FGA", "FTA", "OREB", "TOV", "STL", "BLK"]


def norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'’]", "", s)
    return re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s).strip()


def tn(name):  # team-name key (ESPN says "LA Clippers", the odds feed says "Los Angeles Clippers")
    return name.lower().replace("la clippers", "los angeles clippers")


def dec(o): return 1 + o / 100 if o > 0 else 1 + 100 / -o
def imp(o): return 1 / dec(o)
def ev(p, o): return p * (dec(o) - 1) - (1 - p)


def espn(path, **params):
    r = requests.get(f"{ESPN}/{path}", params=params, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return r.json()


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def made_att(x):
    try:
        a, b = str(x).split("-")
        return float(a), float(b)
    except ValueError:
        return 0.0, 0.0


def parse_game(gid, day):
    out = []
    box = espn("summary", event=gid).get("boxscore", {}).get("players", [])
    if len(box) != 2:
        return out
    abbrs = [t["team"].get("abbreviation", "") for t in box]
    for i, t in enumerate(box):
        tname = t["team"].get("displayName", abbrs[i])
        for grp in t.get("statistics", []):
            for a in grp.get("athletes", []):
                s = dict(zip(grp.get("labels", []), a.get("stats") or []))
                mins = num(s.get("MIN"))
                if mins <= 0:
                    continue
                fgm, fga = made_att(s.get("FG"))
                tpm, tpa = made_att(s.get("3PT"))
                _, fta = made_att(s.get("FT"))
                out.append({
                    "GAME_ID": gid, "PLAYER_NAME": a["athlete"]["displayName"],
                    "TEAM_ABBREVIATION": abbrs[i], "TEAM_NAME": tname, "OPP": abbrs[1 - i],
                    "GAME_DATE": pd.Timestamp(day), "MIN": mins,
                    "PTS": num(s.get("PTS")), "REB": num(s.get("REB")), "AST": num(s.get("AST")),
                    "FG3M": tpm, "FG3A": tpa, "FGM": fgm, "FGA": fga, "FTA": fta,
                    "OREB": num(s.get("OREB")), "TOV": num(s.get("TO")),
                    "STL": num(s.get("STL")), "BLK": num(s.get("BLK")),
                })
    return out


def load_logs():
    start = dt.date.fromisoformat(os.environ.get("SEASON_START", "2026-10-20"))
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    old = pd.read_csv(CACHE, parse_dates=["GAME_DATE"]) if os.path.exists(CACHE) else pd.DataFrame()
    day = start
    if not old.empty:  # refetch the last 2 days so late or unfinished games are never missed
        day = max(start, old["GAME_DATE"].max().date() - dt.timedelta(days=1))
        old = old[old["GAME_DATE"] < pd.Timestamp(day)]
    rows, failed = [], None
    while day <= today:
        try:
            sb = espn("scoreboard", dates=day.strftime("%Y%m%d"))
            ids = [e["id"] for e in sb.get("events", []) if e["status"]["type"].get("completed")]
            day_rows = []
            for gid in ids:
                day_rows += parse_game(gid, day)
                time.sleep(0.3)
        except Exception as e:
            failed = f"{day}: {e}"
            break
        print(f"  {day}: {len(ids)} games, {len(day_rows)} player lines")
        rows += day_rows
        day += dt.timedelta(days=1)
    df = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    if not df.empty:
        df.to_csv(CACHE, index=False)
    if failed:
        sys.exit(f"ESPN data failed on {failed}\nDays already fetched are saved; rerun to continue.")
    if df.empty:
        sys.exit("No game logs found. Check SEASON_START (YYYY-MM-DD) is on or before opening night.")
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]
    df["3PM"] = df["FG3M"]
    df["key"] = df["PLAYER_NAME"].map(norm)
    return df.sort_values("GAME_DATE")


def team_profiles(df):
    """What each team allows (per 100 possessions), how fast it plays, and what it forces."""
    g = df.groupby(["GAME_ID", "TEAM_ABBREVIATION", "OPP"], as_index=False)[SUM_COLS].sum()
    g["POSS"] = g["FGA"] + 0.44 * g["FTA"] - g["OREB"] + g["TOV"]
    m = g.merge(g.add_prefix("o_"), left_on=["GAME_ID", "OPP"], right_on=["o_GAME_ID", "o_TEAM_ABBREVIATION"])
    m["GPACE"] = (m["POSS"] + m["o_POSS"]) / 2
    prof = {}
    for t, x in m.groupby("TEAM_ABBREVIATION"):
        op, sp = x["o_POSS"].sum(), x["POSS"].sum()
        p100 = lambda c: x["o_" + c].sum() / op * 100
        d = {"n": len(x), "pace": x["GPACE"].mean(), "pts": p100("PTS"), "reb": p100("REB"),
             "ast": p100("AST"), "tpm": p100("FG3M"), "oreb": p100("OREB"),
             "fg": x["o_FGM"].sum() / max(x["o_FGA"].sum(), 1) * 100,
             "tp": x["o_FG3M"].sum() / max(x["o_FG3A"].sum(), 1) * 100,
             "tov": x["o_TOV"].sum() / op * 100,  # turnovers this defense forces
             "stl": x["STL"].sum() / sp * 100, "blk": x["BLK"].sum() / sp * 100}
        d["pra"] = d["pts"] + d["reb"] + d["ast"]
        prof[t] = d
    league = {k: sum(p[k] for p in prof.values()) / len(prof) for k in KEYS}
    ranks = {k: {t: i + 1 for i, t in enumerate(sorted(prof, key=lambda t: -prof[t][k]))} for k in KEYS}
    return prof, league, ranks


def project(g, stat, mins_override=None):
    g = g[g["MIN"] > 0]
    if len(g) < MIN_GAMES:
        return None
    l10, l15 = g.tail(10), g.tail(15)
    mins = mins_override or (0.6 * l10["MIN"].mean() + 0.4 * l15["MIN"].mean())
    rate = 0.6 * (l10[stat].sum() / l10["MIN"].sum()) + 0.4 * (g[stat].sum() / g["MIN"].sum())
    return {"proj": rate * mins, "sd": max(pstdev(l15[stat]), 1.0), "mins": mins,
            "last10": list(l10[stat]), "team": g.iloc[-1]["TEAM_ABBREVIATION"]}


def adjust(prof, league, own, opp, stat):
    """Defense and pace multipliers, shrunk toward 1.0 while samples are small."""
    po, pw = prof.get(opp), prof.get(own)
    if not po or not pw:
        return 1.0, 1.0, None
    k = DEF_KEY[stat]
    d = 1 + po["n"] / (po["n"] + 8) * (po[k] / league[k] - 1)
    tonight = (pw["pace"] + po["pace"]) / 2
    base = (pw["pace"] + league["pace"]) / 2
    w = min(po["n"], pw["n"]); w = w / (w + 8)
    return d, 1 + w * (tonight / base - 1), tonight


def get(url, **params):
    r = requests.get(url, params={"apiKey": KEY, **params}, timeout=30)
    r.raise_for_status()
    if r.headers.get("x-requests-remaining"):
        print(f"  odds credits remaining: {r.headers['x-requests-remaining']}")
    return r.json()


def main():
    if not KEY:
        sys.exit("Set ODDS_API_KEY first (free key at the-odds-api.com).")
    injuries = {}
    if os.path.exists("injuries.json"):
        injuries = {norm(k): v for k, v in json.load(open("injuries.json")).items()}

    print("Loading game logs...")
    logs = load_logs()
    by_player = {k: g for k, g in logs.groupby("key")}
    prof, league, ranks = team_profiles(logs)
    names = logs[["TEAM_NAME", "TEAM_ABBREVIATION"]].drop_duplicates()
    abbr_by_name = {tn(n): a for n, a in names.values}
    team_names = dict(zip(names["TEAM_ABBREVIATION"], names["TEAM_NAME"]))

    print("Loading today's games...")
    now = dt.datetime.now(dt.timezone.utc)
    start = lambda e: dt.datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
    events = [e for e in get(f"{API}/events") if now <= start(e) <= now + dt.timedelta(hours=30)]

    rows = []
    for e in events:
        game = f'{e["away_team"]} @ {e["home_team"]}'
        ha, aa = abbr_by_name.get(tn(e["home_team"])), abbr_by_name.get(tn(e["away_team"]))
        print(f"Props: {game}")
        odds = get(f'{API}/events/{e["id"]}/odds', regions="us", markets=",".join(MARKETS),
                   bookmakers=BOOKS, oddsFormat="american")
        for bk in odds.get("bookmakers", []):
            for mk in bk["markets"]:
                stat = MARKETS.get(mk["key"])
                pairs = {}
                for o in mk["outcomes"]:
                    pairs.setdefault((o["description"], o["point"]), {})[o["name"]] = o["price"]
                for (player, line), px in pairs.items():
                    key = norm(player)
                    inj = injuries.get(key, {})
                    if key not in by_player or inj.get("status") == "OUT":
                        continue
                    pr = project(by_player[key], stat, inj.get("mins"))
                    if not pr:
                        continue
                    opp = aa if pr["team"] == ha else ha if pr["team"] == aa else None
                    d, p_adj, gpace = adjust(prof, league, pr["team"], opp, stat)
                    proj, sd = pr["proj"] * d * p_adj, pr["sd"] * d * p_adj
                    p_over = 1 - NormalDist(proj, sd).cdf(line)
                    hit_over = sum(v > line for v in pr["last10"]) / len(pr["last10"])
                    both = "Over" in px and "Under" in px
                    for side in ("Over", "Under"):
                        if side not in px:
                            continue
                        p = p_over if side == "Over" else 1 - p_over
                        fair = None
                        if both:
                            a, b = imp(px["Over"]), imp(px["Under"])
                            fair = (a if side == "Over" else b) / (a + b)
                        rows.append({
                            "player": player, "team": pr["team"], "game": game, "commence": e["commence_time"],
                            "market": stat, "book": bk["title"], "line": line, "side": side, "odds": px[side],
                            "p": round(p, 4), "fair": round(fair, 4) if fair else None, "ev": round(ev(p, px[side]), 4),
                            "proj": round(proj, 1), "base": round(pr["proj"], 1), "sd": round(sd, 1),
                            "mins": round(pr["mins"], 1), "hit": round(hit_over if side == "Over" else 1 - hit_over, 2),
                            "opp": opp, "da": round((d - 1) * 100, 1), "pa": round((p_adj - 1) * 100, 1),
                            "rk": ranks[DEF_KEY[stat]].get(opp) if opp else None,
                            "gp": round(gpace, 1) if gpace else None,
                        })

    teams = {t: {"name": team_names.get(t, t), "n": int(p["n"]),
                 **{k: round(p[k], 1) for k in KEYS}, "rk": {k: ranks[k][t] for k in KEYS}}
             for t, p in prof.items()}
    json.dump({"generated": now.isoformat(), "props": rows, "teams": teams}, open("data.json", "w"))
    print(f"Wrote data.json: {len(rows)} prop sides, {len(events)} games, {len(teams)} team profiles.")


if __name__ == "__main__":
    main()
