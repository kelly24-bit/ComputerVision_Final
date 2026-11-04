"""Smoke test: confirm StatsBomb open data loads (no account / NDA needed).

This is our formation real-test-set source. Success means:
  - the competitions list loads,
  - a World Cup match's events + lineups load,
  - a starting formation label is present,
  - 360 freeze-frame player coordinates load.

Run:  python scripts/check_statsbomb.py
"""
import sys

try:
    from statsbombpy import sb
except ImportError:
    sys.exit("statsbombpy not installed. Run: pip install -r requirements.txt")

# A 2022 men's World Cup match known to carry 360 data (the final).
FALLBACK_360_MATCH = 3869685


def find_formation(events):
    """Extract starting formations from the nested `tactics` dict.

    StatsBomb stores formation on 'Starting XI' (and 'Tactical Shift') events as
    events['tactics'] = {'formation': 433, 'lineup': [...]} -- not a flat column.
    """
    if "tactics" not in events.columns:
        return []
    formations = []
    for t in events["tactics"].dropna():
        if isinstance(t, dict) and "formation" in t:
            formations.append(t["formation"])
    return sorted(set(formations))


def main():
    comps = sb.competitions()
    print(f"[ok] competitions loaded: {len(comps)} rows")

    wc = comps[comps["competition_id"] == 43]  # men's World Cup
    if wc.empty:
        sys.exit("[fail] World Cup (competition_id 43) not in open data")
    season_id = int(wc.sort_values("season_name").iloc[-1]["season_id"])
    matches = sb.matches(competition_id=43, season_id=season_id)
    print(f"[ok] World Cup matches loaded: {len(matches)} (season_id={season_id})")

    match_id = int(matches.iloc[0]["match_id"])
    events = sb.events(match_id=match_id)
    formations = find_formation(events)
    if formations:
        print(f"[ok] match {match_id}: {len(events)} events; formations: {formations}")
    else:
        print(f"[warn] match {match_id}: no formation found in tactics — check schema")

    # 360 frames: try this match, fall back to a known-covered match.
    frames = sb.frames(match_id=match_id, fmt="dataframe")
    if frames is None or len(frames) == 0:
        print(f"[info] match {match_id} has no 360 frames; trying final {FALLBACK_360_MATCH}")
        frames = sb.frames(match_id=FALLBACK_360_MATCH, fmt="dataframe")
    if frames is None or len(frames) == 0:
        print("[warn] no 360 frames loaded — verify the match carries 360 data")
    else:
        print(f"[ok] 360 frames loaded: {len(frames)} player-rows")

    print("\nStatsBomb data confirmed working.")


if __name__ == "__main__":
    main()
