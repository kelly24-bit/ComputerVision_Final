"""StatsBomb open-data loaders for the 2022 men's World Cup formation EDA.

Whatever comes back from the API is cached as gzipped JSON under
data/raw/statsbomb (git-ignored), so each match is only fetched once:

    data/raw/statsbomb/
        matches.json.gz
        events/<match_id>.json.gz     raw sb.events(fmt="json"), {event_id: event}
        frames/<match_id>.json.gz     raw sb.frames(fmt="json"), [{event_uuid, freeze_frame, ...}]

The parsing functions turn that cache into small DataFrames:

    tactics_table()      one row per Starting XI / Tactical Shift event
    formation_labels()   one row per (match, team) with the starting formation
    build_frame_table()  one row per 360 freeze-frame, with visible-player counts
    team_view_table()    one row per (freeze-frame, team), with that team's
                         visible player count and its starting / active formation

Terms:
    team-match   one team in one match (teams x matches). The independent unit.
    snapshot     one team's view of one freeze-frame. Many per team-match and
                 highly correlated with each other, so counts of them are inflated.
"""
from __future__ import annotations

import gzip
import json
import os
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", message="credentials were not supplied")

COMPETITION_ID = 43   # FIFA World Cup (men's)
SEASON_ID = 106       # 2022
CACHE_ROOT = Path("data/raw/statsbomb")


# --------------------------------------------------------------------------- cache
def _path(kind: str, match_id: int | None = None) -> Path:
    if kind == "matches":
        return CACHE_ROOT / "matches.json.gz"
    return CACHE_ROOT / kind / f"{match_id}.json.gz"


def _read(p: Path):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        return json.load(f)


def _write(p: Path, obj) -> None:
    """Write to a temp file first so an interrupted run can't leave a broken cache file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, p)


def _sb():
    from statsbombpy import sb  # imported here so cached runs don't need it

    return sb


# --------------------------------------------------------------------------- raw pulls
def get_matches(refresh: bool = False) -> pd.DataFrame:
    p = _path("matches")
    if p.exists() and not refresh:
        return pd.DataFrame(_read(p))
    df = _sb().matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    _write(p, json.loads(df.to_json(orient="records")))
    return df


def get_events_raw(match_id: int, refresh: bool = False) -> dict:
    p = _path("events", match_id)
    if p.exists() and not refresh:
        return _read(p)
    ev = _sb().events(match_id=int(match_id), fmt="json")
    _write(p, ev)
    return ev


def get_frames_raw(match_id: int, refresh: bool = False) -> list:
    """Raw 360 freeze-frames. Returns (and caches) [] if the match has none."""
    p = _path("frames", match_id)
    if p.exists() and not refresh:
        return _read(p)
    fr = _sb().frames(match_id=int(match_id), fmt="json")
    fr = list(fr) if fr else []
    _write(p, fr)
    return fr


def cache_all(pause: float = 0.2, verbose: bool = True) -> dict:
    """Pull and cache matches, events and frames for every World Cup match.

    Only files that aren't cached yet hit the API, with `pause` seconds between
    fetches. Returns {"fetched": n, "cached": n}, counted per (match, kind).
    """
    matches = get_matches()
    fetched = cached = 0
    for i, mid in enumerate(matches["match_id"].astype(int), start=1):
        for kind, getter in (("events", get_events_raw), ("frames", get_frames_raw)):
            if _path(kind, mid).exists():
                cached += 1
                continue
            getter(mid)
            fetched += 1
            time.sleep(pause)
        if verbose and i % 8 == 0:
            print(f"  {i}/{len(matches)} matches (fetched {fetched}, cached {cached})")
    return {"fetched": fetched, "cached": cached}


# --------------------------------------------------------------------------- parsing
def _match_ids() -> list[int]:
    return get_matches()["match_id"].astype(int).tolist()


def _fmt_formation(x) -> str | None:
    """Turn a raw formation (433, or None/odd) into '433', or None if missing."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    return str(x)


def tactics_table(match_ids: list[int] | None = None) -> pd.DataFrame:
    """One row per event with a `tactics` dict (Starting XI or Tactical Shift).

    Columns: match_id, event_id, index, period, minute, second, team_id, team,
    type, formation (str or None), formation_raw, lineup_len, and tactics_ok
    (False if `tactics` isn't a dict with a 'formation' and a list 'lineup').
    """
    rows = []
    for mid in match_ids or _match_ids():
        for eid, e in get_events_raw(mid).items():
            if "tactics" not in e:
                continue
            t = e["tactics"]
            ok = isinstance(t, dict) and "formation" in t and isinstance(t.get("lineup"), list)
            raw = t.get("formation") if isinstance(t, dict) else None
            rows.append(
                {
                    "match_id": mid,
                    "event_id": eid,
                    "index": e.get("index"),
                    "period": e.get("period"),
                    "minute": e.get("minute"),
                    "second": e.get("second"),
                    "team_id": (e.get("team") or {}).get("id"),
                    "team": (e.get("team") or {}).get("name"),
                    "type": (e.get("type") or {}).get("name"),
                    "formation": _fmt_formation(raw),
                    "formation_raw": raw,
                    "lineup_len": len(t["lineup"]) if ok else None,
                    "tactics_ok": ok,
                }
            )
    return pd.DataFrame(rows)


def formation_labels(tactics: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with its Starting XI formation, the team-match label."""
    s = tactics[tactics["type"] == "Starting XI"]
    return s[["match_id", "team", "formation", "lineup_len", "tactics_ok"]].rename(
        columns={"formation": "formation_start"}
    ).reset_index(drop=True)


def build_frame_table(match_ids: list[int] | None = None) -> pd.DataFrame:
    """One row per 360 freeze-frame, joined to its event.

    Columns: match_id, event_id, index, period, minute, second, event_type,
    team (who performed the event), opponent, n_total (all visible players),
    n_event_team (the event team's players, actor included), n_opponent,
    has_visible_area, event_found.
    """
    rows = []
    for mid in match_ids or _match_ids():
        ev = get_events_raw(mid)
        teams = sorted({(e.get("team") or {}).get("name") for e in ev.values()} - {None})
        for f in get_frames_raw(mid):
            eid = f.get("event_uuid")
            e = ev.get(eid)
            ff = f.get("freeze_frame") or []
            n_tm = sum(1 for p in ff if p.get("teammate"))
            team = (e.get("team") or {}).get("name") if e else None
            opp = next((t for t in teams if t != team), None) if team else None
            rows.append(
                {
                    "match_id": mid,
                    "event_id": eid,
                    "index": e.get("index") if e else None,
                    "period": e.get("period") if e else None,
                    "minute": e.get("minute") if e else None,
                    "second": e.get("second") if e else None,
                    "event_type": (e.get("type") or {}).get("name") if e else None,
                    "team": team,
                    "opponent": opp,
                    "n_total": len(ff),
                    "n_event_team": n_tm,
                    "n_opponent": len(ff) - n_tm,
                    "has_visible_area": bool(f.get("visible_area")),
                    "event_found": e is not None,
                }
            )
    return pd.DataFrame(rows)


def team_view_table(frames: pd.DataFrame, tactics: pd.DataFrame) -> pd.DataFrame:
    """Split each freeze-frame into one snapshot per team and attach formation labels.

    A frame gives two snapshots: the event team's players and the opponent's.
    Columns: match_id, event_id, index, team, view ('event_team' or 'opponent'),
    n_visible, formation_start (the team-match label), formation_active (the
    latest Starting XI / Tactical Shift at or before the event) and
    formation_shifted (active differs from start).
    """
    f = frames[frames["event_found"]]
    own = f.assign(view="event_team", n_visible=f["n_event_team"])
    opp = f.assign(view="opponent", team=f["opponent"], n_visible=f["n_opponent"])
    cols = ["match_id", "event_id", "index", "period", "minute", "second", "team", "view", "n_visible"]
    snaps = pd.concat([own[cols], opp[cols]], ignore_index=True)
    snaps["index"] = snaps["index"].astype(int)

    start = formation_labels(tactics)[["match_id", "team", "formation_start"]]
    snaps = snaps.merge(start, on=["match_id", "team"], how="left")

    t = (
        tactics[tactics["tactics_ok"]][["match_id", "team", "index", "formation"]]
        .rename(columns={"formation": "formation_active"})
        .astype({"index": int})
        .sort_values("index")
    )
    snaps = pd.merge_asof(
        snaps.sort_values("index"), t, on="index", by=["match_id", "team"], direction="backward"
    )
    snaps["formation_shifted"] = snaps["formation_active"] != snaps["formation_start"]
    return snaps.sort_values(["match_id", "index", "view"]).reset_index(drop=True)
