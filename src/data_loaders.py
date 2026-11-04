"""Loaders for the SoccerNet files in data/raw/soccernet.

Layout after running scripts/download_soccernet.py:

    data/raw/soccernet/
        .cache/                      Hugging Face cache, ignored here
        <league>/<season>/<game>/    e.g. england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal
            1_ResNET_TF2_PCA512.npy  first-half features, (frames, 512)
            2_ResNET_TF2_PCA512.npy  second-half features
            Labels-v2.json           {"annotations": [...], "gameHomeTeam": ..., ...}

Only the 'valid' split gets downloaded, so "all games" here means every game
on disk, not all of SoccerNet.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RAW_ROOT = Path("data/raw/soccernet")
FEATURE_FILES = ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy")
LABELS_FILE = "Labels-v2.json"

# the 17 SoccerNet-v2 classes, listed out so a class with zero examples
# still shows up in the count tables
CLASSES_V2 = (
    "Penalty",
    "Kick-off",
    "Goal",
    "Substitution",
    "Offside",
    "Shots on target",
    "Shots off target",
    "Clearance",
    "Ball out of play",
    "Throw-in",
    "Foul",
    "Indirect free-kick",
    "Direct free-kick",
    "Corner",
    "Yellow card",
    "Red card",
    "Yellow->red card",
)


@dataclass(frozen=True)
class GameRef:
    """Points at one game directory. Nothing is loaded until you ask."""

    league: str
    season: str
    game: str
    path: Path

    @property
    def game_id(self) -> str:
        return f"{self.league}/{self.season}/{self.game}"


def _subdirs(p: Path) -> list[Path]:
    """Non-hidden subdirectories, sorted."""
    return sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))


def iter_games(root: Path = RAW_ROOT) -> Iterator[GameRef]:
    """Yield a GameRef for each root/<league>/<season>/<game>/ directory.

    Hidden directories (like .cache/) are skipped. A game directory is still
    yielded if some of its files are missing, so the quality checks can report
    the gap instead of it going unnoticed.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} not found; run scripts/download_soccernet.py")
    for league in _subdirs(root):
        for season in _subdirs(league):
            for game in _subdirs(season):
                yield GameRef(league.name, season.name, game.name, game)


def _parse_game_time(s: str) -> tuple[float, float]:
    """'1 - 05:32' -> (1, 332.0). Returns (nan, nan) if unparseable."""
    try:
        half, clock = (x.strip() for x in s.split(" - "))
        mm, ss = clock.split(":")
        return float(half), int(mm) * 60 + float(ss)
    except (ValueError, AttributeError):
        return float("nan"), float("nan")


def load_game(ref: GameRef, mmap: bool = False) -> dict:
    """Load one game's feature arrays (one per half) and its Labels-v2.json.

    Returns {"ref": GameRef, "features": [half1, half2], "labels": dict | None}.
    A missing or unreadable file comes back as None instead of raising.
    mmap=True avoids reading the arrays into memory.
    """
    features: list[np.ndarray | None] = []
    for fname in FEATURE_FILES:
        fp = ref.path / fname
        try:
            features.append(np.load(fp, mmap_mode="r" if mmap else None))
        except (FileNotFoundError, ValueError, OSError, EOFError):
            features.append(None)

    labels = None
    try:
        with open(ref.path / LABELS_FILE, encoding="utf-8") as f:
            labels = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        pass
    return {"ref": ref, "features": features, "labels": labels}


def build_annotation_table(root: Path = RAW_ROOT) -> pd.DataFrame:
    """All annotations from every game in one long DataFrame.

    One row per annotation: league, season, game, game_id, label, team,
    visibility, half, seconds (within the half), position_ms. Pass it to
    `class_count_table` to get per-class counts.
    """
    rows = []
    for ref in iter_games(root):
        try:
            with open(ref.path / LABELS_FILE, encoding="utf-8") as f:
                anns = json.load(f).get("annotations", [])
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            continue
        for a in anns:
            half, secs = _parse_game_time(a.get("gameTime", ""))
            try:
                pos = int(a.get("position"))
            except (TypeError, ValueError):
                pos = np.nan
            rows.append(
                {
                    "league": ref.league,
                    "season": ref.season,
                    "game": ref.game,
                    "game_id": ref.game_id,
                    "label": a.get("label"),
                    "team": a.get("team"),
                    "visibility": a.get("visibility"),
                    "half": half,
                    "seconds": secs,
                    "position_ms": pos,
                }
            )
    return pd.DataFrame(rows)


def class_count_table(ann: pd.DataFrame, classes=CLASSES_V2) -> pd.DataFrame:
    """Per-class counts across all games, including classes with zero examples.

    Columns: count, share (of all annotations), n_games (games with at least
    one). Sorted by count, largest first. Labels outside `classes` are kept
    so an unexpected class shows up instead of being dropped.
    """
    counts = ann["label"].value_counts()
    n_games = ann.groupby("label")["game_id"].nunique()
    idx = list(classes) + [c for c in counts.index if c not in classes]
    out = pd.DataFrame(
        {
            "count": counts.reindex(idx).fillna(0).astype(int),
            "n_games": n_games.reindex(idx).fillna(0).astype(int),
        }
    )
    out["share"] = out["count"] / out["count"].sum()
    return out.sort_values("count", ascending=False)


EXPECTED_TOP_KEYS = frozenset(
    {"UrlLocal", "UrlYoutube", "annotations", "gameAwayTeam", "gameDate",
     "gameHomeTeam", "gameScore"}
)
EXPECTED_ANN_KEYS = frozenset({"gameTime", "label", "position", "team", "visibility"})


def audit_label_schema(root: Path = RAW_ROOT, tol_s: float = 1.5) -> pd.DataFrame:
    """One row per game with counts of oddities in its Labels-v2.json.

    Columns: game_id, file_ok, n_annotations, extra_top_keys, missing_top_keys,
    n_ann_bad_keys (annotation keys differ from the expected 5), n_bad_gametime
    (gameTime isn't '<half> - mm:ss'), n_bad_position (position isn't an int
    string), n_time_mismatch (position_ms/1000 and the gameTime seconds differ
    by more than tol_s), n_bad_team (team isn't home, away or 'not applicable'),
    n_url_mismatch (UrlLocal doesn't match the game's directory).
    """
    rows = []
    for ref in iter_games(root):
        row = {"game_id": ref.game_id, "file_ok": False, "n_annotations": 0,
               "extra_top_keys": "", "missing_top_keys": "", "n_ann_bad_keys": 0,
               "n_bad_gametime": 0, "n_bad_position": 0, "n_time_mismatch": 0,
               "n_bad_team": 0, "n_url_mismatch": 0}
        try:
            with open(ref.path / LABELS_FILE, encoding="utf-8") as f:
                d = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            rows.append(row)
            continue
        row["file_ok"] = True
        keys = set(d)
        row["extra_top_keys"] = ",".join(sorted(keys - EXPECTED_TOP_KEYS))
        row["missing_top_keys"] = ",".join(sorted(EXPECTED_TOP_KEYS - keys))
        expected_url = f"{ref.league}/{ref.season}/{ref.game}/"
        row["n_url_mismatch"] = int(d.get("UrlLocal") != expected_url)
        anns = d.get("annotations", [])
        row["n_annotations"] = len(anns)
        for a in anns:
            if set(a) != EXPECTED_ANN_KEYS:
                row["n_ann_bad_keys"] += 1
            half, secs = _parse_game_time(a.get("gameTime", ""))
            bad_time = np.isnan(secs)
            row["n_bad_gametime"] += int(bad_time)
            try:
                pos = int(a.get("position"))
            except (TypeError, ValueError):
                row["n_bad_position"] += 1
                pos = None
            if pos is not None and not bad_time and abs(pos / 1000 - secs) > tol_s:
                row["n_time_mismatch"] += 1
            if a.get("team") not in ("home", "away", "not applicable"):
                row["n_bad_team"] += 1
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_summary(root: Path = RAW_ROOT) -> pd.DataFrame:
    """Feature-array info for every game and half.

    One row per (game, half): file_present, n_frames, feature_dim, dtype,
    n_nan, n_inf, min, max, mean, std. A missing or unreadable file gets
    file_present=False and NaNs so it still shows up in the table.
    """
    rows = []
    for ref in iter_games(root):
        for half, fname in enumerate(FEATURE_FILES, start=1):
            row = {
                "league": ref.league,
                "season": ref.season,
                "game": ref.game,
                "game_id": ref.game_id,
                "half": half,
                "file_present": False,
                "n_frames": np.nan,
                "feature_dim": np.nan,
                "dtype": None,
                "n_nan": np.nan,
                "n_inf": np.nan,
                "min": np.nan,
                "max": np.nan,
                "mean": np.nan,
                "std": np.nan,
            }
            try:
                arr = np.load(ref.path / fname)
            except (FileNotFoundError, ValueError, OSError, EOFError):
                rows.append(row)
                continue
            row.update(file_present=True, dtype=str(arr.dtype))
            if arr.ndim == 2:
                row["n_frames"], row["feature_dim"] = arr.shape
            else:
                row["n_frames"] = arr.shape[0] if arr.ndim else 0
            if arr.size:
                a = arr.astype(np.float64, copy=False)
                finite = np.isfinite(a)
                row["n_nan"] = int(np.isnan(a).sum())
                row["n_inf"] = int(np.isinf(a).sum())
                if finite.any():
                    v = a[finite]
                    row.update(min=v.min(), max=v.max(), mean=v.mean(), std=v.std())
            else:
                row.update(n_nan=0, n_inf=0)
            rows.append(row)
    return pd.DataFrame(rows)
