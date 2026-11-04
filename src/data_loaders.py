"""Data loaders for data/raw/soccernet.

On-disk layout (after scripts/download_soccernet.py):

    data/raw/soccernet/
        .cache/                      <- Hugging Face download cache; SKIPPED
        <league>/<season>/<game>/    e.g. england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal
            1_ResNET_TF2_PCA512.npy  <- first-half features, shape (frames, 512)
            2_ResNET_TF2_PCA512.npy  <- second-half features
            Labels-v2.json           <- {"annotations": [...], "gameHomeTeam": ..., ...}

Note: only the splits that were downloaded are present (the download script
fetches 'valid'), so "all games" means all games on disk.

Usage (from notebooks/eda.ipynb):

    from src.data_loaders import iter_games, load_game, build_annotation_table
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

# The 17 SoccerNet-v2 action classes. Kept explicit so classes with zero
# examples still show up in count tables.
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
    """Pointer to one game directory; no data is loaded until asked."""

    league: str
    season: str
    game: str
    path: Path

    @property
    def game_id(self) -> str:
        return f"{self.league}/{self.season}/{self.game}"


def _subdirs(p: Path) -> list[Path]:
    """Visible (non-hidden) subdirectories, sorted for deterministic order."""
    return sorted(d for d in p.iterdir() if d.is_dir() and not d.name.startswith("."))


def iter_games(root: Path = RAW_ROOT) -> Iterator[GameRef]:
    """Walk root/<league>/<season>/<game>/ and yield a GameRef per game dir.

    Hidden directories (notably .cache/) are skipped at every level. A game
    directory is yielded even if some of its files are missing, so that the
    data-quality checks can report the gaps rather than silently hide them.
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
    """Load one game: per-half feature arrays + parsed Labels-v2.json.

    Returns {"ref": GameRef, "features": [half1, half2], "labels": dict | None}.
    A missing/unreadable file yields None in its slot (never raises), so
    callers can audit gaps. Set mmap=True to avoid reading arrays into memory.
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
    """Aggregate annotations across ALL games into one long DataFrame.

    One row per annotation: league, season, game, game_id, label, team,
    visibility, half, seconds (within half), position_ms. Feed it to
    `class_count_table` for the per-class counts.
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
    """Per-class counts across all games, including zero-count classes.

    Columns: count, share (fraction of all annotations), n_games (games with
    >=1 instance). Sorted by count descending. Labels not in `classes` are kept
    (so unexpected classes are visible, not dropped).
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
    """One row per game describing schema oddities in Labels-v2.json.

    Columns: game_id, file_ok, n_annotations, extra_top_keys, missing_top_keys,
    n_ann_bad_keys (annotations whose key set != the expected 5),
    n_bad_gametime (gameTime not '<half> - mm:ss'), n_bad_position (position
    not an int string), n_time_mismatch (|position_ms/1000 - gameTime seconds|
    > tol_s), n_bad_team (team not in home/away/'not applicable'), n_url_mismatch (UrlLocal does
    not match the game's directory path).
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
    """Per-game, per-half feature-array metadata across all games.

    One row per (game, half): file_present, n_frames, feature_dim, dtype,
    n_nan, n_inf, min, max, mean, std. Missing/unreadable files get
    file_present=False and NaNs, so they are visible in the table.
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
