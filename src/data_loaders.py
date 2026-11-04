"""Data loaders for data/raw/soccernet (STUB -- structure only, no logic yet).

Observed on-disk layout (after scripts/download_soccernet.py):

    data/raw/soccernet/
        .cache/                      <- Hugging Face download cache; SKIP
        <league>/<season>/<game>/    e.g. england_epl/2014-2015/2015-04-11 - 19-30 Burnley 0 - 1 Arsenal
            1_ResNET_TF2_PCA512.npy  <- first-half features, shape (frames, 512)
            2_ResNET_TF2_PCA512.npy  <- second-half features
            Labels-v2.json           <- {"annotations": [...], ...}

Intended use (from notebooks/eda.ipynb):

    from src.data_loaders import iter_games, load_game, build_annotation_table
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

RAW_ROOT = Path("data/raw/soccernet")
FEATURE_FILES = ("1_ResNET_TF2_PCA512.npy", "2_ResNET_TF2_PCA512.npy")
LABELS_FILE = "Labels-v2.json"


@dataclass(frozen=True)
class GameRef:
    """Pointer to one game directory; no data is loaded until asked."""

    league: str
    season: str
    game: str
    path: Path


def iter_games(root: Path = RAW_ROOT) -> Iterator[GameRef]:
    """Walk root/<league>/<season>/<game>/ and yield a GameRef per game.

    TODO: skip hidden dirs (.cache); only yield dirs that contain LABELS_FILE.
    """
    raise NotImplementedError


def load_game(ref: GameRef) -> dict:
    """Load one game: per-half feature arrays + parsed Labels-v2.json.

    TODO: return {"features": [half1, half2], "labels": <parsed json>}.
    """
    raise NotImplementedError


def build_annotation_table(root: Path = RAW_ROOT) -> pd.DataFrame:
    """Aggregate annotations across ALL games into one long DataFrame.

    TODO: one row per annotation, with game identifiers (league, season, game),
    half, position/time, and label, ready for class-balance EDA.
    """
    raise NotImplementedError


def build_feature_summary(root: Path = RAW_ROOT) -> pd.DataFrame:
    """Aggregate per-game, per-half feature-array metadata across all games.

    TODO: one row per (game, half): n_frames, feature_dim, and any sanity stats.
    """
    raise NotImplementedError
