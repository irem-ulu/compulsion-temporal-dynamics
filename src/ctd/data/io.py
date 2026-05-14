"""Tiny IO helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_raw_data(df: pd.DataFrame, path: str | Path = "data/raw_data.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_raw_data(path: str | Path = "data/raw_data.csv") -> pd.DataFrame:
    return pd.read_csv(path)
