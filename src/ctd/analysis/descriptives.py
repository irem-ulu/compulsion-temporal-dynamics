"""Descriptive stats — pooled correlations + ICCs.

The pooled correlation matrix from the prototype is still useful as a
sanity check, but on its own it mixes within- and between-person
covariation. We additionally report the intra-class correlation (ICC)
for each outcome which tells us how much of the variance lives
between people vs within them — i.e. how much we *need* the multilevel
machinery.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def correlations(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    cols = cols or ["Stress", "OutsideTime", "Compulsions"]
    return df[cols].corr()


def within_between_correlations(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    group_col: str = "Participant",
) -> dict[str, pd.DataFrame]:
    """Decompose the correlation matrix into within- and between-person parts."""
    cols = cols or ["Stress", "OutsideTime", "Compulsions"]
    between = df.groupby(group_col)[cols].mean().corr()

    within = df[cols].copy()
    for c in cols:
        within[c] = df[c] - df.groupby(group_col)[c].transform("mean")
    return {"between": between, "within": within.corr()}


def icc(df: pd.DataFrame, col: str, group_col: str = "Participant") -> float:
    """ICC(1) — proportion of variance attributable to person.

    Returns a value in [0, 1]. ~0 means observations within a person
    are no more similar than between; ~1 means person identity explains
    everything. Mixed models start paying off above ~.05–.10.
    """
    grand_mean = df[col].mean()
    group_means = df.groupby(group_col)[col].mean()
    group_sizes = df.groupby(group_col)[col].size()

    # Between-person variance (weighted by group size, equivalent to ANOVA MSB).
    ss_between = float(((group_means - grand_mean) ** 2 * group_sizes).sum())
    ss_within = float(((df[col] - df[group_col].map(group_means)) ** 2).sum())

    n_groups = group_means.size
    n_total = len(df)
    df_between = n_groups - 1
    df_within = n_total - n_groups
    if df_between <= 0 or df_within <= 0:
        return float("nan")

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    avg_group_size = float(np.mean(group_sizes))

    # Shrout & Fleiss ICC(1,1).
    denom = ms_between + (avg_group_size - 1) * ms_within
    return (ms_between - ms_within) / denom if denom > 0 else float("nan")


def icc_table(
    df: pd.DataFrame,
    cols: list[str] | None = None,
    group_col: str = "Participant",
) -> pd.DataFrame:
    cols = cols or ["Stress", "OutsideTime", "Compulsions"]
    return pd.DataFrame(
        {"ICC": [icc(df, c, group_col=group_col) for c in cols]},
        index=cols,
    )
