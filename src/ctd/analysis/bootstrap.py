"""Bootstrap CIs for the mixed-effect coefficients.

A plain row bootstrap is wrong here: rows from the same person are
correlated, so resampling rows independently massively underestimates
the standard error. The right thing is the cluster bootstrap — resample
*participants* (with replacement) and take all of their rows. That
preserves the within-person correlation structure.

The naive row version is still exposed as ``row_bootstrap`` so we can
show in the report that it understates uncertainty.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _summarise(samples: list[pd.Series]) -> pd.DataFrame:
    boot = pd.concat(samples, axis=1).T
    return pd.DataFrame(
        {
            "beta_mean": boot.mean(),
            "beta_se": boot.std(ddof=1),
            "ci_lo": boot.quantile(0.025),
            "ci_hi": boot.quantile(0.975),
        }
    )


def row_bootstrap(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], pd.Series],
    n_iter: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Naive row bootstrap — kept for comparison, do not use for inference."""
    rng = np.random.default_rng(seed)
    samples: list[pd.Series] = []
    for _ in range(n_iter):
        idx = rng.integers(0, len(df), size=len(df))
        try:
            samples.append(fit_fn(df.iloc[idx]))
        except Exception:
            continue
    return _summarise(samples)


def cluster_bootstrap(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], pd.Series],
    group_col: str = "Participant",
    n_iter: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    """Cluster bootstrap — resample participants with replacement.

    This is the standard CI for EMA / panel data when N (people) is
    small. Each iteration builds a synthetic dataset by picking the same
    number of participants with replacement and taking all their rows.
    Each resampled participant gets a fresh id so MixedLM treats them
    as independent clusters even when the same real participant is
    drawn twice.
    """
    rng = np.random.default_rng(seed)
    participants = df[group_col].unique()
    n = len(participants)
    grouped = {p: df[df[group_col] == p] for p in participants}

    samples: list[pd.Series] = []
    for _ in range(n_iter):
        picked = rng.choice(participants, size=n, replace=True)
        chunks = []
        for k, p in enumerate(picked):
            chunk = grouped[p].copy()
            chunk[group_col] = f"{p}_b{k}"
            chunks.append(chunk)
        boot_df = pd.concat(chunks, ignore_index=True)
        try:
            samples.append(fit_fn(boot_df))
        except Exception:
            continue
    return _summarise(samples)


# Back-compat shim — keep the old name pointing at the right thing now.
def bootstrap_coefficients(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], pd.Series],
    n_iter: int = 500,
    seed: int = 0,
) -> pd.DataFrame:
    return cluster_bootstrap(df, fit_fn, n_iter=n_iter, seed=seed)
