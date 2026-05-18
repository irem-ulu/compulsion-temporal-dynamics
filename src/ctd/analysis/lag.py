"""Multi-lag dynamic models for the stress ↔ compulsion relationship.

The prototype's lag analysis was a single OLS with one lag:

    Compulsions ~ PrevStress  (lag-1, pooled across people)

That has two problems beyond the usual nesting issue:

1. It only looks at lag-1. A 2-hour-back stress spike could matter too.
2. It only looks at one direction. If we believe stress drives
   compulsions, the model should also reject the reverse (compulsions
   predicting later stress) — or at least quantify it.

This module fits per-person within-centered multi-lag MixedLMs in both
directions and returns a tidy coefficient table.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def add_lag_features(
    df: pd.DataFrame,
    cols: tuple[str, ...] = ("Stress", "Compulsions", "OutsideTime"),
    max_lag: int = 3,
    group_col: str = "Participant",
    sort_cols: tuple[str, ...] = ("Participant", "Day", "TimeBin"),
) -> pd.DataFrame:
    df = df.sort_values(list(sort_cols)).reset_index(drop=True).copy()
    for col in cols:
        # Always work with the within-person deviation so the lag effect
        # we estimate is the within-person dynamic, not a between-person
        # artefact.
        if f"{col}_within" not in df.columns:
            df[f"{col}_within"] = df[col] - df.groupby(group_col)[col].transform("mean")
        for lag in range(1, max_lag + 1):
            df[f"{col}_lag{lag}"] = (
                df.groupby(group_col)[f"{col}_within"].shift(lag)
            )
    return df


def stress_to_compulsion_lag(
    df: pd.DataFrame, max_lag: int = 3
) -> pd.DataFrame:
    """Compulsions ~ contemporaneous + lag1..lagK stress, mixed model.

    Returns a coefficient table.
    """
    df = add_lag_features(df, max_lag=max_lag).dropna(
        subset=[f"Stress_lag{i}" for i in range(1, max_lag + 1)]
    )
    lag_terms = " + ".join(f"Stress_lag{i}" for i in range(1, max_lag + 1))
    f = f"Compulsions ~ Stress_within + {lag_terms}"
    fit = smf.mixedlm(f, data=df, groups=df["Participant"]).fit(method="lbfgs")
    return _tidy_coef_table(fit)


def compulsion_to_stress_lag(
    df: pd.DataFrame, max_lag: int = 3
) -> pd.DataFrame:
    """Reverse-direction check: Stress ~ lag1..K compulsions, mixed model."""
    df = add_lag_features(df, max_lag=max_lag).dropna(
        subset=[f"Compulsions_lag{i}" for i in range(1, max_lag + 1)]
    )
    lag_terms = " + ".join(f"Compulsions_lag{i}" for i in range(1, max_lag + 1))
    f = f"Stress ~ {lag_terms}"
    fit = smf.mixedlm(f, data=df, groups=df["Participant"]).fit(method="lbfgs")
    return _tidy_coef_table(fit)


def _tidy_coef_table(fit) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "beta": fit.params,
            "se": fit.bse,
            "z": fit.tvalues,
            "p": fit.pvalues,
        }
    )
    # Drop the random-effects variance row from display; it's not a fixed effect.
    return out.drop(index=[i for i in out.index if "Var" in i], errors="ignore")
