"""Leave-one-participant-out sensitivity check.

For each participant, refit the target model on the remaining N-1 and
record the coefficient. If dropping any single person flips a sign
or changes a coefficient by >50%, that's worth flagging — it usually
means the headline effect is being driven by one or two people.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def leave_one_out(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], pd.Series],
    group_col: str = "Participant",
) -> pd.DataFrame:
    rows = {}
    for p in df[group_col].unique():
        sub = df[df[group_col] != p]
        try:
            rows[p] = fit_fn(sub)
        except Exception:
            continue
    out = pd.DataFrame(rows).T
    out.index.name = "dropped"
    return out


def loo_summary(
    df: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], pd.Series],
    full_fit: pd.Series,
    group_col: str = "Participant",
) -> pd.DataFrame:
    loo = leave_one_out(df, fit_fn, group_col=group_col)
    summary = pd.DataFrame(
        {
            "full": full_fit,
            "loo_min": loo.min(),
            "loo_max": loo.max(),
            "loo_range": loo.max() - loo.min(),
            "flips_sign": ((loo * full_fit) < 0).any(),
        }
    )
    return summary
