"""Recreates the prototype's exact analysis side-by-side with the new one.

We re-run the legacy four OLS models on the legacy data, run the new
mixed-effects equivalents on the same data, and print the headline
deltas. That's the most honest way to show what changed:

- which coefficients survive?
- which standard errors blow up once you stop pretending observations
  are independent?
- which become non-significant, or change sign?

Run with ``ctd compare`` after installing the package.
"""

from __future__ import annotations

import warnings

import pandas as pd
import statsmodels.formula.api as smf

from .data.generate import generate_synthetic_data
from .preprocess import (
    add_time_features,
    add_trait_features,
    assign_compulsion_groups,
    clean,
    within_between_decompose,
)


def _legacy_ols(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Recreates the four OLS fits from the prototype's main.py."""
    fits = {
        "stress_model": smf.ols("Stress ~ OutsideTime + Compulsions", data=df).fit(),
        "compulsion_model": smf.ols("Compulsions ~ Stress + OutsideTime", data=df).fit(),
        "interaction_model": smf.ols(
            "Stress ~ OutsideTime * TimeBin * CompulsionGroup", data=df
        ).fit(),
    }
    df_lag = df.copy()
    df_lag["PrevStress"] = df_lag.groupby("Participant")["Stress"].shift(1)
    df_lag = df_lag.dropna(subset=["PrevStress"])
    fits["lag_model"] = smf.ols("Compulsions ~ PrevStress", data=df_lag).fit()

    return {
        name: pd.DataFrame(
            {"beta": fit.params, "se": fit.bse, "p": fit.pvalues}
        )
        for name, fit in fits.items()
    }


def _new_mixed(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Equivalent mixed-effects fits on the same data."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stress = smf.mixedlm(
            "Stress ~ OutsideTime_within + OutsideTime_between "
            "+ Compulsions_within + Compulsions_between",
            data=df,
            groups=df["Participant"],
        ).fit(method="lbfgs")

        compulsion = smf.mixedlm(
            "Compulsions ~ Stress_within + Stress_between "
            "+ OutsideTime_within + OutsideTime_between",
            data=df,
            groups=df["Participant"],
        ).fit(method="lbfgs")

        interaction = smf.mixedlm(
            "Stress ~ OutsideTime_within * TimeBin * CompulsionGroup "
            "+ OutsideTime_between",
            data=df,
            groups=df["Participant"],
        ).fit(method="lbfgs")

        df_lag = df.copy()
        df_lag["PrevStress_within"] = df_lag.groupby("Participant")["Stress_within"].shift(1)
        df_lag = df_lag.dropna(subset=["PrevStress_within"])
        lag = smf.mixedlm(
            "Compulsions ~ Stress_within + PrevStress_within",
            data=df_lag,
            groups=df_lag["Participant"],
        ).fit(method="lbfgs")

    return {
        "stress_model": pd.DataFrame(
            {"beta": stress.params, "se": stress.bse, "p": stress.pvalues}
        ),
        "compulsion_model": pd.DataFrame(
            {"beta": compulsion.params, "se": compulsion.bse, "p": compulsion.pvalues}
        ),
        "interaction_model": pd.DataFrame(
            {"beta": interaction.params, "se": interaction.bse, "p": interaction.pvalues}
        ),
        "lag_model": pd.DataFrame(
            {"beta": lag.params, "se": lag.bse, "p": lag.pvalues}
        ),
    }


def compare_on_legacy_data(seed: int = 42) -> dict[str, pd.DataFrame]:
    """Run both pipelines on the legacy generator's data."""
    raw = generate_synthetic_data(n_participants=10, n_days=3, n_bins=12, seed=seed)
    df = clean(raw)
    df = add_time_features(df)
    df = assign_compulsion_groups(df)
    df = add_trait_features(df)
    df = within_between_decompose(df, ["Stress", "OutsideTime", "Compulsions"])

    return {"ols": _legacy_ols(df), "mixed": _new_mixed(df)}


def render_comparison(results: dict[str, dict[str, pd.DataFrame]]) -> str:
    """Render the comparison to a single human-readable string."""
    lines = ["Prototype OLS vs reanalysis MixedLM on the legacy 10×3×12 dataset", "=" * 70, ""]
    for model in ("stress_model", "compulsion_model", "interaction_model", "lag_model"):
        lines.append(f"### {model}")
        lines.append("OLS:")
        lines.append(results["ols"][model].round(3).to_string())
        lines.append("")
        lines.append("Mixed:")
        lines.append(results["mixed"][model].round(3).to_string())
        lines.append("")
        lines.append("")
    return "\n".join(lines)
