"""Mixed-effects regression models — the core of the reanalysis.

The prototype fit four OLS models on pooled data:

    Stress       ~ OutsideTime + Compulsions
    Compulsions  ~ Stress + OutsideTime
    Stress       ~ OutsideTime * TimeBin * CompulsionGroup
    Compulsions  ~ PrevStress

Treating 36 observations per person as 36 independent rows is the
mistake that this module fixes. Each model below has a random intercept
for ``Participant``, and the time-varying predictors are split into
within- and between-person components so the slope you get for
``Stress_within`` is the actual within-person effect (the one that
matters for "when *this person* feels stressed, do *they* compulse?").

statsmodels' ``MixedLM`` is the workhorse. It's not lme4 — REML only,
no glmer-style binomial/Poisson families out of the box — but for the
3 continuous-ish outcomes here it's the right tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class FitResult:
    name: str
    formula: str
    summary: str
    params: pd.Series
    bse: pd.Series
    pvalues: pd.Series
    aic: float | None = None
    bic: float | None = None
    n_obs: int = 0
    n_groups: int | None = None

    @classmethod
    def from_ols(cls, name: str, formula: str, fit: Any) -> FitResult:
        return cls(
            name=name,
            formula=formula,
            summary=str(fit.summary()),
            params=fit.params,
            bse=fit.bse,
            pvalues=fit.pvalues,
            aic=float(fit.aic),
            bic=float(fit.bic),
            n_obs=int(fit.nobs),
            n_groups=None,
        )

    @classmethod
    def from_mixed(cls, name: str, formula: str, fit: Any) -> FitResult:
        # MixedLM has no AIC/BIC by default (REML); compute via -2*ll if possible.
        try:
            aic = float(fit.aic)
        except Exception:
            aic = None
        try:
            bic = float(fit.bic)
        except Exception:
            bic = None
        return cls(
            name=name,
            formula=formula,
            summary=str(fit.summary()),
            params=fit.params,
            bse=fit.bse,
            pvalues=fit.pvalues,
            aic=aic,
            bic=bic,
            n_obs=int(fit.nobs),
            n_groups=int(fit.model.n_groups),
        )


# ─── Naive OLS — replicates what the prototype was doing ──────────────────


def ols_stress(df: pd.DataFrame) -> FitResult:
    f = "Stress ~ OutsideTime + Compulsions"
    return FitResult.from_ols("ols_stress", f, smf.ols(f, data=df).fit())


def ols_compulsion(df: pd.DataFrame) -> FitResult:
    f = "Compulsions ~ Stress + OutsideTime"
    return FitResult.from_ols("ols_compulsion", f, smf.ols(f, data=df).fit())


# ─── Mixed-effects equivalents ────────────────────────────────────────────


def mixed_stress(df: pd.DataFrame) -> FitResult:
    """Stress ~ within-person OT + between-person OT + compulsions, RI per ppl.

    Splitting OutsideTime/Compulsions into within/between lets us read the
    coefficients separately. ``OutsideTime_within`` answers
    "on a given moment when *I* go outside more than usual, am *I* less
    stressed?" — the actually causal-relevant question.
    """
    f = (
        "Stress ~ OutsideTime_within + OutsideTime_between "
        "+ Compulsions_within + Compulsions_between"
    )
    fit = smf.mixedlm(f, data=df, groups=df["Participant"]).fit(method="lbfgs")
    return FitResult.from_mixed("mixed_stress", f, fit)


def mixed_compulsion(df: pd.DataFrame) -> FitResult:
    f = (
        "Compulsions ~ Stress_within + Stress_between "
        "+ OutsideTime_within + OutsideTime_between"
    )
    fit = smf.mixedlm(f, data=df, groups=df["Participant"]).fit(method="lbfgs")
    return FitResult.from_mixed("mixed_compulsion", f, fit)


# ─── Interaction model — modernised ───────────────────────────────────────


def mixed_interaction(df: pd.DataFrame) -> FitResult:
    """Stress ~ within OT × HourOfDay × CompulsionGroup, RI per ppl.

    The prototype used the integer TimeBin which assumes a linear time
    effect, and 3-way TimeBin × OutsideTime × group on the pooled data.
    Switching to HourOfDay (same info, real-world units) and putting it
    inside a mixed model makes the coefficients interpretable as effects
    *within* a person at a given time of day.
    """
    f = (
        "Stress ~ OutsideTime_within * HourOfDay * CompulsionGroup "
        "+ OutsideTime_between"
    )
    fit = smf.mixedlm(f, data=df, groups=df["Participant"]).fit(method="lbfgs")
    return FitResult.from_mixed("mixed_interaction", f, fit)


# ─── Side-by-side comparison ──────────────────────────────────────────────


def compare_ols_vs_mixed(df: pd.DataFrame) -> pd.DataFrame:
    """Quick table that shows how slopes shift once we respect nesting.

    For each shared coefficient, prints the OLS estimate, the mixed
    estimate, and their SEs. The interesting cases are the rows where
    the OLS SE was anti-conservative (too small) because it pretended
    all 360 rows were independent.
    """
    ols_s = ols_stress(df)
    mix_s = mixed_stress(df)
    ols_c = ols_compulsion(df)
    mix_c = mixed_compulsion(df)

    rows = []

    def add(model_label, ols_fit, mix_fit, ols_term, mix_term):
        rows.append(
            {
                "model": model_label,
                "term": ols_term,
                "ols_beta": ols_fit.params.get(ols_term, float("nan")),
                "ols_se": ols_fit.bse.get(ols_term, float("nan")),
                "mixed_term": mix_term,
                "mixed_beta": mix_fit.params.get(mix_term, float("nan")),
                "mixed_se": mix_fit.bse.get(mix_term, float("nan")),
            }
        )

    add("stress", ols_s, mix_s, "OutsideTime", "OutsideTime_within")
    add("stress", ols_s, mix_s, "Compulsions", "Compulsions_within")
    add("compulsion", ols_c, mix_c, "Stress", "Stress_within")
    add("compulsion", ols_c, mix_c, "OutsideTime", "OutsideTime_within")

    out = pd.DataFrame(rows)
    out["se_ratio"] = out["mixed_se"] / out["ols_se"]
    return out
