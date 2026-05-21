"""Simulation-based power analysis.

For each (N participants, days, true within-stress -> compulsion slope)
combination, simulate ``n_sims`` datasets, fit the mixed compulsion
model, and record whether the within-stress coefficient was significant
at p<.05. Power = the proportion of significant fits.

This is the kind of pre-registration calculation you'd want to do
before running the *real* version of this study; producing it from the
same simulator that the analysis was validated on closes the loop.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from ..data.generate import SimConfig, generate_synthetic_data_v2
from ..preprocess import preprocess


@dataclass
class PowerGrid:
    ns: tuple[int, ...] = (10, 20, 30, 50)
    days: tuple[int, ...] = (3, 7, 14)
    betas: tuple[float, ...] = (0.15, 0.25, 0.35)
    n_sims: int = 50
    alpha: float = 0.05
    seed: int = 0


def _one_sim(n: int, days: int, beta: float, seed: int) -> dict[str, float] | None:
    cfg = SimConfig(
        n_participants=n,
        n_days=days,
        seed=seed,
        beta_stress_on_compulsion=beta,
    )
    df = preprocess(generate_synthetic_data_v2(cfg))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(
                "Compulsions ~ Stress_within + Stress_between",
                data=df,
                groups=df["Participant"],
            ).fit(method="lbfgs")
        return {
            "p_within": float(fit.pvalues.get("Stress_within", np.nan)),
            "p_between": float(fit.pvalues.get("Stress_between", np.nan)),
        }
    except Exception:
        return None


def run_power_grid(grid: PowerGrid | None = None) -> pd.DataFrame:
    grid = grid or PowerGrid()
    rng = np.random.default_rng(grid.seed)
    rows = []
    for n in grid.ns:
        for d in grid.days:
            for b in grid.betas:
                hits_w = hits_b = total = 0
                for _ in range(grid.n_sims):
                    p = _one_sim(n, d, b, seed=int(rng.integers(0, 2**31 - 1)))
                    if p is None:
                        continue
                    total += 1
                    if not np.isnan(p["p_within"]) and p["p_within"] < grid.alpha:
                        hits_w += 1
                    if not np.isnan(p["p_between"]) and p["p_between"] < grid.alpha:
                        hits_b += 1
                rows.append(
                    {
                        "N": n,
                        "days": d,
                        "beta": b,
                        "power_within": hits_w / total if total else float("nan"),
                        "power_between": hits_b / total if total else float("nan"),
                        "valid_sims": total,
                    }
                )
    return pd.DataFrame(rows)
