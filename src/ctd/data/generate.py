"""Synthetic EMA data generation.

Two generators live here:

- :func:`generate_synthetic_data` — the original prototype's generator,
  preserved bit-for-bit (well, RNG-equivalent) so the legacy results are
  reproducible.

- :func:`generate_synthetic_data_v2` — what we actually use for the
  reanalysis. The differences matter for the kind of inference we want
  to do afterwards:

  1. **Person-level random intercepts** for stress, outside time and
     compulsions. The prototype only varied the base stress; here, every
     participant has their own trait level for each outcome plus a shared
     correlated component (anxious people stay inside more *and*
     compulse more).
  2. **AR(1) within-person stress dynamics** — your stress at 11am is
     correlated with your stress at 10am. The prototype assumed
     independence which is not how EMA data behave.
  3. **Explicit causal structure** — see ``CAUSAL_DAG`` below. Stress
     causes outside-time (negatively, with random per-person slope) and
     compulsions; outside-time also has a small negative effect on
     compulsions on its own. Lagged stress contributes to current
     compulsions (the relationship the lag model is meant to find).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = ["Participant", "Day", "TimeBin", "Stress", "OutsideTime", "Compulsions"]


# Edges in the data-generating DAG. Useful for documentation and for the
# `viz.dag` plot — keeps the simulator and the picture in sync.
CAUSAL_DAG: list[tuple[str, str, str]] = [
    ("circadian", "Stress", "+"),
    ("PersonStress", "Stress", "+"),
    ("Stress_{t-1}", "Stress", "+"),  # AR(1)
    ("Stress", "OutsideTime", "-"),
    ("PersonOutside", "OutsideTime", "+"),
    ("Stress", "Compulsions", "+"),
    ("Stress_{t-1}", "Compulsions", "+"),
    ("OutsideTime", "Compulsions", "-"),
    ("PersonCompulsion", "Compulsions", "+"),
]


# ─── Original prototype generator ─────────────────────────────────────────


def generate_synthetic_data(
    n_participants: int = 10,
    n_days: int = 3,
    n_bins: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """The prototype generator — preserved for reproducibility."""
    rng = np.random.default_rng(seed)

    participants = [f"P{i}" for i in range(1, n_participants + 1)]
    rows = []

    for p in participants:
        base_stress = rng.integers(2, 6)
        for d in range(n_days):
            for b in range(n_bins):
                circadian = np.sin(b / n_bins * 2 * np.pi) + 1
                stress = base_stress + circadian + rng.normal(0, 0.5)
                outside = max(0.0, rng.normal(1.5 - stress * 0.2, 0.5))
                compulsion = int(max(0, rng.poisson(stress / 2)))
                rows.append([p, d, b, stress, outside, compulsion])

    return pd.DataFrame(rows, columns=COLUMNS)


# ─── v2: richer EMA-like generator ────────────────────────────────────────


@dataclass
class SimConfig:
    n_participants: int = 30
    n_days: int = 7
    n_bins: int = 12
    seed: int = 42

    # Person-level random effects (SDs).
    sd_stress_intercept: float = 0.8
    sd_outside_intercept: float = 0.6
    sd_compulsion_intercept: float = 0.5
    # Correlated person-level latent "anxious-trait": pushes stress up,
    # outside-time down, compulsions up.
    sd_trait: float = 0.7

    # Within-person dynamics.
    ar1_stress: float = 0.4
    sd_stress_within: float = 0.6

    # Slopes — these are what the models should recover.
    beta_stress_on_outside: float = -0.30
    beta_outside_on_compulsion: float = -0.20
    beta_stress_on_compulsion: float = 0.35
    beta_lag_stress_on_compulsion: float = 0.15

    # Stress circadian amplitude.
    circadian_amplitude: float = 0.7


def generate_synthetic_data_v2(config: SimConfig | None = None) -> pd.DataFrame:
    """Generate EMA data from an explicit multilevel DGP.

    See :data:`CAUSAL_DAG` for the structural picture.
    """
    cfg = config or SimConfig()
    rng = np.random.default_rng(cfg.seed)
    participants = [f"P{i:02d}" for i in range(1, cfg.n_participants + 1)]

    # Person-level latent trait — drives all three outcomes.
    trait = rng.normal(0, cfg.sd_trait, size=cfg.n_participants)
    stress_intercept = 3.5 + trait + rng.normal(0, cfg.sd_stress_intercept, size=cfg.n_participants)
    outside_intercept = 1.5 - 0.4 * trait + rng.normal(0, cfg.sd_outside_intercept, size=cfg.n_participants)
    comp_intercept = -0.3 + 0.5 * trait + rng.normal(0, cfg.sd_compulsion_intercept, size=cfg.n_participants)

    rows = []

    for pi, p in enumerate(participants):
        prev_stress = stress_intercept[pi]  # initialise AR(1)
        prev_stress_for_lag = prev_stress

        for d in range(cfg.n_days):
            for b in range(cfg.n_bins):
                circadian = cfg.circadian_amplitude * np.sin(2 * np.pi * b / cfg.n_bins)
                stress_mean = (
                    stress_intercept[pi]
                    + circadian
                    + cfg.ar1_stress * (prev_stress - stress_intercept[pi])
                )
                stress = stress_mean + rng.normal(0, cfg.sd_stress_within)

                outside_mean = (
                    outside_intercept[pi] + cfg.beta_stress_on_outside * (stress - stress_intercept[pi])
                )
                outside = max(0.0, rng.normal(outside_mean, 0.4))

                # Compulsions: Poisson with stress + lag-stress + outside.
                log_rate = (
                    comp_intercept[pi]
                    + cfg.beta_stress_on_compulsion * (stress - stress_intercept[pi])
                    + cfg.beta_lag_stress_on_compulsion * (prev_stress_for_lag - stress_intercept[pi])
                    + cfg.beta_outside_on_compulsion * (outside - outside_intercept[pi])
                )
                rate = float(np.exp(log_rate))
                rate = min(rate, 25.0)  # cap for stability
                compulsion = int(rng.poisson(rate))

                rows.append([p, d, b, stress, outside, compulsion])

                prev_stress_for_lag = prev_stress
                prev_stress = stress

    return pd.DataFrame(rows, columns=COLUMNS)
