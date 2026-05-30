"""End-to-end analysis pipeline.

Glues data → preprocess → models → bootstrap → power → plots together
and writes everything under ``results/``. The CLI calls into this
directly so ``python -m ctd.pipeline`` and ``ctd run`` produce the same
artefacts.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

from .analysis.bootstrap import cluster_bootstrap
from .analysis.descriptives import (
    correlations,
    icc_table,
    within_between_correlations,
)
from .analysis.lag import compulsion_to_stress_lag, stress_to_compulsion_lag
from .analysis.models import (
    compare_ols_vs_mixed,
    mixed_compulsion,
    mixed_interaction,
    mixed_stress,
)
from .analysis.sensitivity import loo_summary
from .data.generate import SimConfig, generate_synthetic_data_v2
from .data.io import save_raw_data
from .logging import get_logger
from .preprocess import preprocess
from .report import build_report
from .viz.dag import render_dag
from .viz.plots import (
    circadian_stress,
    coef_forest,
    group_stress_over_time,
    lag_profile,
    power_heatmap,
    stress_vs_outside,
)

log = get_logger()


@dataclass
class PipelineConfig:
    sim: SimConfig = None  # type: ignore[assignment]
    data_path: Path = Path("data/raw_data.csv")
    results_dir: Path = Path("results")
    bootstrap_iters: int = 300
    run_power: bool = False
    power_n_sims: int = 30

    def __post_init__(self) -> None:
        if self.sim is None:
            self.sim = SimConfig()
        self.data_path = Path(self.data_path)
        self.results_dir = Path(self.results_dir)


def _save_text(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def run(config: PipelineConfig | None = None) -> dict[str, Path]:
    cfg = config or PipelineConfig()
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = cfg.results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    artefacts: dict[str, Path] = {}

    log.info("Simulating dataset: N=%d, days=%d, bins=%d",
             cfg.sim.n_participants, cfg.sim.n_days, cfg.sim.n_bins)
    raw = generate_synthetic_data_v2(cfg.sim)
    artefacts["raw"] = save_raw_data(raw, cfg.data_path)

    df = preprocess(raw, n_bins=cfg.sim.n_bins)

    log.info("Descriptives")
    corr = correlations(df)
    icc = icc_table(df)
    wb = within_between_correlations(df)
    artefacts["corr"] = _save_text(corr.round(3).to_string(), cfg.results_dir / "correlations.txt")
    artefacts["icc"] = _save_text(icc.round(3).to_string(), cfg.results_dir / "icc.txt")
    artefacts["wb_corr"] = _save_text(
        "BETWEEN-PERSON:\n" + wb["between"].round(3).to_string()
        + "\n\nWITHIN-PERSON:\n" + wb["within"].round(3).to_string(),
        cfg.results_dir / "within_between_correlations.txt",
    )

    # Two summary tables the prototype saved — kept here as CSVs because
    # they're often handy as inputs to downstream plotting / reporting.
    time_series = df.groupby("TimeBin")["Stress"].mean().rename("Stress_mean").to_frame()
    time_series.to_csv(cfg.results_dir / "time_series.csv")
    artefacts["time_series"] = cfg.results_dir / "time_series.csv"
    group_time = df.groupby(["CompulsionGroup", "TimeBin"])["Stress"].mean().unstack()
    group_time.to_csv(cfg.results_dir / "group_time_series.csv")
    artefacts["group_time_series"] = cfg.results_dir / "group_time_series.csv"

    log.info("Mixed-effects models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = mixed_stress(df)
        mc = mixed_compulsion(df)
        mi = mixed_interaction(df)
        cmp = compare_ols_vs_mixed(df)

    artefacts["mixed_stress"] = _save_text(ms.summary, cfg.results_dir / "mixed_stress.txt")
    artefacts["mixed_compulsion"] = _save_text(mc.summary, cfg.results_dir / "mixed_compulsion.txt")
    artefacts["mixed_interaction"] = _save_text(mi.summary, cfg.results_dir / "mixed_interaction.txt")
    artefacts["ols_vs_mixed"] = _save_text(
        cmp.round(3).to_string(), cfg.results_dir / "ols_vs_mixed.txt"
    )

    log.info("Lag models")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lag_fwd = stress_to_compulsion_lag(df)
        lag_rev = compulsion_to_stress_lag(df)
    artefacts["lag_fwd"] = _save_text(
        lag_fwd.round(3).to_string(),
        cfg.results_dir / "lag_stress_to_compulsion.txt",
    )
    artefacts["lag_rev"] = _save_text(
        lag_rev.round(3).to_string(),
        cfg.results_dir / "lag_compulsion_to_stress.txt",
    )

    log.info("Cluster bootstrap on the compulsion model (%d iters)", cfg.bootstrap_iters)

    def fit_compulsion(d: pd.DataFrame) -> pd.Series:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = smf.mixedlm(
                "Compulsions ~ Stress_within + Stress_between "
                "+ OutsideTime_within + OutsideTime_between",
                data=d,
                groups=d["Participant"],
            ).fit(method="lbfgs")
        return f.params

    boot = cluster_bootstrap(
        df, fit_compulsion, n_iter=cfg.bootstrap_iters, seed=cfg.sim.seed
    )
    artefacts["bootstrap"] = _save_text(
        boot.round(3).to_string(), cfg.results_dir / "bootstrap_compulsion.txt"
    )

    log.info("Leave-one-out sensitivity")
    full = mc.params
    loo = loo_summary(df, fit_compulsion, full)
    artefacts["loo"] = _save_text(loo.round(3).to_string(), cfg.results_dir / "loo_compulsion.txt")

    log.info("Figures")
    artefacts["fig_dag"] = render_dag(fig_dir / "dag.png")
    artefacts["fig_circadian"] = circadian_stress(df, fig_dir / "circadian_stress.png")
    artefacts["fig_group"] = group_stress_over_time(df, fig_dir / "group_stress.png")
    artefacts["fig_scatter"] = stress_vs_outside(df, fig_dir / "stress_vs_outside.png")

    artefacts["fig_forest"] = coef_forest(
        boot,
        fig_dir / "compulsion_forest.png",
        beta_col="beta_mean",
        title="Compulsion model coefficients (cluster bootstrap 95% CI)",
    )

    artefacts["fig_lag_fwd"] = lag_profile(
        lag_fwd, fig_dir / "lag_fwd.png", "Stress -> compulsions (mixed model)"
    )
    artefacts["fig_lag_rev"] = lag_profile(
        lag_rev, fig_dir / "lag_rev.png", "Compulsions -> stress (mixed model)"
    )

    if cfg.run_power:
        log.info("Power analysis (this is slow)")
        from .analysis.power import PowerGrid, run_power_grid

        grid = run_power_grid(PowerGrid(n_sims=cfg.power_n_sims))
        artefacts["power"] = _save_text(
            grid.round(3).to_string(), cfg.results_dir / "power_grid.txt"
        )
        artefacts["fig_power"] = power_heatmap(grid, fig_dir / "power.png")

    log.info("Building markdown report")
    artefacts["report"] = build_report(cfg.results_dir, data_path=cfg.data_path)

    log.info("Done. Wrote %d artefacts to %s", len(artefacts), cfg.results_dir)
    return artefacts


if __name__ == "__main__":
    run()
