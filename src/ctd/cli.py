"""Typer CLI — entry point for the project.

``ctd simulate`` writes raw data only.
``ctd run`` is the full reanalysis pipeline.
``ctd power`` runs just the power grid.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from .data.generate import SimConfig, generate_synthetic_data_v2
from .data.io import save_raw_data
from .logging import get_logger
from .pipeline import PipelineConfig, run as run_pipeline


app = typer.Typer(no_args_is_help=True, help="compulsion-temporal-dynamics CLI")
log = get_logger()


@app.command()
def simulate(
    n: int = typer.Option(30, help="Participants"),
    days: int = typer.Option(7, help="Days per participant"),
    bins: int = typer.Option(12, help="Time bins per day"),
    seed: int = typer.Option(42),
    out: Path = typer.Option(Path("data/raw_data.csv")),
) -> None:
    """Generate the v2 synthetic dataset and save it."""
    df = generate_synthetic_data_v2(
        SimConfig(n_participants=n, n_days=days, n_bins=bins, seed=seed)
    )
    path = save_raw_data(df, out)
    rprint(f"[green]wrote[/green] {path}  ({df.shape[0]} rows, {df['Participant'].nunique()} ppl)")


@app.command()
def run(
    n: int = typer.Option(30, help="Participants"),
    days: int = typer.Option(7, help="Days per participant"),
    bins: int = typer.Option(12, help="Time bins per day"),
    seed: int = typer.Option(42),
    boot: int = typer.Option(300, help="Cluster-bootstrap iterations"),
    power: bool = typer.Option(False, "--power", help="Also run the power grid"),
    results: Path = typer.Option(Path("results")),
) -> None:
    """Run the full reanalysis pipeline."""
    cfg = PipelineConfig(
        sim=SimConfig(n_participants=n, n_days=days, n_bins=bins, seed=seed),
        bootstrap_iters=boot,
        run_power=power,
        results_dir=results,
    )
    artefacts = run_pipeline(cfg)
    rprint(f"[green]wrote {len(artefacts)} artefacts to {results}/[/green]")


@app.command()
def power(
    n_sims: int = typer.Option(30, help="Sims per cell"),
    out: Path = typer.Option(Path("results/power_grid.txt")),
) -> None:
    """Run only the power grid (slow)."""
    from .analysis.power import PowerGrid, run_power_grid

    grid = run_power_grid(PowerGrid(n_sims=n_sims))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(grid.round(3).to_string())
    rprint(grid)


if __name__ == "__main__":
    app()
