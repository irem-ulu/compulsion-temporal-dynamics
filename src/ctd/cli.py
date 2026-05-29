"""Typer CLI — entry point for the project.

``ctd simulate`` writes raw data only.
``ctd run`` is the full reanalysis pipeline.
``ctd power`` runs just the power grid.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich import print as rprint

from . import __version__
from .data.generate import SimConfig, generate_synthetic_data_v2
from .data.io import save_raw_data
from .logging import get_logger
from .pipeline import PipelineConfig
from .pipeline import run as run_pipeline

app = typer.Typer(no_args_is_help=True, help="compulsion-temporal-dynamics CLI")
log = get_logger()


def _version_cb(value: bool) -> None:
    if value:
        rprint(f"ctd {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_cb,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """compulsion-temporal-dynamics CLI."""


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
    rprint(
        f"[green]wrote[/green] {path}  "
        f"({df.shape[0]} rows, {df['Participant'].nunique()} participants)"
    )


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
    report_path = artefacts.get("report")
    if report_path is not None:
        rprint(f"[cyan]open[/cyan] {report_path} to see the assembled writeup")


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


@app.command()
def report(
    results: Path = typer.Option(Path("results")),
) -> None:
    """Re-assemble REPORT.md from existing results/ artefacts."""
    from .report import build_report

    path = build_report(results)
    rprint(f"[green]wrote[/green] {path}")


@app.command()
def describe(
    data: Path = typer.Option(
        Path("data/raw_data.csv"),
        help="Path to a CSV with the expected EMA columns.",
    ),
) -> None:
    """Print pooled correlations and ICCs for an existing data file.

    Quick way to check whether your data has enough between-person
    variance to justify the multilevel machinery, without running the
    whole pipeline.
    """
    if not data.exists():
        rprint(f"[red]no such file:[/red] {data}")
        raise typer.Exit(code=1)

    import pandas as pd

    from .analysis.descriptives import correlations, icc_table
    from .preprocess import clean

    df = clean(pd.read_csv(data))
    rprint(f"[bold]rows[/bold]: {len(df)}  "
           f"[bold]participants[/bold]: {df['Participant'].nunique()}")
    rprint("\n[bold]Pooled correlations[/bold]")
    rprint(correlations(df).round(3))
    rprint("\n[bold]ICCs (variance attributable to person)[/bold]")
    rprint(icc_table(df).round(3))


@app.command()
def compare(
    seed: int = typer.Option(42),
    out: Path = typer.Option(Path("results/legacy_vs_new.txt")),
) -> None:
    """Run prototype OLS and new mixed-effects on the same legacy dataset."""
    from .legacy import compare_on_legacy_data, render_comparison

    res = compare_on_legacy_data(seed=seed)
    text = render_comparison(res)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    rprint(text)
    rprint(f"[green]wrote[/green] {out}")


if __name__ == "__main__":
    app()
