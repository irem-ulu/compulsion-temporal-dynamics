# Changelog

All notable changes to this project will be documented here. Format
inspired by [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.1.0] — 2026-05-29

### Added
- Synthetic EMA data generators (legacy + v2 with person-level random effects).
- Preprocessing: cleaning, time features, within/between decomposition.
- Mixed-effects models, multi-lag dynamics, cluster bootstrap, sensitivity, power grid.
- Saved visualisations, DAG renderer, end-to-end pipeline, typer CLI (`ctd`).
- Auto-generated `REPORT.md`, legacy OLS-vs-Mixed comparison, Streamlit dashboard.
- pytest suite, GitHub Actions CI, pre-commit, Makefile.
- `ctd describe`, `--version`, and `python -m ctd` entry points.
- Checked-in sample dataset for reproducible demos.

### Changed
- Pooled OLS replaced with mixed-effects models throughout the pipeline.
- Visualisations save to `results/figures/` instead of interactive display.
