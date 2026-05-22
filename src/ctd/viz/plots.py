"""Visualisations — saved to disk, not shown.

The prototype called ``plt.show()`` everywhere which is fine for a
notebook but useless for a pipeline run. These helpers all take an
``out_path`` and write a PNG.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid", palette="muted")


def _save(fig: plt.Figure, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def circadian_stress(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Average stress by time bin, overlaid by participant + group mean."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for _p, sub in df.groupby("Participant"):
        ax.plot(
            sub.groupby("TimeBin")["Stress"].mean(),
            color="grey",
            alpha=0.25,
            lw=1,
        )
    pooled = df.groupby("TimeBin")["Stress"].mean()
    ax.plot(pooled.index, pooled.values, color="C0", lw=2.5, label="pooled mean")
    ax.set(xlabel="Time bin (0 = early morning)", ylabel="Stress", title="Circadian stress pattern")
    ax.legend()
    return _save(fig, out_path)


def group_stress_over_time(df: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    for g, sub in df.groupby("CompulsionGroup"):
        means = sub.groupby("TimeBin")["Stress"].mean()
        sems = sub.groupby("TimeBin")["Stress"].sem()
        ax.errorbar(means.index, means.values, yerr=sems.values, label=g, capsize=2)
    ax.set(xlabel="Time bin", ylabel="Stress", title="Stress by compulsion group (mean ± SE)")
    ax.legend(title="CompulsionGroup")
    return _save(fig, out_path)


def stress_vs_outside(df: pd.DataFrame, out_path: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="OutsideTime",
        y="Stress",
        hue="CompulsionGroup",
        ax=ax,
        alpha=0.6,
    )
    # Add a per-group regression line for visual aid.
    for _g, sub in df.groupby("CompulsionGroup"):
        if len(sub) > 1:
            z = np.polyfit(sub["OutsideTime"], sub["Stress"], 1)
            xs = np.linspace(sub["OutsideTime"].min(), sub["OutsideTime"].max(), 30)
            ax.plot(xs, np.polyval(z, xs), lw=1.5)
    ax.set_title("Stress vs outside time")
    return _save(fig, out_path)


def coef_forest(
    coef_df: pd.DataFrame,
    out_path: str | Path,
    beta_col: str = "beta",
    lo_col: str = "ci_lo",
    hi_col: str = "ci_hi",
    title: str = "Coefficients (95% CI)",
) -> Path:
    """Forest plot from a coefficient table (e.g. cluster bootstrap output)."""
    df = coef_df.copy()
    # Drop the random-effect variance row if present — it's on a
    # different scale and breaks the layout.
    df = df.drop(index=[i for i in df.index if "Var" in i], errors="ignore")
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(df))))
    y = np.arange(len(df))
    ax.errorbar(
        df[beta_col],
        y,
        xerr=[df[beta_col] - df[lo_col], df[hi_col] - df[beta_col]],
        fmt="o",
        capsize=3,
        color="C0",
    )
    ax.axvline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_yticks(y, df.index)
    ax.set_xlabel("β")
    ax.set_title(title)
    return _save(fig, out_path)


def lag_profile(lag_df: pd.DataFrame, out_path: str | Path, title: str) -> Path:
    """Lag plot from `analysis.lag` output (beta + se columns by lag term)."""
    df = lag_df.copy()
    df = df.drop(index=[i for i in df.index if i in ("Intercept",) or "Var" in i], errors="ignore")
    fig, ax = plt.subplots(figsize=(7, 4))
    y = df["beta"]
    err = 1.96 * df["se"]
    ax.errorbar(np.arange(len(y)), y, yerr=err, fmt="o-", capsize=3)
    ax.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(y)), df.index, rotation=30, ha="right")
    ax.set_ylabel("β (within)")
    ax.set_title(title)
    return _save(fig, out_path)


def power_heatmap(power_df: pd.DataFrame, out_path: str | Path) -> Path:
    """Heatmap of between-person power as a function of N and days, per beta."""
    fig, axes = plt.subplots(
        1, power_df["beta"].nunique(), figsize=(5 * power_df["beta"].nunique(), 4), sharey=True
    )
    if power_df["beta"].nunique() == 1:
        axes = [axes]
    for ax, (b, sub) in zip(axes, power_df.groupby("beta")):
        pivot = sub.pivot(index="N", columns="days", values="power_between")
        sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis", ax=ax)
        ax.set_title(f"β_between = {b}")
        ax.set_xlabel("days")
        ax.set_ylabel("N participants")
    fig.suptitle("Power for the between-person stress→compulsion effect", y=1.02)
    return _save(fig, out_path)
