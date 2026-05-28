"""Streamlit dashboard for interactive exploration.

Run with:
    streamlit run dashboard/app.py

Lets you slide the simulator's parameters around and see the fitted
mixed-effects coefficients move with them. The point isn't a polished
product — it's to make the analytical pipeline tactile so a reviewer
can see "ah, when I dial up sd_trait the between-person effect gets
fatter, exactly like you'd expect."
"""

from __future__ import annotations

import warnings

import pandas as pd
import statsmodels.formula.api as smf
import streamlit as st

from ctd.data.generate import SimConfig, generate_synthetic_data_v2
from ctd.preprocess import preprocess

st.set_page_config(page_title="ctd — interactive reanalysis", layout="wide")
st.title("Digital phenotyping mock — interactive reanalysis")
st.caption(
    "Move the sliders. The mixed-effects coefficients refit. The point is to "
    "build intuition about how within- vs between-person decomposition "
    "behaves as you change the data-generating parameters."
)


# ─── Sidebar: sim controls ────────────────────────────────────────────────
with st.sidebar:
    st.header("Simulator")
    n = st.slider("participants", 5, 100, 30, step=5)
    days = st.slider("days", 1, 21, 7)
    bins = st.slider("bins / day", 4, 24, 12)
    seed = st.number_input("seed", 0, 10_000, 42)

    st.subheader("True effects")
    b_so = st.slider("stress → outside", -1.0, 1.0, -0.30, step=0.05)
    b_sc = st.slider("stress → compulsion", -1.0, 1.0, 0.35, step=0.05)
    b_oc = st.slider("outside → compulsion", -1.0, 1.0, -0.20, step=0.05)
    b_lag = st.slider("lag-stress → compulsion", -1.0, 1.0, 0.15, step=0.05)
    ar1 = st.slider("AR(1) stress", 0.0, 0.95, 0.4, step=0.05)
    trait = st.slider("trait SD (between-person)", 0.0, 2.0, 0.7, step=0.1)


cfg = SimConfig(
    n_participants=n,
    n_days=days,
    n_bins=bins,
    seed=int(seed),
    beta_stress_on_outside=b_so,
    beta_stress_on_compulsion=b_sc,
    beta_outside_on_compulsion=b_oc,
    beta_lag_stress_on_compulsion=b_lag,
    ar1_stress=ar1,
    sd_trait=trait,
)


@st.cache_data(show_spinner=False)
def simulate(cfg_dict: dict) -> pd.DataFrame:
    cfg = SimConfig(**cfg_dict)
    df = generate_synthetic_data_v2(cfg)
    return preprocess(df, n_bins=cfg.n_bins)


df = simulate(cfg.__dict__)


# ─── Data preview ─────────────────────────────────────────────────────────
left, right = st.columns(2)
with left:
    st.subheader("Sample")
    st.dataframe(df.head(20), use_container_width=True, height=300)
with right:
    st.subheader("Per-person means")
    st.dataframe(
        df.groupby("Participant")[["Stress", "OutsideTime", "Compulsions"]]
        .mean()
        .round(2),
        use_container_width=True,
        height=300,
    )


# ─── Plots ────────────────────────────────────────────────────────────────
st.subheader("Stress over time")
st.line_chart(
    df.groupby(["TimeBin", "CompulsionGroup"])["Stress"]
    .mean()
    .unstack("CompulsionGroup"),
    height=320,
)


# ─── Mixed-effects fits ───────────────────────────────────────────────────
st.subheader("Mixed-effects estimates (refit on every change)")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stress_fit = smf.mixedlm(
        "Stress ~ OutsideTime_within + OutsideTime_between "
        "+ Compulsions_within + Compulsions_between",
        data=df,
        groups=df["Participant"],
    ).fit(method="lbfgs")
    comp_fit = smf.mixedlm(
        "Compulsions ~ Stress_within + Stress_between "
        "+ OutsideTime_within + OutsideTime_between",
        data=df,
        groups=df["Participant"],
    ).fit(method="lbfgs")


def coef_table(fit) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "β": fit.params,
            "SE": fit.bse,
            "z": fit.tvalues,
            "p": fit.pvalues,
        }
    ).round(3)


left, right = st.columns(2)
with left:
    st.markdown("**Stress model**")
    st.dataframe(coef_table(stress_fit), use_container_width=True)
with right:
    st.markdown("**Compulsion model**")
    st.dataframe(coef_table(comp_fit), use_container_width=True)


# ─── Helpful annotations ──────────────────────────────────────────────────
st.markdown(
    """
    **What to look for**
    - `*_within` coefficients should recover the slopes you set on the
      sliders (modulo the Poisson link being log-linear).
    - The `*_between` coefficients can have completely different signs
      from `*_within` when the trait correlation is strong — that's the
      Simpson-like effect a pooled OLS would have smeared away.
    - Crank `trait SD` up: between-person effects strengthen,
      within-person ones stay put.
    - Drop `participants` to 5: the between-person SE explodes; within-
      person stays narrow because the per-person time series is still long.
    """
)
