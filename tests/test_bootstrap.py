"""Tests for the cluster + row bootstrap helpers."""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from ctd.analysis.bootstrap import cluster_bootstrap, row_bootstrap
from ctd.data.generate import SimConfig, generate_synthetic_data_v2
from ctd.preprocess import preprocess


@pytest.fixture(scope="module")
def df():
    return preprocess(generate_synthetic_data_v2(SimConfig(n_participants=10, n_days=4, seed=5)))


def _fit_simple(d: pd.DataFrame) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = smf.mixedlm(
            "Compulsions ~ Stress_within + Stress_between",
            data=d,
            groups=d["Participant"],
        ).fit(method="lbfgs")
    return m.params


def test_cluster_bootstrap_returns_expected_columns(df):
    out = cluster_bootstrap(df, _fit_simple, n_iter=20, seed=0)
    assert set(out.columns) == {"beta_mean", "beta_se", "ci_lo", "ci_hi"}
    # Should contain the model's fixed-effect names (intercept + two slopes).
    for col in ("Intercept", "Stress_within", "Stress_between"):
        assert col in out.index


def test_cluster_bootstrap_ci_brackets_mean(df):
    out = cluster_bootstrap(df, _fit_simple, n_iter=30, seed=1)
    # 95% CI quantiles must surround the bootstrap mean.
    assert (out["ci_lo"] <= out["beta_mean"]).all()
    assert (out["beta_mean"] <= out["ci_hi"]).all()


def test_cluster_bootstrap_deterministic_for_same_seed(df):
    a = cluster_bootstrap(df, _fit_simple, n_iter=15, seed=42)
    b = cluster_bootstrap(df, _fit_simple, n_iter=15, seed=42)
    assert np.allclose(a["beta_mean"].values, b["beta_mean"].values)


def test_cluster_bootstrap_se_exceeds_row_bootstrap_se(df):
    """Cluster resampling preserves within-person dependence and therefore
    should usually give a larger SE than the naive row bootstrap. We check
    the intercept, where the gap is most reliable."""
    cl = cluster_bootstrap(df, _fit_simple, n_iter=60, seed=7)
    rw = row_bootstrap(df, _fit_simple, n_iter=60, seed=7)
    assert cl.loc["Intercept", "beta_se"] > rw.loc["Intercept", "beta_se"]
