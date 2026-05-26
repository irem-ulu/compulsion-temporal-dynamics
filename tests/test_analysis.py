import warnings

import pytest

from ctd.analysis.descriptives import correlations, icc, icc_table
from ctd.analysis.lag import (
    add_lag_features,
    stress_to_compulsion_lag,
)
from ctd.analysis.models import (
    compare_ols_vs_mixed,
    mixed_compulsion,
    mixed_stress,
)
from ctd.data.generate import SimConfig, generate_synthetic_data_v2
from ctd.preprocess import preprocess


@pytest.fixture(scope="module")
def df():
    return preprocess(generate_synthetic_data_v2(SimConfig(n_participants=12, n_days=4, seed=5)))


def test_correlations_symmetric(df):
    c = correlations(df)
    assert (c.values.diagonal() == 1.0).all()
    assert (c.values == c.values.T).all()


def test_icc_in_unit_interval(df):
    table = icc_table(df)
    assert ((table["ICC"] >= 0) & (table["ICC"] <= 1)).all()


def test_icc_zero_when_no_between_variance():
    # If everyone has the same mean, ICC should be ~0.
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(0)
    n_people = 5
    n_obs = 30
    rows = []
    for p in range(n_people):
        for _t in range(n_obs):
            rows.append({"Participant": f"P{p}", "x": rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    assert icc(df, "x") < 0.1


def test_mixed_stress_recovers_negative_outside_within(df):
    """DGP: beta_stress_on_outside = -0.30. The reverse coefficient
    (OutsideTime_within on Stress) should be negative."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mixed_stress(df)
    assert fit.params["OutsideTime_within"] < 0


def test_mixed_compulsion_recovers_positive_stress_within(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = mixed_compulsion(df)
    assert fit.params["Stress_within"] > 0


def test_ols_vs_mixed_runs(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = compare_ols_vs_mixed(df)
    assert {"ols_beta", "mixed_beta", "se_ratio"}.issubset(cmp.columns)


def test_add_lag_features_creates_lags(df):
    d = add_lag_features(df, max_lag=2)
    assert "Stress_lag1" in d.columns
    assert "Stress_lag2" in d.columns
    # First observation per participant should be NaN for lag1.
    first = d.groupby("Participant").head(1)
    assert first["Stress_lag1"].isna().all()


def test_stress_to_compulsion_lag_runs(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tbl = stress_to_compulsion_lag(df, max_lag=2)
    assert "beta" in tbl.columns
