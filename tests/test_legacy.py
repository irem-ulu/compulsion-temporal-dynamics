"""Sanity tests for the legacy-vs-new comparison."""

import warnings

from ctd.legacy import _legacy_ols, compare_on_legacy_data, render_comparison

EXPECTED_MODELS = {"stress_model", "compulsion_model", "interaction_model", "lag_model"}


def test_legacy_ols_returns_expected_models():
    res = compare_on_legacy_data(seed=42)
    assert set(res["ols"].keys()) == EXPECTED_MODELS
    assert set(res["mixed"].keys()) == EXPECTED_MODELS


def test_render_comparison_includes_each_model():
    res = compare_on_legacy_data(seed=42)
    text = render_comparison(res)
    for model in EXPECTED_MODELS:
        assert model in text
    assert "OLS:" in text
    assert "Mixed:" in text


def test_lag_effect_attenuates_in_mixed_model():
    """The whole point of the comparison: OLS lag-stress effect
    should shrink (typically a lot) once we go mixed + within-centred."""
    from ctd.data.generate import generate_synthetic_data
    from ctd.preprocess import (
        add_time_features,
        add_trait_features,
        assign_compulsion_groups,
        clean,
        within_between_decompose,
    )

    raw = generate_synthetic_data(seed=42)
    df = clean(raw)
    df = add_time_features(df)
    df = assign_compulsion_groups(df)
    df = add_trait_features(df)
    df = within_between_decompose(df, ["Stress", "OutsideTime", "Compulsions"])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ols = _legacy_ols(df)["lag_model"]

    ols_lag_beta = ols.loc["PrevStress", "beta"]
    # The legacy OLS lag coefficient should be clearly positive.
    assert ols_lag_beta > 0.2
