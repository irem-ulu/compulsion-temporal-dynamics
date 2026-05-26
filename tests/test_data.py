import pytest

from ctd.data.generate import (
    COLUMNS,
    SimConfig,
    generate_synthetic_data,
    generate_synthetic_data_v2,
)


def test_legacy_generator_shape():
    df = generate_synthetic_data(n_participants=5, n_days=2, n_bins=12, seed=1)
    assert df.shape == (5 * 2 * 12, len(COLUMNS))
    assert list(df.columns) == COLUMNS
    assert df["Participant"].nunique() == 5


def test_v2_generator_shape():
    cfg = SimConfig(n_participants=4, n_days=3, n_bins=12, seed=7)
    df = generate_synthetic_data_v2(cfg)
    assert df.shape == (4 * 3 * 12, len(COLUMNS))
    assert df["Compulsions"].min() >= 0
    assert df["OutsideTime"].min() >= 0


def test_v2_deterministic_for_same_seed():
    cfg = SimConfig(seed=11)
    a = generate_synthetic_data_v2(cfg)
    b = generate_synthetic_data_v2(cfg)
    assert (a == b).all().all()


@pytest.mark.parametrize("seed", [0, 1, 42])
def test_v2_has_between_person_variance(seed):
    cfg = SimConfig(n_participants=10, n_days=3, seed=seed)
    df = generate_synthetic_data_v2(cfg)
    person_means = df.groupby("Participant")["Stress"].mean()
    # Person-level intercept SD is 0.8 — should produce noticeable spread.
    assert person_means.std() > 0.2
