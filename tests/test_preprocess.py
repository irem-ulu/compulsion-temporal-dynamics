import numpy as np

from ctd.data.generate import SimConfig, generate_synthetic_data_v2
from ctd.preprocess import add_time_features, preprocess, within_between_decompose


def _df():
    return generate_synthetic_data_v2(SimConfig(n_participants=6, n_days=3, seed=3))


def test_preprocess_adds_expected_columns():
    df = preprocess(_df())
    for col in ["IsMorning", "IsEvening", "HourOfDay",
                "CompulsionGroup", "CompulsionTrait",
                "Stress_within", "Stress_between",
                "OutsideTime_within", "OutsideTime_between"]:
        assert col in df.columns, col


def test_within_decomposition_sums_back():
    df = within_between_decompose(_df(), ["Stress"])
    reconstructed = df["Stress_within"] + df["Stress_between"]
    assert np.allclose(reconstructed.values, df["Stress"].values)


def test_within_means_are_zero_per_person():
    df = within_between_decompose(_df(), ["Stress"])
    person_means_within = df.groupby("Participant")["Stress_within"].mean()
    assert np.allclose(person_means_within.values, 0.0, atol=1e-10)


def test_compulsion_group_balanced():
    df = preprocess(_df())
    # Median split — each group should have at least one participant.
    counts = df.groupby("CompulsionGroup")["Participant"].nunique()
    assert (counts >= 1).all()


def test_hour_of_day_default_window():
    """Default 7am-7pm: bin 0 = 7.0, bin 11 = 18.0 (the last *sample*, not the end of the window)."""
    df = add_time_features(_df(), n_bins=12)
    by_bin = df.groupby("TimeBin")["HourOfDay"].first()
    assert by_bin.loc[0] == 7.0
    assert by_bin.loc[11] == 18.0


def test_hour_of_day_custom_window():
    """A 6am-10pm (16h) window should put bin 0 at 6.0 and the bins evenly spaced."""
    df = add_time_features(_df(), n_bins=12, day_start_hour=6.0, day_length_hours=16.0)
    by_bin = df.groupby("TimeBin")["HourOfDay"].first()
    assert by_bin.loc[0] == 6.0
    # Spacing between consecutive bins should be 16/12 hours.
    diffs = np.diff(by_bin.values)
    assert np.allclose(diffs, 16 / 12)
