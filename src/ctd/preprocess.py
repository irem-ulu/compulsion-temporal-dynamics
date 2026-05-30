"""Cleaning, time features, grouping, person-mean centering.

Most of this is a refactor of the prototype's `preprocessing`, `grouping`
and `feature_engineering` modules into one cohesive step.

The interesting addition is `within_between_decompose` — splits a
time-varying predictor into a person-mean (between) and within-person
deviation. This is the standard EMA move; the prototype didn't do it,
which is why its slopes mixed two very different effects.
"""

from __future__ import annotations

import pandas as pd


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Participant"] = df["Participant"].astype(str)
    df["Day"] = df["Day"].astype(int)
    df["TimeBin"] = df["TimeBin"].astype(int)
    return df


def add_time_features(
    df: pd.DataFrame,
    n_bins: int = 12,
    day_start_hour: float = 7.0,
    day_length_hours: float = 12.0,
) -> pd.DataFrame:
    df = df.copy()
    # Approximate morning/evening as the first and last quarters of the day.
    quarter = n_bins // 4
    df["IsMorning"] = df["TimeBin"].between(0, quarter - 1).astype(int)
    df["IsEvening"] = df["TimeBin"].between(n_bins - quarter, n_bins - 1).astype(int)
    # Continuous time in hours. Defaults to a 7am–7pm sampling window which
    # matches the simulator; override to match your real EMA schedule.
    df["HourOfDay"] = day_start_hour + day_length_hours * df["TimeBin"] / n_bins
    return df


def assign_compulsion_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comp_mean = df.groupby("Participant")["Compulsions"].mean()
    threshold = comp_mean.median()
    df["CompulsionGroup"] = df["Participant"].map(
        lambda p: "High" if comp_mean[p] >= threshold else "Low"
    )
    return df


def add_trait_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["CompulsionTrait"] = df["Participant"].map(
        df.groupby("Participant")["Compulsions"].mean()
    )
    df["StressTrait"] = df["Participant"].map(
        df.groupby("Participant")["Stress"].mean()
    )
    return df


def within_between_decompose(
    df: pd.DataFrame, columns: list[str], group_col: str = "Participant"
) -> pd.DataFrame:
    """Split each column into a person-mean and within-person deviation.

    Hamaker-style centering: ``X = X_between + X_within`` where
    ``X_between`` is the participant's grand mean and ``X_within`` is
    the deviation from it. Putting both into a regression separates the
    between-person effect ("higher-stress people compulse more") from the
    within-person effect ("when stress goes up for *you*, do *your*
    compulsions go up?"), which are conceptually different things that
    the prototype's pooled OLS conflated.
    """
    df = df.copy()
    for col in columns:
        between = df.groupby(group_col)[col].transform("mean")
        df[f"{col}_between"] = between
        df[f"{col}_within"] = df[col] - between
    return df


def preprocess(
    df: pd.DataFrame,
    n_bins: int = 12,
    day_start_hour: float = 7.0,
    day_length_hours: float = 12.0,
) -> pd.DataFrame:
    """One-shot preprocess used by the pipeline."""
    df = clean(df)
    df = add_time_features(
        df,
        n_bins=n_bins,
        day_start_hour=day_start_hour,
        day_length_hours=day_length_hours,
    )
    df = assign_compulsion_groups(df)
    df = add_trait_features(df)
    df = within_between_decompose(df, ["Stress", "OutsideTime", "Compulsions"])
    return df
