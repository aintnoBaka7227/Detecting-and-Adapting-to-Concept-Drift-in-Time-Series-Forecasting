"""Reusable demand preprocessing for drift-detection experiments.

The functions in this module transform an already-cleaned demand series.
They do not perform raw AEMO data cleaning.

Two distinct preprocessing techniques are provided:

1. aggregate_daily_demand()
   Converts half-hourly demand into one mean value per complete day.

2. remove_daily_weekly_profile()
   Removes the expected weekday/half-hour demand profile from a
   half-hourly demand series.
"""

from __future__ import annotations

import pandas as pd

HALF_HOURS_PER_DAY = 48


def _validate_datetime_series(series: pd.Series) -> None:
    """Validate a time-indexed numerical demand series."""

    if not isinstance(series, pd.Series):
        raise TypeError("Demand input must be a pandas Series.")

    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError(
            "Demand series must use a pandas DatetimeIndex."
        )

    if series.empty:
        raise ValueError("Demand series must not be empty.")

    if series.isna().any():
        raise ValueError(
            "Demand series contains missing values."
        )


def aggregate_daily_demand(
    series: pd.Series,
) -> pd.Series:
    """Aggregate half-hourly demand into complete daily means.

    A valid day must contain exactly 48 half-hourly observations.
    Incomplete days are excluded instead of being imputed.

    Parameters
    ----------
    series:
        Cleaned half-hourly demand indexed by timestamp.

    Returns
    -------
    pandas.Series
        One mean demand value per complete day.
    """

    _validate_datetime_series(series)

    grouped = series.resample("1D")

    counts = grouped.count()
    daily = grouped.mean()

    daily = daily[counts == HALF_HOURS_PER_DAY]

    if daily.empty:
        raise ValueError(
            "Daily aggregation produced no complete days."
        )

    if daily.isna().any():
        raise ValueError(
            "Daily aggregation produced missing values."
        )

    daily.name = series.name

    return daily


def remove_daily_weekly_profile(
    series: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    """Remove the normal weekday/half-hour demand profile.

    The expected demand is calculated separately for each combination of:

    - day of week: Monday to Sunday
    - half-hour slot: 0 to 47

    The expected profile is learned from ``reference`` and then subtracted
    from ``series``.

    Using a separate reference period allows experiments to learn normal
    seasonality from historical data without using the future test stream
    to estimate the seasonal profile.

    Parameters
    ----------
    series:
        Half-hourly demand series to adjust.

    reference:
        Historical half-hourly demand used to estimate the normal profile.

    Returns
    -------
    pandas.Series
        Seasonally adjusted demand residuals.
    """

    _validate_datetime_series(series)
    _validate_datetime_series(reference)

    reference_frame = pd.DataFrame(
        {
            "demand": reference.astype(float),
        }
    )

    reference_frame["weekday"] = (
        reference_frame.index.dayofweek
    )

    reference_frame["half_hour"] = (
        reference_frame.index.hour * 2
        + reference_frame.index.minute // 30
    )

    profile = (
        reference_frame
        .groupby(
            ["weekday", "half_hour"]
        )["demand"]
        .mean()
    )

    target = pd.DataFrame(
        {
            "demand": series.astype(float),
        }
    )

    target["weekday"] = target.index.dayofweek

    target["half_hour"] = (
        target.index.hour * 2
        + target.index.minute // 30
    )

    expected = pd.MultiIndex.from_arrays(
        [
            target["weekday"],
            target["half_hour"],
        ],
        names=[
            "weekday",
            "half_hour",
        ],
    ).map(profile)

    if pd.isna(expected).any():
        raise ValueError(
            "Seasonal reference does not contain every "
            "weekday/half-hour combination required by the target series."
        )

    adjusted = pd.Series(
        target["demand"].to_numpy()
        - expected.to_numpy(),
        index=series.index,
        name=f"{series.name or 'demand'}_seasonally_adjusted",
    )

    return adjusted