"""Reusable AEMO demand preprocessing for drift-detection experiments.

The transformations in this module operate on already-cleaned demand
series. Seasonal profiles and standardisation statistics are learned from
a historical reference series so future observations are not used when
estimating the preprocessing transformation.
"""

from __future__ import annotations

import numpy as np
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


def aggregate_daily_demand(series: pd.Series) -> pd.Series:
    """Aggregate half-hourly demand into complete daily means."""

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


def _seasonal_frame(series: pd.Series) -> pd.DataFrame:
    """Build calendar features used by the seasonal demand profile."""

    frame = pd.DataFrame({"demand": series.astype(float)})

    frame["day_of_year"] = frame.index.dayofyear
    frame["weekday"] = frame.index.dayofweek
    frame["half_hour"] = (
        frame.index.hour * 2
        + frame.index.minute // 30
    )

    # Treat 29 February like 28 February so a leap-day observation does
    # not require a seasonal cell that is absent from non-leap years.
    leap_day = (
        (frame.index.month == 2)
        & (frame.index.day == 29)
    )
    frame.loc[leap_day, "day_of_year"] = 59

    return frame


def remove_daily_weekly_profile(
    series: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    """Remove annual, weekly and intraday seasonality.

    The seasonal profile is fitted exclusively from ``reference``.
    It combines day-of-year, day-of-week and half-hour effects using
    an additive decomposition around the reference mean.

    The target ``series`` is never used to fit the seasonal profile,
    preventing information from the target/test period leaking into
    preprocessing.
    """

    _validate_datetime_series(series)
    _validate_datetime_series(reference)

    reference_frame = _seasonal_frame(reference)
    target = _seasonal_frame(series)

    overall_mean = float(reference_frame["demand"].mean())

    annual_effect = (
        reference_frame.groupby("day_of_year")["demand"].mean()
        - overall_mean
    )

    weekly_effect = (
        reference_frame.groupby("weekday")["demand"].mean()
        - overall_mean
    )

    intraday_effect = (
        reference_frame.groupby("half_hour")["demand"].mean()
        - overall_mean
    )

    expected = pd.Series(
        overall_mean,
        index=target.index,
        dtype=float,
    )

    expected += (
        target["day_of_year"]
        .map(annual_effect)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    expected += (
        target["weekday"]
        .map(weekly_effect)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    expected += (
        target["half_hour"]
        .map(intraday_effect)
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    adjusted = pd.Series(
        target["demand"].to_numpy(dtype=float)
        - expected.to_numpy(dtype=float),
        index=series.index,
        name=f"{series.name or 'demand'}_seasonally_adjusted",
    )

    return adjusted


def standardise_from_reference(
    series: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    """Z-score a series using mean and standard deviation from reference.

    Statistics are fitted exclusively on ``reference`` and then applied
    unchanged to ``series``:

        z = (x - reference_mean) / reference_std

    This keeps detector thresholds scale-free and prevents target/test
    observations from influencing the transformation.
    """

    _validate_datetime_series(series)
    _validate_datetime_series(reference)

    reference_values = reference.astype(float)

    mean = float(reference_values.mean())
    std = float(reference_values.std(ddof=0))

    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError(
            "Reference mean and standard deviation must be finite."
        )

    if std <= 0.0:
        raise ValueError(
            "Reference standard deviation must be greater than zero."
        )

    standardised = (
        series.astype(float) - mean
    ) / std

    standardised.name = (
        f"{series.name or 'demand'}_standardised"
    )

    return standardised
