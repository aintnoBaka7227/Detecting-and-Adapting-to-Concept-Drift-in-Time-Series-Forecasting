"""Seasonal preprocessing for already-cleaned AEMO demand series.

Profiles are fitted on training data only and then applied unchanged to later
splits. No raw-data quality validation is performed in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


Frequency = Literal["daily", "30min"]


@dataclass(frozen=True)
class SeasonalProfile:
    """A fitted multiplicative seasonal profile."""

    frequency: Frequency
    base_demand: float
    annual_factor: pd.Series
    weekly_factor: pd.Series


def aggregate_daily_demand(series: pd.Series) -> pd.Series:
    """Aggregate half-hourly demand into daily means."""

    daily = series.resample("1D").mean()
    daily.name = series.name
    return daily


def _seasonal_day(index: pd.DatetimeIndex) -> np.ndarray:
    """Map dates onto a common 365-day cycle, with leap day at 59.5."""

    day = index.dayofyear.to_numpy(dtype=float)
    day[index.is_leap_year & (index.month > 2)] -= 1.0
    day[(index.month == 2) & (index.day == 29)] = 59.5
    return day


def _half_hour_slot(index: pd.DatetimeIndex) -> np.ndarray:
    """Return half-hour slots numbered from 0 to 47."""

    return (index.hour * 2 + index.minute // 30).to_numpy()


def _cyclic_rolling_mean(values: pd.Series, window: int) -> pd.Series:
    """Smooth a profile while joining the end of December to January."""

    padding = window // 2
    extended = pd.concat(
        [values.iloc[-padding:], values, values.iloc[:padding]],
        ignore_index=True,
    )
    smoothed = extended.rolling(window=window, center=True).mean()
    result = smoothed.iloc[padding:padding + len(values)].copy()
    result.index = values.index
    return result


def _annual_values(
    index: pd.DatetimeIndex,
    annual_factor: pd.Series,
) -> np.ndarray:
    """Map timestamps to annual factors and interpolate 29 February."""

    return np.interp(
        _seasonal_day(index),
        annual_factor.index.to_numpy(dtype=float),
        annual_factor.to_numpy(dtype=float),
    )


def fit_seasonal_profile(
    reference: pd.Series,
    frequency: Frequency,
    annual_smoothing_days: int = 31,
) -> SeasonalProfile:
    """Fit a multiplicative seasonal profile on training data only.

    Use a daily training series with ``frequency="daily"``. This fits
    day-of-year x day-of-week factors.

    Use a half-hourly training series with ``frequency="30min"``. This fits
    day-of-year x combined weekday/half-hour factors.
    """

    reference = reference.astype(float)
    base_demand = float(reference.mean())

    if frequency == "daily":
        daily_reference = reference
    elif frequency == "30min":
        daily_reference = reference.resample("1D").mean()
    else:
        raise ValueError("frequency must be either 'daily' or '30min'.")

    daily_frame = pd.DataFrame(
        {
            "demand": daily_reference,
            "seasonal_day": _seasonal_day(daily_reference.index),
        }
    )
    annual_level = (
        daily_frame.groupby("seasonal_day")["demand"]
        .mean()
        .reindex(np.arange(1.0, 366.0))
        .interpolate(limit_direction="both")
    )
    annual_level = _cyclic_rolling_mean(
        annual_level,
        window=annual_smoothing_days,
    )
    annual_factor = annual_level / annual_level.mean()
    annual_factor.name = "annual_factor"

    reference_frame = pd.DataFrame(
        {
            "annual_adjusted": (
                reference.to_numpy()
                / _annual_values(reference.index, annual_factor)
            ),
            "weekday": reference.index.dayofweek,
        },
        index=reference.index,
    )

    if frequency == "daily":
        weekly_level = (
            reference_frame.groupby("weekday")["annual_adjusted"].mean()
        )
    else:
        reference_frame["half_hour"] = _half_hour_slot(reference.index)
        weekly_level = (
            reference_frame
            .groupby(["weekday", "half_hour"])["annual_adjusted"]
            .mean()
        )

    weekly_factor = weekly_level / weekly_level.mean()
    weekly_factor.name = "weekly_factor"

    return SeasonalProfile(
        frequency=frequency,
        base_demand=base_demand,
        annual_factor=annual_factor,
        weekly_factor=weekly_factor,
    )


def apply_seasonal_profile(
    series: pd.Series,
    profile: SeasonalProfile,
) -> pd.Series:
    """Subtract a frozen multiplicative profile from a demand series."""

    annual = _annual_values(series.index, profile.annual_factor)

    if profile.frequency == "daily":
        weekly = profile.weekly_factor.reindex(
            series.index.dayofweek
        ).to_numpy()
    else:
        keys = pd.MultiIndex.from_arrays(
            [series.index.dayofweek, _half_hour_slot(series.index)],
            names=["weekday", "half_hour"],
        )
        weekly = profile.weekly_factor.reindex(keys).to_numpy()

    expected = profile.base_demand * annual * weekly

    return pd.Series(
        series.astype(float).to_numpy() - expected,
        index=series.index,
        name=f"{series.name or 'demand'}_seasonally_adjusted",
    )


def remove_daily_seasonal_profile(
    series: pd.Series,
    reference: pd.Series,
    annual_smoothing_days: int = 31,
) -> pd.Series:
    """Remove day-of-year x day-of-week effects from daily demand."""

    profile = fit_seasonal_profile(
        reference=reference,
        frequency="daily",
        annual_smoothing_days=annual_smoothing_days,
    )
    return apply_seasonal_profile(series, profile)


def remove_daily_weekly_profile(
    series: pd.Series,
    reference: pd.Series,
    annual_smoothing_days: int = 31,
) -> pd.Series:
    """Remove annual and weekly/intraday effects from half-hourly demand.

    This preserves the existing function name while changing its model to:

        base demand x day-of-year factor x weekday/half-hour factor
    """

    profile = fit_seasonal_profile(
        reference=reference,
        frequency="30min",
        annual_smoothing_days=annual_smoothing_days,
    )
    return apply_seasonal_profile(series, profile)