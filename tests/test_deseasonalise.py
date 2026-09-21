"""Demand preprocessing: daily aggregation + weekday/half-hour deseasonalising."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.aemo.deseasonalise import (
    aggregate_daily_demand,
    remove_daily_weekly_profile,
    standardise_from_reference,
)


def _half_hourly_series(n_days: int = 5, start: str = "2020-03-02") -> pd.Series:
    # start on a Monday so weekday-indexed tests are easy to reason about
    index = pd.date_range(start, periods=n_days * 48, freq="30min")
    return pd.Series(np.arange(len(index), dtype=float), index=index, name="TOTALDEMAND")


def test_aggregate_collapses_to_one_row_per_complete_day():
    daily = aggregate_daily_demand(_half_hourly_series(n_days=5))
    assert len(daily) == 5
    assert daily.index.is_monotonic_increasing
    assert daily.name == "TOTALDEMAND"


def test_aggregate_mean_matches_manual_daily_mean():
    series = _half_hourly_series(n_days=2)
    daily = aggregate_daily_demand(series)
    expected_day1 = series.iloc[:48].mean()
    assert daily.iloc[0] == pytest.approx(expected_day1)


def test_aggregate_drops_incomplete_days():
    series = _half_hourly_series(n_days=3)
    incomplete = series.drop(series.index[48])  # day 2 now has 47 obs, not 48
    daily = aggregate_daily_demand(incomplete)
    assert len(daily) == 2


def test_aggregate_raises_on_non_series():
    with pytest.raises(TypeError):
        aggregate_daily_demand([1.0, 2.0, 3.0])


def test_aggregate_raises_on_non_datetime_index():
    series = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(TypeError):
        aggregate_daily_demand(series)


def test_aggregate_raises_on_empty_series():
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=float)
    with pytest.raises(ValueError):
        aggregate_daily_demand(empty)


def test_aggregate_raises_on_any_nan():
    series = _half_hourly_series(n_days=2)
    series.iloc[0] = np.nan
    with pytest.raises(ValueError):
        aggregate_daily_demand(series)


def test_aggregate_raises_when_no_complete_days_remain():
    series = _half_hourly_series(n_days=1).iloc[:-1]  # 47 obs, never a full day
    with pytest.raises(ValueError):
        aggregate_daily_demand(series)


def test_remove_profile_zeroes_out_a_series_matching_its_own_reference():
    # A weekly pattern (48 * 7 values) tiled across 2 weeks -> every
    # (weekday, half_hour) combination's mean across the reference equals
    # that single week's own value at that slot, so a target built from one
    # more occurrence of the same pattern should net to exactly zero.
    week_pattern = np.arange(48 * 7, dtype=float)
    reference = pd.Series(
        np.tile(week_pattern, 2),
        index=pd.date_range("2020-03-02", periods=48 * 7 * 2, freq="30min"),
        name="TOTALDEMAND",
    )
    target = pd.Series(
        week_pattern,
        index=pd.date_range("2020-04-06", periods=48 * 7, freq="30min"),  # also a Monday
        name="TOTALDEMAND",
    )

    adjusted = remove_daily_weekly_profile(target, reference=reference)
    np.testing.assert_allclose(adjusted.to_numpy(), 0.0, atol=1e-9)


def test_remove_profile_output_name_and_shape():
    reference = _half_hourly_series(n_days=14)
    target = _half_hourly_series(n_days=2, start="2020-04-06")

    adjusted = remove_daily_weekly_profile(target, reference=reference)
    assert adjusted.name == "TOTALDEMAND_seasonally_adjusted"
    assert len(adjusted) == len(target)


def test_remove_profile_handles_unseen_calendar_combination():
    reference = _half_hourly_series(n_days=1)
    target = _half_hourly_series(n_days=2, start="2020-04-06")

    adjusted = remove_daily_weekly_profile(
        target,
        reference=reference,
    )

    assert len(adjusted) == len(target)
    assert not adjusted.isna().any()


def test_remove_profile_uses_reference_not_target_to_fit_profile():
    reference_index = pd.date_range(
        "2019-01-01",
        periods=48,
        freq="30min",
    )
    reference = pd.Series(
        np.full(48, 100.0),
        index=reference_index,
        name="TOTALDEMAND",
    )

    target_index = pd.date_range(
        "2020-01-01",
        periods=48,
        freq="30min",
    )
    target = pd.Series(
        np.full(48, 150.0),
        index=target_index,
        name="TOTALDEMAND",
    )

    adjusted = remove_daily_weekly_profile(
        target,
        reference=reference,
    )

    np.testing.assert_allclose(
        adjusted.to_numpy(),
        50.0,
    )


def test_standardise_uses_reference_statistics_only():
    reference = pd.Series(
        [0.0, 2.0],
        index=pd.date_range(
            "2019-01-01",
            periods=2,
            freq="30min",
        ),
        name="residual",
    )

    target = pd.Series(
        [1.0, 3.0],
        index=pd.date_range(
            "2020-01-01",
            periods=2,
            freq="30min",
        ),
        name="residual",
    )

    standardised = standardise_from_reference(
        target,
        reference=reference,
    )

    np.testing.assert_allclose(
        standardised.to_numpy(),
        [0.0, 2.0],
    )


def test_standardise_reference_has_zero_mean_unit_variance():
    reference = _half_hourly_series(n_days=2)

    standardised = standardise_from_reference(
        reference,
        reference=reference,
    )

    assert standardised.mean() == pytest.approx(0.0)
    assert standardised.std(ddof=0) == pytest.approx(1.0)


def test_standardise_rejects_zero_variance_reference():
    index = pd.date_range(
        "2019-01-01",
        periods=48,
        freq="30min",
    )
    reference = pd.Series(
        np.ones(48),
        index=index,
        name="TOTALDEMAND",
    )

    with pytest.raises(ValueError):
        standardise_from_reference(
            reference,
            reference=reference,
        )

def test_remove_profile_raises_on_invalid_input():
    reference = _half_hourly_series(n_days=14)
    with pytest.raises(TypeError):
        remove_daily_weekly_profile([1.0, 2.0], reference=reference)
