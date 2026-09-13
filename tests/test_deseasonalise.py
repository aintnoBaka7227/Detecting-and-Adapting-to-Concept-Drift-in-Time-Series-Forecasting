"""Daily-aggregation helper: reduces a half-hourly split to one point/day."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.aemo.deseasonalise import daily_aggregate


def _half_hourly_split(n_days: int = 5) -> pd.DataFrame:
    index = pd.date_range("2020-03-01", periods=n_days * 48, freq="30min")
    return pd.DataFrame(
        {
            "REGION": "NSW1",
            "SETTLEMENTDATE": index,
            "TOTALDEMAND": np.arange(len(index), dtype=float),
            "RRP": np.arange(len(index), dtype=float) / 10,
        }
    )


def test_collapses_to_one_row_per_day():
    daily = daily_aggregate(_half_hourly_split(n_days=5))
    assert len(daily) == 5
    assert daily.index.is_monotonic_increasing


def test_mean_matches_manual_daily_mean():
    split = _half_hourly_split(n_days=2)
    daily = daily_aggregate(split)
    expected_day1 = split["TOTALDEMAND"].iloc[:48].mean()
    assert daily.iloc[0] == pytest.approx(expected_day1)


def test_reusable_on_any_column():
    daily_price = daily_aggregate(_half_hourly_split(), column="RRP")
    assert daily_price.name == "RRP"


def test_drops_all_nan_days():
    split = _half_hourly_split(n_days=3)
    split.loc[split["SETTLEMENTDATE"].dt.day == 2, "TOTALDEMAND"] = np.nan
    daily = daily_aggregate(split)
    assert len(daily) == 2


def test_missing_column_raises():
    split = _half_hourly_split()
    with pytest.raises(ValueError):
        daily_aggregate(split, column="NOT_A_COLUMN")


def test_missing_settlementdate_raises():
    split = _half_hourly_split().drop(columns=["SETTLEMENTDATE"])
    with pytest.raises(ValueError):
        daily_aggregate(split)
