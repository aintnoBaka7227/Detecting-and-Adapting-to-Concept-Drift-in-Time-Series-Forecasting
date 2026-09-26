"""Tests for SeasonalNaive, mirroring forecasting/."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.aemo import loader
from drift_lab.config import DATA_DIR, RAW_DATA_DIR, REGIONS
from drift_lab.forecasting.seasonal_naive import SeasonalNaive
from drift_lab.forecasting.seasonal_naive_aditya import (
    SeasonalNaive as AdityaSeasonalNaive,
)


def _demand_series(frame):
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def _series(start, periods, values, name="TOTALDEMAND"):
    index = pd.date_range(start, periods=periods, freq="30min")
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SeasonalNaive().predict(pd.DataFrame(index=pd.date_range("2020", periods=3, freq="30min")))


def test_observe_before_fit_raises():
    with pytest.raises(RuntimeError):
        SeasonalNaive().observe(_series("2020-01-01", 3, [1, 2, 3]))


def test_forecast_is_the_value_one_season_ago():
    model = SeasonalNaive(season_length=2)
    train = _series("2020-01-01 00:00", 6, [10, 20, 11, 21, 12, 22])
    model.fit(pd.DataFrame(index=train.index), train)

    test_index = pd.date_range("2020-01-01 03:00", periods=2, freq="30min")
    preds = model.predict(pd.DataFrame(index=test_index))

    # season_length=2 -> each forecast is the observation two steps earlier.
    np.testing.assert_allclose(preds, [12, 22])


def test_aditya_model_blends_day_and_week_lags():
    train = _series("2020-01-01", 14 * 48, np.ones(14 * 48))
    train.loc[pd.Timestamp("2020-01-14 00:00")] = 10.0
    train.loc[pd.Timestamp("2020-01-08 00:00")] = 30.0
    model = AdityaSeasonalNaive(day_weight=0.5)
    model.fit(pd.DataFrame(index=train.index), train)

    test_index = pd.DatetimeIndex(["2020-01-15 00:00"])
    preds = model.predict(pd.DataFrame(index=test_index))

    np.testing.assert_allclose(preds, [20.0])


def test_aditya_model_uses_day_lag_when_seasonal_values_diverge():
    train = _series("2020-01-01", 14 * 48, np.ones(14 * 48))
    train.loc[pd.Timestamp("2020-01-14 00:00")] = 10.0
    train.loc[pd.Timestamp("2020-01-08 00:00")] = 30.0
    model = AdityaSeasonalNaive(day_weight=0.5, max_blend_difference=10.0)
    model.fit(pd.DataFrame(index=train.index), train)

    test_index = pd.DatetimeIndex(["2020-01-15 00:00"])
    preds = model.predict(pd.DataFrame(index=test_index))

    np.testing.assert_allclose(preds, [10.0])


def test_aditya_model_rejects_invalid_day_weight():
    with pytest.raises(ValueError, match="day_weight"):
        AdityaSeasonalNaive(day_weight=1.1)


def test_reproduces_shift_of_concatenated_history():
    rng = np.random.default_rng(0)
    full = _series("2020-01-01", 2000, rng.normal(1000, 50, 2000))
    train, calibration, test = full.iloc[:1400], full.iloc[1400:1600], full.iloc[1600:]

    model = SeasonalNaive(season_length=48)
    model.fit(pd.DataFrame(index=train.index), train)
    model.observe(calibration)
    preds = model.predict(pd.DataFrame({"TOTALDEMAND": test.to_numpy()}, index=test.index))

    expected = full.shift(48).reindex(test.index).to_numpy()
    np.testing.assert_allclose(preds, expected)


@pytest.mark.parametrize("region", REGIONS)
def test_reproduces_committed_baseline_csv(region):
    expected_path = DATA_DIR / "baseline" / "seasonal_naive" / f"{region}_seasonal_naive_test.csv"
    if not (RAW_DATA_DIR / region).exists() or not expected_path.exists():
        pytest.skip("raw AEMO data / baseline CSV not present")

    train, calibration, test = (_demand_series(f) for f in loader.load(region))

    model = SeasonalNaive()
    model.fit(pd.DataFrame(index=train.index), train)
    model.observe(calibration)
    preds = model.predict(pd.DataFrame({"TOTALDEMAND": test.to_numpy()}, index=test.index))

    expected = pd.read_csv(expected_path)["PREDICTION"].to_numpy()
    np.testing.assert_allclose(preds, expected, rtol=0, atol=1e-6)
