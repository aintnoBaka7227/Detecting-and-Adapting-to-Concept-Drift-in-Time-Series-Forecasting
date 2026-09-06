"""observe() behaviour + committed-baseline reproduction for XGBoostForecaster.

The recursive full-test forecast is minutes long, so the reproduction test is
opt-in: RUN_BASELINE_REPRO=1 uv run --extra dev pytest tests/test_xgboost_forecaster.py
"""

import os

import numpy as np
import pandas as pd
import pytest

from drift_lab.aemo import loader
from drift_lab.config import DATA_DIR, RAW_DATA_DIR, REGIONS
from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster


def _demand_series(frame):
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def _ar1(n_train: int, n_rest: int, seed: int = 0):
    index = pd.date_range("2018-01-01", periods=n_train + n_rest, freq="30min")
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, n_train + n_rest)
    series = np.zeros(n_train + n_rest)
    series[0] = 1000.0
    for i in range(1, len(series)):
        series[i] = 0.9 * series[i - 1] + 100 + noise[i]
    full = pd.Series(series, index=index, name="TOTALDEMAND")
    return full.iloc[:n_train], full.iloc[n_train:]


def _model() -> XGBoostForecaster:
    return XGBoostForecaster(
        n_estimators=80, max_depth=3, learning_rate=0.1, early_stopping_rounds=None
    )


def test_observe_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().observe(pd.Series([1.0, 2.0], index=pd.date_range("2018", periods=2, freq="30min")))


def test_observe_matches_the_old_history_concat_hack():
    train_y, rest = _ar1(600, 400)
    cal_y, test_y = rest.iloc[:200], rest.iloc[200:]

    refactored = _model().fit(pd.DataFrame(index=train_y.index), train_y)
    refactored.observe(cal_y)

    hacked = _model().fit(pd.DataFrame(index=train_y.index), train_y)
    hacked._history = pd.concat([hacked._history, cal_y])  # pre-refactor notebook line

    pd.testing.assert_series_equal(refactored._history, hacked._history)
    np.testing.assert_allclose(
        refactored.predict(pd.DataFrame(index=test_y.index)),
        hacked.predict(pd.DataFrame(index=test_y.index)),
    )


@pytest.mark.skipif(
    os.environ.get("RUN_BASELINE_REPRO") != "1", reason="set RUN_BASELINE_REPRO=1"
)
@pytest.mark.parametrize("region", REGIONS)
def test_reproduces_committed_baseline_csv(region):
    expected_path = DATA_DIR / "baseline" / "xgboost" / f"{region}_xgboost_test.csv"
    if not (RAW_DATA_DIR / region).exists() or not expected_path.exists():
        pytest.skip("raw AEMO data / baseline CSV not present")

    train, calibration, test = (_demand_series(f) for f in loader.load(region))

    model = XGBoostForecaster(
        interval="30min",
        max_lag=48,
        extra_lags=(96, 144, 336),
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(pd.DataFrame(index=train.index), train)
    model.observe(calibration)
    preds = model.predict(pd.DataFrame(index=test.index))

    # atol covers the committed CSV's ~4-dp rounding of XGBoost's float32
    # output; a real lag-seeding regression would be MW-scale.
    expected = pd.read_csv(expected_path)["PREDICTION"].to_numpy()
    np.testing.assert_allclose(preds, expected, rtol=0, atol=1e-3)
