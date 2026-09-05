"""observe() no-op + committed-baseline reproduction for DHRArima.

The SARIMAX fit + 67k-step forecast is minutes long, so the reproduction test
is opt-in: RUN_BASELINE_REPRO=1 uv run --extra dev pytest tests/test_dhr_arima.py
"""

import os

import numpy as np
import pandas as pd
import pytest

from drift_lab.aemo import loader
from drift_lab.config import DATA_DIR, RAW_DATA_DIR, REGIONS
from drift_lab.evaluation import calculate_rolling_mae
from drift_lab.forecasting.dhr_arima import DHRArima, _fourier_terms


def _demand_series(frame):
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def test_fourier_terms_shape_and_phase():
    terms = _fourier_terms([0, 12, 24], periods=[48], n_harmonics=[2])
    assert list(terms.columns) == ["sin_48_1", "cos_48_1", "sin_48_2", "cos_48_2"]
    assert terms["sin_48_1"].iloc[0] == pytest.approx(0.0)
    assert terms["cos_48_1"].iloc[0] == pytest.approx(1.0)


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        DHRArima().predict(pd.DataFrame(index=pd.date_range("2020", periods=3, freq="30min")))


def test_observe_is_a_noop():
    index = pd.date_range("2018-01-01", periods=500, freq="30min")
    y = pd.Series(1000 + 50 * np.sin(np.arange(500) * 2 * np.pi / 48), index=index)
    model = DHRArima(periods=(48,), n_harmonics=(2,)).fit(pd.DataFrame(index=index), y)
    before = model._fitted
    model.observe(y)
    assert model._fitted is before


@pytest.mark.skipif(
    os.environ.get("RUN_BASELINE_REPRO") != "1", reason="set RUN_BASELINE_REPRO=1"
)
@pytest.mark.parametrize("region", REGIONS)
def test_reproduces_committed_rolling_mae_csv(region):
    expected_path = DATA_DIR / "baseline" / "dhr_arima" / f"{region}_rolling_mae.csv"
    if not (RAW_DATA_DIR / region).exists() or not expected_path.exists():
        pytest.skip("raw AEMO data / reference CSV not present")

    train, _calibration, test = loader.load(region)
    train, test = _demand_series(train), _demand_series(test)

    model = DHRArima()  # defaults: periods (48, 336), harmonics (4, 4), order (2, 0, 2)
    model.fit(pd.DataFrame(index=train.index), train)
    preds = model.predict(pd.DataFrame(index=test.index))
    curve = calculate_rolling_mae(test.to_numpy(), preds)

    expected = pd.read_csv(expected_path)["mae"].to_numpy()
    np.testing.assert_allclose(curve.to_numpy(), expected, rtol=1e-6, atol=1e-4, equal_nan=True)
