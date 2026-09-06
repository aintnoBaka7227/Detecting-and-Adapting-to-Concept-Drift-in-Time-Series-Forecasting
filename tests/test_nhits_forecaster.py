"""NHITSForecaster: interface contract + the roll_forecast chunking adapter.

Trains real (tiny) NHITS models — max_steps is kept low so this stays fast
enough to run unconditionally, unlike the opt-in xgboost/dhr_arima baseline
repro tests (there's no committed NHITS baseline to reproduce yet anyway).
"""

import numpy as np
import pandas as pd
import pytest

from drift_lab.forecasting.nhits_forecaster import NHITSForecaster


def _series(n: int, seed: int = 0) -> pd.Series:
    index = pd.date_range("2018-01-01", periods=n, freq="30min")
    rng = np.random.default_rng(seed)
    values = 1000 + 200 * np.sin(2 * np.pi * np.arange(n) / 48) + rng.normal(0, 10, n)
    return pd.Series(values, index=index, name="TOTALDEMAND")


def _model(**overrides) -> NHITSForecaster:
    params = {"horizon": 24, "input_size": 96, "max_steps": 20}
    params.update(overrides)
    return NHITSForecaster(**params)


def test_observe_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().observe(_series(10))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        _model().predict(pd.DataFrame(index=_series(10).index))


def test_fit_predict_covers_multiple_horizon_chunks_without_nans():
    train_y = _series(500)
    test_y = _series(120, seed=1)
    test_y.index = (
        train_y.index[-1]
        + pd.Timedelta("30min")
        + pd.timedelta_range("0min", periods=120, freq="30min")
    )

    model = _model().fit(pd.DataFrame(index=train_y.index), train_y)
    # 120 points at horizon=24 forces roll_forecast through 5 chunks.
    preds = model.predict(pd.DataFrame(index=test_y.index))

    assert preds.shape == (120,)
    assert np.isfinite(preds).all()


def test_observe_extends_history_and_predict_uses_walk_forward_column():
    train_y = _series(500)
    cal_y = _series(48, seed=2)
    cal_y.index = train_y.index[-1] + pd.timedelta_range(
        "30min", periods=48, freq="30min"
    )
    test_y = _series(48, seed=3)
    test_y.index = cal_y.index[-1] + pd.timedelta_range(
        "30min", periods=48, freq="30min"
    )

    model = _model().fit(pd.DataFrame(index=train_y.index), train_y)
    model.observe(cal_y)
    assert len(model._history) == len(train_y) + len(cal_y)

    preds = model.predict(
        pd.DataFrame({"TOTALDEMAND": test_y.to_numpy()}, index=test_y.index)
    )
    assert preds.shape == (48,)
    assert np.isfinite(preds).all()


def test_predict_row_order_matches_input_regardless_of_x_order():
    train_y = _series(500)
    test_y = _series(48, seed=4)
    test_y.index = train_y.index[-1] + pd.timedelta_range(
        "30min", periods=48, freq="30min"
    )

    model = _model().fit(pd.DataFrame(index=train_y.index), train_y)
    in_order = model.predict(pd.DataFrame(index=test_y.index))

    shuffled_index = test_y.index[::-1]
    reversed_preds = model.predict(pd.DataFrame(index=shuffled_index))

    np.testing.assert_allclose(in_order, reversed_preds[::-1])


def test_config_of_only_exposes_public_hyperparameters():
    from experiments.run_harness import config_of

    model = _model(random_seed=7)
    config = config_of(model)

    assert config == {
        "horizon": 24,
        "input_size": 96,
        "freq": "30min",
        "max_steps": 20,
        "learning_rate": 0.001,
        "random_seed": 7,
    }
