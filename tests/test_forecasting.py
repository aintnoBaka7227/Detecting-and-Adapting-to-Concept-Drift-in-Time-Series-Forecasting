"""Tests for forecasting implementations, mirroring forecasting/."""

import numpy as np
import pandas as pd
import pytest

from drift_forecasting.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
)


def _ar1_demand(n_train: int, n_test: int, seed: int = 0):
    train_index = pd.date_range(
        "2018-01-01",
        periods=n_train,
        freq="30min",
    )
    test_index = pd.date_range(
        train_index[-1] + pd.Timedelta("30min"),
        periods=n_test,
        freq="30min",
    )

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 5, n_train + n_test)

    series = np.zeros(n_train + n_test)
    series[0] = 1000.0 + noise[0]

    for i in range(1, len(series)):
        series[i] = 0.9 * series[i - 1] + noise[i]

    train_y = pd.Series(
        series[:n_train],
        index=train_index,
        name="TOTALDEMAND",
    )
    test_y = pd.Series(
        series[n_train:],
        index=test_index,
        name="TOTALDEMAND",
    )

    return train_y, test_y


def _model() -> XGBoostForecaster:
    return XGBoostForecaster(
        n_estimators=120,
        max_depth=3,
        learning_rate=0.1,
        early_stopping_rounds=None,
    )


def test_predict_returns_array_of_length_matching_x():
    train_y, test_y = _ar1_demand(800, 200)

    model = _model().fit(pd.DataFrame(index=train_y.index), train_y)

    predictions = model.predict(pd.DataFrame(index=test_y.index))

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (len(test_y),)
    assert np.isfinite(predictions).all()


def test_fit_rejects_non_datetime_index():
    model = _model()

    with pytest.raises(ValueError):
        model.fit(
            pd.DataFrame(index=range(10)),
            pd.Series(np.arange(10.0)),
        )


def test_walk_forward_actuals_beat_recursive():
    train_y, test_y = _ar1_demand(800, 200, seed=1)

    model = _model().fit(pd.DataFrame(index=train_y.index), train_y)

    recursive = model.predict(pd.DataFrame(index=test_y.index))

    # Supply observed values in the target column; lags reference only
    # timestamps strictly before each forecast point.
    walk_forward = model.predict(
        pd.DataFrame(
            {"TOTALDEMAND": test_y.to_numpy()},
            index=test_y.index,
        )
    )

    recursive_mae = np.mean(np.abs(recursive - test_y.to_numpy()))
    walk_forward_mae = np.mean(np.abs(walk_forward - test_y.to_numpy()))

    assert walk_forward_mae < 0.7 * recursive_mae


def test_predict_is_invariant_to_row_order():
    train_y, test_y = _ar1_demand(800, 200, seed=2)

    model = _model().fit(pd.DataFrame(index=train_y.index), train_y)

    ordered = model.predict(pd.DataFrame(index=test_y.index))

    shuffled_index = test_y.index.to_numpy().copy()
    np.random.default_rng(7).shuffle(shuffled_index)

    shuffled = model.predict(pd.DataFrame(index=pd.DatetimeIndex(shuffled_index)))

    by_timestamp = dict(zip(pd.DatetimeIndex(shuffled_index), shuffled))
    reordered = np.array([by_timestamp[timestamp] for timestamp in test_y.index])

    np.testing.assert_allclose(reordered, ordered)
