"""Tests for the four-arm adaptation comparison infrastructure."""

import numpy as np
import pandas as pd

from drift_lab.adaptation.never_retrain import NeverRetrain
from drift_lab.adaptation.retrain_full_history import RetrainFullHistory
from drift_lab.adaptation.scheduled_retrain import ScheduledRetrain
from drift_lab.forecasting.base import Forecaster


class FakeForecaster(Forecaster):
    """Record fit calls without training a real forecasting model."""

    name = "fake"

    def __init__(self) -> None:
        self.fit_calls: list[tuple[pd.DataFrame, pd.Series]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FakeForecaster":
        self.fit_calls.append((X, y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def _history(n_days: int = 100) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame(
        {"TOTALDEMAND": np.arange(n_days, dtype=float)},
        index=index,
    )


def test_never_retrain_keeps_model_unchanged():
    data = _history()
    model = FakeForecaster()
    adapter = NeverRetrain()

    result = adapter.adapt([50], model, data)

    assert result is model
    assert model.fit_calls == []
    assert adapter.retrain_count == 0


def test_scheduled_retrain_no_trigger_is_noop():
    data = _history()
    model = FakeForecaster()
    adapter = ScheduledRetrain(target_column="TOTALDEMAND")

    result = adapter.adapt([], model, data)

    assert result is model
    assert model.fit_calls == []
    assert adapter.retrain_count == 0
    assert adapter.train_samples == 0


def test_scheduled_retrain_uses_history_up_to_schedule_point():
    data = _history()
    model = FakeForecaster()
    adapter = ScheduledRetrain(
        interval_days=30,
        target_column="TOTALDEMAND",
    )

    adapter.adapt([59], model, data)

    assert len(model.fit_calls) == 1

    X, y = model.fit_calls[0]

    assert len(y) == 60
    assert y.index[-1] == data.index[59]
    assert list(X.index) == list(y.index)
    assert adapter.retrain_count == 1
    assert adapter.train_samples == 60


def test_scheduled_retrain_accumulates_training_cost():
    data = _history()
    model = FakeForecaster()
    adapter = ScheduledRetrain(target_column="TOTALDEMAND")

    adapter.adapt([29], model, data)
    adapter.adapt([59], model, data)

    assert adapter.retrain_count == 2
    assert adapter.train_samples == 30 + 60


def test_full_history_no_changepoint_is_noop():
    data = _history()
    model = FakeForecaster()
    adapter = RetrainFullHistory(target_column="TOTALDEMAND")

    result = adapter.adapt([], model, data)

    assert result is model
    assert model.fit_calls == []
    assert adapter.retrain_count == 0
    assert adapter.train_samples == 0


def test_full_history_retrain_uses_all_observed_history():
    data = _history()
    model = FakeForecaster()
    adapter = RetrainFullHistory(target_column="TOTALDEMAND")

    adapter.adapt([49], model, data)

    assert len(model.fit_calls) == 1

    X, y = model.fit_calls[0]

    assert len(y) == 50
    assert y.index[0] == data.index[0]
    assert y.index[-1] == data.index[49]
    assert list(X.index) == list(y.index)
    assert adapter.retrain_count == 1
    assert adapter.train_samples == 50


def test_full_history_accumulates_training_cost():
    data = _history()
    model = FakeForecaster()
    adapter = RetrainFullHistory(target_column="TOTALDEMAND")

    adapter.adapt([24], model, data)
    adapter.adapt([74], model, data)

    assert adapter.retrain_count == 2
    assert adapter.train_samples == 25 + 75