"""ScheduledRetrain adapter (Arm B), mirroring adaptation/."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.adaptation.scheduled_retrain import ScheduledRetrain
from drift_lab.forecasting.base import Forecaster


class FakeForecaster(Forecaster):
    """Records every fit() call's (X, y) instead of actually training."""

    name = "fake"

    def __init__(self) -> None:
        self.fit_calls: list[tuple[pd.DataFrame, pd.Series]] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FakeForecaster":
        self.fit_calls.append((X, y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


def _history(n_days: int) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"TOTALDEMAND": np.arange(n_days, dtype=float)}, index=index)


def test_first_call_always_retrains_even_with_no_changepoints():
    model = FakeForecaster()
    adapter = ScheduledRetrain(interval_days=30)

    result = adapter.adapt([], model, _history(10))

    assert result is model
    assert len(model.fit_calls) == 1
    assert adapter.retrain_count == 1


def test_changepoints_are_ignored_entirely():
    model = FakeForecaster()
    adapter = ScheduledRetrain(interval_days=30)
    data = _history(10)

    adapter.adapt([1, 2, 3], model, data)  # would-be drift, irrelevant here

    assert len(model.fit_calls) == 1  # still just the first-call retrain
    assert adapter.retrain_count == 1


def test_second_call_before_interval_elapses_is_a_no_op():
    model = FakeForecaster()
    adapter = ScheduledRetrain(interval_days=30)

    adapter.adapt([], model, _history(10))
    adapter.adapt([], model, _history(20))  # only 10 more days: < 30

    assert len(model.fit_calls) == 1
    assert adapter.retrain_count == 1


def test_retrains_again_once_interval_days_have_elapsed():
    model = FakeForecaster()
    adapter = ScheduledRetrain(interval_days=30)

    adapter.adapt([], model, _history(10))  # retrains at day 10
    adapter.adapt([], model, _history(41))  # 31 days later: retrains again

    assert len(model.fit_calls) == 2
    assert adapter.retrain_count == 2
    _X, y = model.fit_calls[1]
    assert y.index.max() == pd.Timestamp("2020-01-01") + pd.Timedelta(days=40)


def test_retrain_uses_full_history_not_a_window():
    model = FakeForecaster()
    adapter = ScheduledRetrain(interval_days=30)
    data = _history(41)

    adapter.adapt([], model, data)

    _X, y = model.fit_calls[0]
    assert len(y) == len(data)  # full history, not a trailing window


def test_interval_days_is_configurable():
    model = FakeForecaster()
    adapter = ScheduledRetrain(interval_days=7)

    adapter.adapt([], model, _history(1))
    adapter.adapt([], model, _history(9))  # 8 days later: >= 7, retrains

    assert adapter.retrain_count == 2


def test_target_column_selects_from_multi_column_data():
    data = _history(10)
    data["RRP"] = 1.0
    model = FakeForecaster()
    adapter = ScheduledRetrain(target_column="TOTALDEMAND")

    adapter.adapt([], model, data)

    _X, y = model.fit_calls[0]
    assert y.name == "TOTALDEMAND"


def test_multi_column_data_without_target_column_raises():
    data = _history(10)
    data["RRP"] = 1.0
    adapter = ScheduledRetrain()

    with pytest.raises(ValueError):
        adapter.adapt([], FakeForecaster(), data)


def test_single_column_dataframe_infers_target_without_target_column():
    data = _history(10)
    model = FakeForecaster()
    adapter = ScheduledRetrain()

    adapter.adapt([], model, data)

    _X, y = model.fit_calls[0]
    assert y.name == "TOTALDEMAND"
