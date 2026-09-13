"""RetrainUsing3MonthWindows adapter, mirroring adaptation/."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.adaptation.retrain_using_3_month_windows import (
    RetrainUsing3MonthWindows,
)
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


def _history(n_days: int = 400) -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=n_days, freq="D")
    return pd.DataFrame({"TOTALDEMAND": np.arange(n_days, dtype=float)}, index=index)


def test_no_changepoints_returns_model_unchanged_and_does_not_retrain():
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows()

    result = adapter.adapt([], model, _history())

    assert result is model
    assert model.fit_calls == []
    assert adapter.retrain_count == 0


def test_retrains_on_trailing_3_calendar_months_ending_at_changepoint():
    data = _history(n_days=400)
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows()

    changepoint_idx = 300  # data.index[300]
    detection_date = data.index[changepoint_idx]
    adapter.adapt([changepoint_idx], model, data)

    assert len(model.fit_calls) == 1
    X, y = model.fit_calls[0]
    expected_start = detection_date - pd.DateOffset(months=3)
    assert y.index.min() > expected_start
    assert y.index.max() == detection_date
    assert list(X.index) == list(y.index)


def test_window_clips_to_available_history_near_the_start():
    data = _history(n_days=30)  # far less than 3 months
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows()

    adapter.adapt([data.index.get_loc(data.index[-1])], model, data)

    _X, y = model.fit_calls[0]
    assert len(y) == len(data)  # nothing to clip away, entire history used


def test_anchors_on_the_most_recent_changepoint_in_a_batch():
    data = _history(n_days=400)
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows()

    adapter.adapt([50, 100, 300], model, data)

    _X, y = model.fit_calls[0]
    assert y.index.max() == data.index[300]


def test_retrain_count_increments_once_per_adapt_call_with_detections():
    data = _history(n_days=400)
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows()

    adapter.adapt([100], model, data)
    adapter.adapt([], model, data)  # no-op, must not count
    adapter.adapt([200], model, data)

    assert adapter.retrain_count == 2


def test_single_column_dataframe_infers_target_without_target_column():
    data = _history(n_days=200)
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows()

    adapter.adapt([150], model, data)

    _X, y = model.fit_calls[0]
    assert y.name == "TOTALDEMAND"


def test_target_column_selects_from_multi_column_data():
    data = _history(n_days=200)
    data["RRP"] = 1.0
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows(target_column="TOTALDEMAND")

    adapter.adapt([150], model, data)

    _X, y = model.fit_calls[0]
    assert y.name == "TOTALDEMAND"


def test_multi_column_data_without_target_column_raises():
    data = _history(n_days=200)
    data["RRP"] = 1.0
    adapter = RetrainUsing3MonthWindows()

    with pytest.raises(ValueError):
        adapter.adapt([150], FakeForecaster(), data)


def test_window_months_is_configurable():
    data = _history(n_days=400)
    model = FakeForecaster()
    adapter = RetrainUsing3MonthWindows(window_months=1)

    changepoint_idx = 300
    adapter.adapt([changepoint_idx], model, data)

    _X, y = model.fit_calls[0]
    expected_start = data.index[changepoint_idx] - pd.DateOffset(months=1)
    assert y.index.min() > expected_start
    assert len(y) < 95  # roughly one month, not three
