"""Step 5 adaptation arms A-D: training windows, retrain bookkeeping and
the arm B schedule."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.adaptation.never_retrain import NeverRetrain
from drift_lab.adaptation.retrain_every_30_days import RetrainEvery30Days
from drift_lab.adaptation.retrain_using_full_history import RetrainUsingFullHistory
from drift_lab.adaptation.retrain_using_recent_window import RetrainUsingRecentWindow
from drift_lab.forecasting.base import Forecaster


class RecordingForecaster(Forecaster):
    """Forecasts a constant equal to how many times it has been fitted, and
    records every fit window and predict index."""

    name = "recording"

    def __init__(self) -> None:
        self.fit_calls: list[pd.Series] = []
        self.predict_calls: list[pd.DatetimeIndex] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecordingForecaster":
        self.fit_calls.append(y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self.predict_calls.append(pd.DatetimeIndex(X.index))
        return np.full(len(X), float(len(self.fit_calls)))


def half_hourly(start: str, days: int) -> pd.Series:
    index = pd.date_range(start, periods=days * 48, freq="30min")
    return pd.Series(np.arange(len(index), dtype=float), index=index, name="TOTALDEMAND")


def frame(y: pd.Series) -> pd.DataFrame:
    return y.to_frame("TOTALDEMAND")


# --- adapters --------------------------------------------------------------


def test_never_retrain_returns_model_unchanged():
    model = RecordingForecaster()
    adapter = NeverRetrain()
    data = frame(half_hourly("2020-01-01", 10))

    assert adapter.adapt([len(data) - 1], model, data) is model
    assert model.fit_calls == []
    assert (adapter.retrain_count, adapter.train_samples) == (0, 0)


def test_full_history_arm_trains_from_first_day_to_boundary():
    data = frame(half_hourly("2018-01-01", 60))
    boundary = 30 * 48 - 1  # last half-hour of day 30
    model = RecordingForecaster()
    adapter = RetrainUsingFullHistory(target_column="TOTALDEMAND")

    adapter.adapt([boundary], model, data)

    (y,) = model.fit_calls
    assert y.index.min() == data.index[0]
    assert y.index.max() == data.index[boundary]
    assert adapter.retrain_count == 1
    assert adapter.train_samples == boundary + 1
    assert adapter.retrain_timestamps == [data.index[boundary]]


def test_every_30_days_trains_on_last_30_days_only():
    data = frame(half_hourly("2018-01-01", 400))
    boundary = 300 * 48 - 1  # 23:30 on day 300
    model = RecordingForecaster()
    adapter = RetrainEvery30Days(target_column="TOTALDEMAND")

    adapter.adapt([boundary], model, data)

    (y,) = model.fit_calls
    assert y.index.max() == data.index[boundary]
    assert len(y) == 30 * 48
    assert y.index.min() == data.index[boundary].normalize() - pd.Timedelta(days=29)
    assert adapter.train_samples == 30 * 48


@pytest.mark.parametrize("window_days", [30, 60, 120, 180])
def test_recent_window_is_exactly_window_days_ending_at_boundary(window_days):
    data = frame(half_hourly("2019-01-01", 400))
    boundary = 300 * 48 - 1  # 23:30 on day 300
    model = RecordingForecaster()
    adapter = RetrainUsingRecentWindow(window_days, target_column="TOTALDEMAND")

    adapter.adapt([boundary], model, data)

    (y,) = model.fit_calls
    assert y.index.max() == data.index[boundary]
    assert len(y) == window_days * 48
    assert y.index.min() == data.index[boundary].normalize() - pd.Timedelta(days=window_days - 1)


def test_adapters_never_train_past_the_boundary():
    data = frame(half_hourly("2019-01-01", 200))
    boundary = 100 * 48 - 1
    for adapter in (
        RetrainUsingFullHistory(),
        RetrainEvery30Days(),
        RetrainUsingRecentWindow(30),
    ):
        model = RecordingForecaster()
        adapter.adapt([boundary], model, data)
        assert model.fit_calls[0].index.max() <= data.index[boundary]


def test_empty_changepoints_do_not_retrain():
    data = frame(half_hourly("2019-01-01", 5))
    for adapter in (RetrainUsingFullHistory(), RetrainEvery30Days(), RetrainUsingRecentWindow(30)):
        model = RecordingForecaster()
        assert adapter.adapt([], model, data) is model
        assert adapter.retrain_count == 0


def test_every_30_days_schedule_is_calendar_only():
    days = RetrainEvery30Days().schedule("2020-03-01 00:30", "2020-06-15 23:30")
    assert days == [
        pd.Timestamp("2020-03-30"),
        pd.Timestamp("2020-04-29"),
        pd.Timestamp("2020-05-29"),
    ]


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        RetrainUsingRecentWindow(0)
    with pytest.raises(ValueError):
        RetrainEvery30Days(interval_days=0)
    with pytest.raises(ValueError):
        RetrainEvery30Days(window_days=0)


# --- retraining boundary in the arm runner -----------------------------------

from experiments.run.adaptation.run_aemo_adaptation_arms import run_arm


def test_run_arm_old_model_through_day_d_new_model_from_d_plus_1():
    history = half_hourly("2020-01-01", 60)
    test = pd.Series(1.0, index=pd.date_range("2020-03-01", periods=48 * 10, freq="30min"), name="TOTALDEMAND")

    model = RecordingForecaster()
    model.fit(pd.DataFrame(index=history.index), history)  # initial TRAIN fit
    adapter = RetrainUsingFullHistory(target_column="TOTALDEMAND")
    trigger_day = pd.Timestamp("2020-03-04")

    preds = run_arm(model, adapter, [trigger_day], history, test, lambda s: pd.DataFrame(index=s.index))

    next_day = trigger_day + pd.Timedelta(days=1)
    assert (preds[preds.index < next_day] == 1.0).all()  # old model through d 23:30
    assert (preds[preds.index >= next_day] == 2.0).all()  # retrained model from d+1
    retrain_y = model.fit_calls[1]
    assert retrain_y.index.max() == trigger_day + pd.Timedelta(hours=23, minutes=30)
    assert retrain_y.index.min() == history.index[0]
    assert adapter.retrain_count == 1
    assert preds.index.equals(test.index)


def test_run_arm_ignores_trigger_on_last_day_and_arm_a_never_refits():
    history = half_hourly("2020-01-01", 30)
    test = pd.Series(1.0, index=pd.date_range("2020-03-01", periods=48 * 3, freq="30min"), name="TOTALDEMAND")

    model = RecordingForecaster()
    model.fit(pd.DataFrame(index=history.index), history)
    adapter = RetrainUsingFullHistory()
    run_arm(model, adapter, [pd.Timestamp("2020-03-03")], history, test, lambda s: pd.DataFrame(index=s.index))
    assert adapter.retrain_count == 0

    model_a = RecordingForecaster()
    model_a.fit(pd.DataFrame(index=history.index), history)
    run_arm(model_a, NeverRetrain(), [pd.Timestamp("2020-03-01")], history, test, lambda s: pd.DataFrame(index=s.index))
    assert len(model_a.fit_calls) == 1
