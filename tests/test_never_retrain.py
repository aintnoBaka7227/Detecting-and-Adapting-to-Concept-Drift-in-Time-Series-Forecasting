"""NeverRetrain adapter (Arm A), mirroring adaptation/."""

import numpy as np
import pandas as pd

from drift_lab.adaptation.never_retrain import NeverRetrain
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
    adapter = NeverRetrain()

    result = adapter.adapt([], model, _history())

    assert result is model
    assert model.fit_calls == []
    assert adapter.retrain_count == 0


def test_changepoints_present_still_never_retrains():
    model = FakeForecaster()
    adapter = NeverRetrain()
    data = _history()

    result = adapter.adapt([50, 100, 300], model, data)

    assert result is model
    assert model.fit_calls == []
    assert adapter.retrain_count == 0


def test_repeated_calls_never_retrain():
    model = FakeForecaster()
    adapter = NeverRetrain()
    data = _history()

    adapter.adapt([10], model, data)
    adapter.adapt([], model, data)
    adapter.adapt([20, 30], model, data)

    assert model.fit_calls == []
    assert adapter.retrain_count == 0
