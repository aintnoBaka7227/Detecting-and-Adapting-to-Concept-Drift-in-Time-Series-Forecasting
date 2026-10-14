"""Independent DHR-ARIMA implementation for baseline cross-checking.

Uses harmonic regression for daily/weekly seasonality with ARIMA errors.
This implementation is intentionally separate from the existing project
baseline so its forecasts can be used as an independent validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from drift_lab.forecasting.base import Forecaster


class DHRArimaCrosscheck(Forecaster):
    name = "dhr_arima_crosscheck"

    def __init__(
        self,
        daily_harmonics: int = 4,
        weekly_harmonics: int = 4,
        arima_order: tuple[int, int, int] = (2, 0, 2),
        interval: str = "30min",
    ) -> None:
        self.daily_harmonics = daily_harmonics
        self.weekly_harmonics = weekly_harmonics
        self.arima_order = arima_order
        self.interval = interval

        self._interval = pd.Timedelta(interval)
        self._origin: pd.Timestamp | None = None
        self._result = None

    def _time_steps(self, index: pd.DatetimeIndex) -> np.ndarray:
        if self._origin is None:
            raise RuntimeError("Model has not been fitted.")

        elapsed = (index - self._origin) / self._interval
        return np.asarray(np.rint(elapsed), dtype=int)

    def _harmonic_features(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        steps = self._time_steps(index)
        features: dict[str, np.ndarray] = {}

        for k in range(1, self.daily_harmonics + 1):
            angle = 2.0 * np.pi * k * steps / 48
            features[f"daily_sin_{k}"] = np.sin(angle)
            features[f"daily_cos_{k}"] = np.cos(angle)

        for k in range(1, self.weekly_harmonics + 1):
            angle = 2.0 * np.pi * k * steps / 336
            features[f"weekly_sin_{k}"] = np.sin(angle)
            features[f"weekly_cos_{k}"] = np.cos(angle)

        return pd.DataFrame(features, index=index)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DHRArimaCrosscheck:
        index = pd.DatetimeIndex(X.index)

        if len(index) != len(y):
            raise ValueError("X and y must have the same number of observations.")

        if len(index) == 0:
            raise ValueError("Training data must not be empty.")

        self._origin = index[0]
        exog = self._harmonic_features(index)

        model = SARIMAX(
            np.asarray(y, dtype=float),
            exog=exog.to_numpy(),
            order=self.arima_order,
            trend="c",
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        self._result = model.fit(
            disp=False,
            method="bfgs",
            maxiter=300,
        )

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._result is None:
            raise RuntimeError("predict() called before fit()")

        index = pd.DatetimeIndex(X.index)
        exog = self._harmonic_features(index)

        forecast = self._result.get_forecast(
            steps=len(index),
            exog=exog.to_numpy(),
        )

        return np.asarray(forecast.predicted_mean, dtype=float)
