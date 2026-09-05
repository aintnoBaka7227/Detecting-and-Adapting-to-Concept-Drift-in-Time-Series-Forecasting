"""Dynamic harmonic regression + ARIMA errors (classical baseline).

Fourier terms for the daily and weekly cycles enter a SARIMAX model as
exogenous regressors while an ARMA process absorbs the short-range error
structure. Fourier phase is measured in real elapsed half-hour steps from a
fixed epoch, so the daily/weekly signal stays aligned across the gap between
the training window and the test period (the calibration months sit in that
gap). `fit` uses the training window only; `predict` then rolls the frozen
model forward through the test stream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from drift_lab.forecasting.base import Forecaster


def _fourier_terms(steps, periods, n_harmonics) -> pd.DataFrame:
    steps = np.asarray(steps)
    columns: dict[str, np.ndarray] = {}
    for period, n in zip(periods, n_harmonics):
        for k in range(1, n + 1):
            columns[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * steps / period)
            columns[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * steps / period)
    return pd.DataFrame(columns)


class DHRArima(Forecaster):
    name = "dhr_arima"

    def __init__(
        self,
        periods: tuple[int, ...] = (48, 336),
        n_harmonics: tuple[int, ...] = (4, 4),
        arima_order: tuple[int, int, int] = (2, 0, 2),
        interval: str = "30min",
    ) -> None:
        self.periods = list(periods)
        self.n_harmonics = list(n_harmonics)
        self.arima_order = tuple(arima_order)
        self._interval = pd.Timedelta(interval)
        self._epoch: pd.Timestamp | None = None
        self._fitted = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> DHRArima:
        index = pd.DatetimeIndex(X.index)
        self._epoch = index[0]
        exog = _fourier_terms(self._elapsed(index), self.periods, self.n_harmonics)
        self._fitted = SARIMAX(
            np.asarray(y, dtype=float),
            exog=exog,
            order=self.arima_order,
            trend="c",
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, method="bfgs", maxiter=300)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._fitted is None:
            raise RuntimeError("predict() called before fit()")
        index = pd.DatetimeIndex(X.index)
        exog = _fourier_terms(self._elapsed(index), self.periods, self.n_harmonics)
        forecast = self._fitted.get_forecast(steps=len(index), exog=exog)
        return np.asarray(forecast.predicted_mean)

    def _elapsed(self, index: pd.DatetimeIndex) -> np.ndarray:
        delta = (pd.Series(index) - self._epoch) / self._interval
        return delta.round().astype(int).to_numpy()
