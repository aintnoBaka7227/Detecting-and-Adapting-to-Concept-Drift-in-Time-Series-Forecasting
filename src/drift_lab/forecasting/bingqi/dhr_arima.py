"""Independent DHR-ARIMA implementation for cross-validation.

Dynamic Harmonic Regression:
    Fourier daily/weekly seasonal terms
    +
    non-seasonal ARIMA errors.

This implementation is intentionally separate from the project's existing
DHRArima implementation so its results can be independently cross-checked.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from drift_lab.forecasting.base import Forecaster


def _build_fourier_terms(
    steps: np.ndarray,
    periods: tuple[int, ...],
    harmonics: tuple[int, ...],
) -> pd.DataFrame:
    """Generate sine/cosine Fourier regressors."""

    steps = np.asarray(steps, dtype=float)

    if len(periods) != len(harmonics):
        raise ValueError("periods and harmonics must have the same length")

    columns: dict[str, np.ndarray] = {}

    for period, n_harmonics in zip(periods, harmonics):
        if period <= 0:
            raise ValueError("period must be positive")

        if n_harmonics <= 0:
            raise ValueError("number of harmonics must be positive")

        for k in range(1, n_harmonics + 1):
            angle = 2.0 * np.pi * k * steps / period

            columns[f"sin_{period}_{k}"] = np.sin(angle)
            columns[f"cos_{period}_{k}"] = np.cos(angle)

    return pd.DataFrame(columns)


class BingqiDHRArima(Forecaster):
    """Dynamic Harmonic Regression with ARIMA errors.

    Default setup follows the shared project experiment configuration:

    - half-hourly demand data
    - daily period = 48
    - weekly period = 336
    - four Fourier harmonics for each period
    - ARIMA(2, 0, 2) residual model

    The implementation is deterministic; therefore repeated random seeds are
    not required for the model itself.
    """

    name = "bingqi_dhr_arima"

    def __init__(
        self,
        periods: tuple[int, ...] = (48, 336),
        n_harmonics: tuple[int, ...] = (4, 4),
        arima_order: tuple[int, int, int] = (2, 0, 2),
        interval: str = "30min",
    ) -> None:

        if len(periods) != len(n_harmonics):
            raise ValueError(
                "periods and n_harmonics must contain the same number of items"
            )

        self.periods = tuple(periods)
        self.n_harmonics = tuple(n_harmonics)
        self.arima_order = tuple(arima_order)
        self.interval = pd.Timedelta(interval)

        self._epoch: pd.Timestamp | None = None
        self._fitted = None

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> "BingqiDHRArima":
        """Fit DHR-ARIMA on the training window."""

        if len(y) == 0:
            raise ValueError("Training target y must not be empty")

        index = pd.DatetimeIndex(X.index)

        if len(index) != len(y):
            raise ValueError("X and y must have the same number of rows")

        if not index.is_monotonic_increasing:
            raise ValueError("Training timestamps must be sorted")

        y_values = np.asarray(y, dtype=float)

        if np.isnan(y_values).any():
            raise ValueError("Training target contains missing values")

        self._epoch = index[0]

        steps = self._elapsed_steps(index)

        exog = _build_fourier_terms(
            steps,
            self.periods,
            self.n_harmonics,
        )

        self._fitted = SARIMAX(
            y_values,
            exog=exog,
            order=self.arima_order,
            trend="c",
            seasonal_order=(0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(
            disp=False,
            method="bfgs",
            maxiter=300,
        )

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """Forecast the supplied timestamps."""

        if self._fitted is None:
            raise RuntimeError("predict() called before fit()")

        if len(X) == 0:
            return np.asarray([], dtype=float)

        index = pd.DatetimeIndex(X.index)

        steps = self._elapsed_steps(index)

        exog = _build_fourier_terms(
            steps,
            self.periods,
            self.n_harmonics,
        )

        forecast = self._fitted.get_forecast(
            steps=len(index),
            exog=exog,
        )

        return np.asarray(
            forecast.predicted_mean,
            dtype=float,
        )

    def _elapsed_steps(
        self,
        index: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Convert timestamps to half-hour step numbers."""

        if self._epoch is None:
            raise RuntimeError("Model epoch is not initialised")

        delta = (
            pd.Series(index)
            - self._epoch
        ) / self.interval

        return delta.round().astype(int).to_numpy()