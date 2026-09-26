"""Seasonal-naive forecaster using the value one season earlier."""

import numpy as np
import pandas as pd

from drift_lab.forecasting.base import Forecaster


class SeasonalNaive(Forecaster):
    """Forecast each timestamp with the observed value one season earlier.

    The project data is half-hourly, so ``season_length=48`` represents one
    day. Values supplied in ``X`` under the target column are treated as
    observed and can provide lags for later forecasts in the same batch.
    """

    name = "seasonal_naive"

    def __init__(self, season_length: int = 48) -> None:
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self._season = pd.Timedelta(minutes=30 * season_length)
        self._history: dict[pd.Timestamp, float] | None = None
        self._target_name: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaive":
        """Store the training observations for future seasonal lookup."""
        timestamps = self._as_datetime_index(X.index)
        aligned_y = y.reindex(timestamps)
        self._history = {
            timestamp: float(value)
            for timestamp, value in zip(timestamps, aligned_y.to_numpy())
            if not pd.isna(value)
        }
        self._target_name = y.name
        return self

    def observe(self, y: pd.Series) -> None:
        """Add already-observed values without changing the model."""
        if self._history is None:
            raise RuntimeError("observe() called before fit()")

        timestamps = self._as_datetime_index(y.index)
        for timestamp, value in zip(timestamps, y.to_numpy()):
            if not pd.isna(value):
                self._history[timestamp] = float(value)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return seasonal-lag forecasts in the original order of ``X``."""
        if self._history is None:
            raise RuntimeError("predict() called before fit()")

        timestamps = self._as_datetime_index(X.index)
        actuals: dict[pd.Timestamp, float] = {}
        if self._target_name is not None and self._target_name in X.columns:
            for timestamp, value in zip(timestamps, X[self._target_name].to_numpy()):
                if not pd.isna(value):
                    actuals[timestamp] = float(value)

        values = self._history.copy()
        forecasts: dict[pd.Timestamp, float] = {}
        for timestamp in timestamps.sort_values().unique():
            lag_timestamp = timestamp - self._season
            if lag_timestamp in actuals:
                forecast = actuals[lag_timestamp]
            else:
                forecast = values.get(lag_timestamp, np.nan)

            forecasts[timestamp] = forecast
            if timestamp not in actuals:
                values[timestamp] = forecast

        return np.asarray([forecasts[timestamp] for timestamp in timestamps], dtype=float)

    @staticmethod
    def _as_datetime_index(index) -> pd.DatetimeIndex:
        if pd.api.types.is_numeric_dtype(pd.Series(index)):
            raise ValueError("A datetime index is required for seasonal-naive forecasts")
        try:
            return pd.DatetimeIndex(index)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "A datetime index is required for seasonal-naive forecasts"
            ) from error
