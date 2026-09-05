"""Seasonal-naive baseline: forecast = the observed value one season ago.

For half-hourly demand a season is one day (48 intervals), so the forecast
for time ``t`` is the demand at ``t - 24h``. `fit` only records the training
history; `observe` extends it (the calibration window) so the first test
forecasts resolve across the train->test boundary; `predict` reads the lag,
preferring an observed target column in ``X`` when present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_lab.config import SEASON_LENGTH
from drift_lab.forecasting.base import Forecaster


class SeasonalNaive(Forecaster):
    name = "seasonal_naive"

    def __init__(self, season_length: int = SEASON_LENGTH) -> None:
        if season_length < 1:
            raise ValueError("season_length must be >= 1")
        self.season_length = season_length
        self._history: pd.Series | None = None
        self._target_name: str | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> SeasonalNaive:
        self._history = self._as_series(y)
        self._target_name = y.name
        return self

    def observe(self, y: pd.Series) -> None:
        if self._history is None:
            raise RuntimeError("observe() called before fit()")
        self._history = self._append(self._history, self._as_series(y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("predict() called before fit()")

        index = pd.DatetimeIndex(X.index)
        known = self._history
        if self._target_name is not None and self._target_name in X.columns:
            observed = pd.Series(
                np.asarray(X[self._target_name], dtype=float), index=index
            ).dropna()
            known = self._append(known, observed)

        # Positional lag over the continuous half-hourly grid: one step back
        # per interval, so season_length steps == one season ago.
        grid = known.reindex(known.index.union(index)).sort_index()
        return grid.shift(self.season_length).reindex(index).to_numpy()

    @staticmethod
    def _as_series(y: pd.Series) -> pd.Series:
        return pd.Series(
            np.asarray(y, dtype=float),
            index=pd.DatetimeIndex(y.index),
            name=getattr(y, "name", None),
        )

    @staticmethod
    def _append(history: pd.Series, extra: pd.Series) -> pd.Series:
        extra = extra[~extra.index.isin(history.index)]
        return pd.concat([history, extra]).sort_index()
