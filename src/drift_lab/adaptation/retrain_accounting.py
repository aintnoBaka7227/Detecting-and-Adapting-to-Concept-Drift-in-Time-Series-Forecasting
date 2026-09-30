"""Shared retrain bookkeeping for the four Step 5 adaptation arms.

Not an arm itself. Every arm (A-D) inherits this so the cost fields T4
needs are measured the same way everywhere:

- `retrain_count`: TEST-period retrains only; the initial TRAIN fit is
  done by the runner, never through an adapter, so it is never counted.
- `train_samples`: sum of observations passed to every retrain `fit()`
  (a repeated observation counts again -- it is trained on again).
- `fit_seconds`: summed wall-clock of those `fit()` calls only.
- `retrain_timestamps`: the last observation each retrain saw, i.e. the
  retraining boundary (end of alarm/schedule day `d`).

`data` follows the frozen `adapt(changepoints, model, data)` contract used
by `RetrainUsing3MonthWindows`: a `DatetimeIndex`, target as its only
column (or `target_column=`), already sliced by the caller to what is
observable at the boundary.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class RetrainAccounting(Adapter):
    def __init__(self, target_column: str | None = None) -> None:
        self.target_column = target_column
        self._retrain_count = 0
        self._train_samples = 0
        self._fit_seconds = 0.0
        self._retrain_timestamps: list[pd.Timestamp] = []

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    @property
    def train_samples(self) -> int:
        return self._train_samples

    @property
    def fit_seconds(self) -> float:
        return self._fit_seconds

    @property
    def retrain_timestamps(self) -> list[pd.Timestamp]:
        return list(self._retrain_timestamps)

    def _refit(self, model: Forecaster, window: pd.DataFrame) -> Forecaster:
        y = self._target(window)
        X = pd.DataFrame(index=window.index)

        t0 = time.perf_counter()
        model = model.fit(X, y)
        self._fit_seconds += time.perf_counter() - t0

        self._retrain_count += 1
        self._train_samples += len(y)
        self._retrain_timestamps.append(pd.Timestamp(window.index.max()))
        return model

    def _target(self, window: pd.DataFrame) -> pd.Series:
        if self.target_column is not None:
            return window[self.target_column]
        if window.shape[1] == 1:
            return window.iloc[:, 0]
        raise ValueError("data has more than one column; set target_column")
