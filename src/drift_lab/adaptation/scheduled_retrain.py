"""Arm B: scheduled retraining using all observed history."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class ScheduledRetrain(Adapter):
    """Refit on all available history when the runner supplies a schedule point."""

    name = "scheduled_retrain"

    def __init__(self, interval_days: int = 30, target_column: str | None = None) -> None:
        if interval_days <= 0:
            raise ValueError("interval_days must be positive")

        self.interval_days = interval_days
        self.target_column = target_column
        self._retrain_count = 0
        self._train_samples = 0

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    @property
    def train_samples(self) -> int:
        """Cumulative samples consumed by scheduled retraining."""
        return self._train_samples

    def adapt(
        self,
        changepoints: Sequence[int],
        model: Forecaster,
        data: pd.DataFrame,
    ) -> Forecaster:
        if not changepoints:
            return model

        retrain_position = max(changepoints)
        history = data.iloc[: retrain_position + 1]

        y = self._target(history)
        X = pd.DataFrame(index=history.index)

        self._retrain_count += 1
        self._train_samples += len(history)

        return model.fit(X, y)

    def _target(self, data: pd.DataFrame) -> pd.Series:
        if self.target_column is not None:
            return data[self.target_column]

        if data.shape[1] == 1:
            return data.iloc[:, 0]

        raise ValueError("data has more than one column; set target_column")