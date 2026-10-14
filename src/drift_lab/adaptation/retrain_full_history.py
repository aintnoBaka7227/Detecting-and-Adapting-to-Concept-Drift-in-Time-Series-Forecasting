"""Arm C: retrain on drift using all history observed up to the detection."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class RetrainFullHistory(Adapter):
    """Refit the forecasting model on all available history after a drift."""

    name = "retrain_full_history"

    def __init__(self, target_column: str | None = None) -> None:
        self.target_column = target_column
        self._retrain_count = 0
        self._train_samples = 0

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    @property
    def train_samples(self) -> int:
        """Cumulative number of samples consumed by adaptation retrains."""
        return self._train_samples

    def adapt(
        self,
        changepoints: Sequence[int],
        model: Forecaster,
        data: pd.DataFrame,
    ) -> Forecaster:
        if not changepoints:
            return model

        detection_position = max(changepoints)
        history = data.iloc[: detection_position + 1]

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