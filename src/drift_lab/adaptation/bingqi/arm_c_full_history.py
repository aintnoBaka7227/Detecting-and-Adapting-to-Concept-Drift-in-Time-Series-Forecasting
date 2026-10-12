from __future__ import annotations

from collections.abc import Sequence
import time

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class ArmCFullHistory(Adapter):
    """
    Arm C: retrain on every accepted drift alarm using all
    observations available up to the latest changepoint.

    The detector is NOT run here. The caller supplies the accepted
    ADWIN changepoints.
    """

    name = "C_drift_full_history"

    def __init__(self, target_column: str | None = None) -> None:
        self.target_column = target_column

        self._retrain_count = 0
        self._train_samples = 0
        self._wall_clock_s = 0.0
        self._retrain_timestamps = []

    @property
    def retrain_count(self) -> int:
        return self._retrain_count
    @property
    def retrain_timestamps(self) -> list[pd.Timestamp]:
        return self._retrain_timestamps

    @property
    def train_samples(self) -> int:
        return self._train_samples

    @property
    def fit_seconds(self) -> float:
        return self._wall_clock_s

    @property
    def wall_clock_s(self) -> float:
        return self._wall_clock_s

    def adapt(
        self,
        changepoints: Sequence[int],
        model: Forecaster,
        data: pd.DataFrame,
    ) -> Forecaster:
        """
        Retrain using all observed history up to the latest changepoint.

        If there is no changepoint, return the model unchanged.
        """
        if not changepoints:
            return model

        index = pd.DatetimeIndex(data.index)

        latest_position = max(changepoints)
        detection_date = index[latest_position]

        history = data.loc[index <= detection_date]

        y = self._target(history)
        X = pd.DataFrame(index=history.index)

        start_time = time.perf_counter()

        updated_model = model.fit(X, y)

        elapsed = time.perf_counter() - start_time

        self._retrain_count += 1
        self._train_samples += len(history)
        self._wall_clock_s += elapsed
        self._retrain_timestamps.append(pd.Timestamp(detection_date))

        return updated_model

    def _target(self, history: pd.DataFrame) -> pd.Series:
        if self.target_column is not None:
            return history[self.target_column]

        if history.shape[1] == 1:
            return history.iloc[:, 0]

        raise ValueError(
            "data has more than one column; set target_column"
        )