from __future__ import annotations

from collections.abc import Sequence
import time

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class ArmDRecentWindow(Adapter):
    """
    Arm D: retrain on every accepted drift alarm using only the
    most recent N calendar days of observed data.

    The detector is NOT run here. The caller supplies the accepted
    ADWIN changepoints.
    """

    def __init__(
        self,
        window_days: int,
        target_column: str | None = None,
    ) -> None:
        self.window_days = window_days
        self.target_column = target_column

        self._retrain_count = 0
        self._train_samples = 0
        self._wall_clock_s = 0.0
        self._retrain_timestamps = []

    @property
    def name(self) -> str:
        return f"D_drift_recent_{self.window_days}d"

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

        if not changepoints:
            return model

        index = pd.DatetimeIndex(data.index)

        latest_position = max(changepoints)
        detection_date = index[latest_position]

        window_start = detection_date.normalize() - pd.Timedelta(
            days=self.window_days - 1
        )

        recent = data.loc[
            (index >= window_start)
            & (index <= detection_date)
        ]

        y = self._target(recent)
        X = pd.DataFrame(index=recent.index)

        start_time = time.perf_counter()

        updated_model = model.fit(X, y)

        elapsed = time.perf_counter() - start_time

        self._retrain_count += 1
        self._train_samples += len(recent)
        self._retrain_timestamps.append(pd.Timestamp(detection_date))
        self._wall_clock_s += elapsed

        return updated_model

    def _target(self, recent: pd.DataFrame) -> pd.Series:
        if self.target_column is not None:
            return recent[self.target_column]

        if recent.shape[1] == 1:
            return recent.iloc[:, 0]

        raise ValueError(
            "data has more than one column; set target_column"
        )