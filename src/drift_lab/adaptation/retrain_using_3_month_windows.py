"""Retrain-on-drift adaptation arm: refit on the trailing 3 calendar months.

No signature change to the frozen `adapt(changepoints, model, data)`
contract -- `data` is the history the caller has already sliced up to
"now"; this arm just decides how much of it to retrain on. Anchors the
window on the *most recent* changepoint, allows reaching back across
split boundaries (all past, already-observed data), and never throttles
-- controlling detector cadence is the caller's job (see
`aemo/deseasonalise.py`). `data` needs a `DatetimeIndex` and the target as
its only column or `target_column`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class RetrainUsing3MonthWindows(Adapter):
    name = "retrain_3mo_window"

    def __init__(
        self,
        window_months: int = 3,
        target_column: str | None = None,
    ) -> None:
        self.window_months = window_months
        self.target_column = target_column
        self._retrain_count = 0
        self._train_samples = 0

    @property
    def retrain_count(self) -> int:
        """Number of adaptation retrains performed."""
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
        """Retrain the model on the trailing window before the latest drift.

        No-op and return the model unchanged when no changepoints are
        supplied.
        """
        if not changepoints:
            return model

        index = pd.DatetimeIndex(data.index)
        detection_position = max(changepoints)
        detection_date = index[detection_position]

        window_start = detection_date - pd.DateOffset(months=self.window_months)

        window = data.loc[
            (index > window_start) & (index <= detection_date)
        ]

        y = self._target(window)
        X = pd.DataFrame(index=window.index)

        self._retrain_count += 1
        self._train_samples += len(window)

        return model.fit(X, y)

    def _target(self, window: pd.DataFrame) -> pd.Series:
        """Return the configured forecasting target."""
        if self.target_column is not None:
            return window[self.target_column]

        if window.shape[1] == 1:
            return window.iloc[:, 0]

        raise ValueError(
            "data has more than one column; set target_column"
        )