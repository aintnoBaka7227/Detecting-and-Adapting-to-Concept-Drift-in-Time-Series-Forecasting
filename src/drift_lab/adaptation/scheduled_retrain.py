"""Arm B -- scheduled retrain: refit on a fixed calendar cadence, on the
full history seen so far, regardless of `changepoints`.

No signature change to the frozen `adapt(changepoints, model, data)`
contract -- `changepoints` is accepted (the contract requires it) but
ignored entirely: this arm reacts to the calendar, not to drift. The
intended usage is "predict `interval_days` -> retrain -> predict the next
`interval_days` -> repeat", so the first call always retrains (there is
nothing to compare it to yet) and later calls retrain only once
`interval_days` have elapsed since the last retrain -- calling `adapt`
more often than the schedule (e.g. once per day while the schedule is
30-day) is safe and just no-ops in between. `data` needs a `DatetimeIndex`
and the target as its only column or `target_column`, same as
`RetrainUsing3MonthWindows`.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class ScheduledRetrain(Adapter):
    name = "scheduled_retrain_30d"

    def __init__(self, interval_days: int = 30, target_column: str | None = None) -> None:
        self.interval_days = interval_days
        self.target_column = target_column
        self._retrain_count = 0
        self._last_retrain_date: pd.Timestamp | None = None

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def adapt(
        self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame
    ) -> Forecaster:
        """Retrain on the full history in `data` every `interval_days`,
        ignoring `changepoints`. The first call always retrains (there is
        no prior retrain date to measure against); after that, a call
        is a no-op unless at least `interval_days` have passed since the
        last retrain.
        """
        index = pd.DatetimeIndex(data.index)
        now = index.max()

        if self._last_retrain_date is not None:
            elapsed_days = (now - self._last_retrain_date).days
            if elapsed_days < self.interval_days:
                return model

        y = self._target(data)
        X = pd.DataFrame(index=data.index)

        self._retrain_count += 1
        self._last_retrain_date = now
        return model.fit(X, y)

    def _target(self, data: pd.DataFrame) -> pd.Series:
        if self.target_column is not None:
            return data[self.target_column]
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError("data has more than one column; set target_column")
