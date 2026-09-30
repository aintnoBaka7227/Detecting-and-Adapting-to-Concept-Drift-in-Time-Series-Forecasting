"""Arm B -- retrain every 30 days, no matter what (scheduled control).

The model trained on the TRAIN split forecasts the first 30 days of TEST;
at the end of day 30 it is refitted on the last 30 days of observed data
only (the 30 days it just forecast), forecasts the next 30 days, and so on.

Owns its own calendar: `schedule(test_start, test_end)` returns the
retraining days (day 30, 60, ... of the TEST stream), independent of any
detector. On scheduled day `d` the training window is the `window_days`
calendar days ending at the end of `d` (`d - window_days + 1` 00:00
through `d` 23:30) -- never the full history (that is arm C).

`changepoints` is the frozen contract's positional trigger list; here it
marks the scheduled boundary (`len(data) - 1`), not a detection.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.retrain_accounting import RetrainAccounting
from drift_lab.forecasting.base import Forecaster


class RetrainEvery30Days(RetrainAccounting):
    name = "retrain_every_30_days"

    def __init__(
        self,
        interval_days: int = 30,
        window_days: int = 30,
        target_column: str | None = None,
    ) -> None:
        super().__init__(target_column=target_column)
        if interval_days < 1:
            raise ValueError("interval_days must be >= 1")
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        self.interval_days = interval_days
        self.window_days = window_days

    def schedule(self, test_start, test_end) -> list[pd.Timestamp]:
        """Calendar days on which to retrain: the last day of every full
        `interval_days` block counted from `test_start`, up to (not
        including) the day of `test_end` -- a retrain on the final day
        would never forecast anything."""
        start = pd.Timestamp(test_start).normalize()
        last_day = pd.Timestamp(test_end).normalize()
        days = []
        day = start + pd.Timedelta(days=self.interval_days - 1)
        while day < last_day:
            days.append(day)
            day += pd.Timedelta(days=self.interval_days)
        return days

    def adapt(
        self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame
    ) -> Forecaster:
        if not changepoints:
            return model
        index = pd.DatetimeIndex(data.index)
        boundary = index[max(changepoints)]
        window_start = boundary.normalize() - pd.Timedelta(days=self.window_days - 1)
        window = data.loc[(index >= window_start) & (index <= boundary)]
        return self._refit(model, window)
