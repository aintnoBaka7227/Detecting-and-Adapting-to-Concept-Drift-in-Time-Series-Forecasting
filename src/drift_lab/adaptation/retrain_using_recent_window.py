"""Arm D -- on an accepted drift alarm, retrain on a recent window only.

`window_days` is the swept parameter (30 / 60 / 120 / 180, see
TEAM_DECISIONS.md 5.1): each value is its own run, and the official arm D
is picked afterwards from the logged results. For an alarm on day `d`
the window is the `window_days` calendar days ending at the end of `d`
(`d - window_days + 1` 00:00 through `d` 23:30). It may reach back into
Calibration or TRAIN -- already observed, not leakage -- but never past
the boundary.

Kept separate from `RetrainUsing3MonthWindows` (calendar months,
smoke-test runner), which is left unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.retrain_accounting import RetrainAccounting
from drift_lab.forecasting.base import Forecaster

SWEEP_WINDOW_DAYS = (30, 60, 120, 180)


class RetrainUsingRecentWindow(RetrainAccounting):
    name = "retrain_recent_window"

    def __init__(self, window_days: int, target_column: str | None = None) -> None:
        super().__init__(target_column=target_column)
        if window_days < 1:
            raise ValueError("window_days must be >= 1")
        self.window_days = window_days

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
