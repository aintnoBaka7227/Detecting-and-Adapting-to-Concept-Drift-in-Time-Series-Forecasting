"""Arm C -- on an accepted drift alarm, retrain on the full history.

For an accepted alarm on day `d` the caller slices `data` to everything
observable through the end of `d`; this arm refits on all of it, back to
the first observation (TRAIN start). No cooldown -- the 14-day refractory
filter upstream is what limits retraining frequency.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.retrain_accounting import RetrainAccounting
from drift_lab.forecasting.base import Forecaster


class RetrainUsingFullHistory(RetrainAccounting):
    name = "retrain_full_history"

    def adapt(
        self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame
    ) -> Forecaster:
        if not changepoints:
            return model
        boundary = pd.DatetimeIndex(data.index)[max(changepoints)]
        return self._refit(model, data.loc[data.index <= boundary])
