"""Arm A -- never retrain (no-adaptation control).

Always returns the model unchanged, whatever `changepoints` holds. Exists
as a real `Adapter` so all four arms run through the identical runner
loop and report the same (all-zero) cost fields.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.retrain_accounting import RetrainAccounting
from drift_lab.forecasting.base import Forecaster


class NeverRetrain(RetrainAccounting):
    name = "never_retrain"

    def adapt(
        self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame
    ) -> Forecaster:
        return model
