"""Arm A: frozen model — never retrain during the test stream."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class NeverRetrain(Adapter):
    """Return the fitted model unchanged for the entire test stream."""

    name = "never_retrain"

    def __init__(self) -> None:
        self._retrain_count = 0

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def adapt(
        self,
        changepoints: Sequence[int],
        model: Forecaster,
        data: pd.DataFrame,
    ) -> Forecaster:
        return model