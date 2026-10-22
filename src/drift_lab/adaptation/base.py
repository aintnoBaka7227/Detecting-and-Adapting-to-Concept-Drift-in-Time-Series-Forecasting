"""Adaptation contract, frozen in Sprint 1:

    adapter(changepoints, model, data) -> updated model

Each arm is its own small Adapter subclass (e.g. never_retrain.py,
retrain_using_recent_window.py), not a branch inside one class. Retrain cost
must stay readable afterwards (retrain_accounting.py), because T4 reports
cost next to accuracy.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from drift_lab.forecasting.base import Forecaster


class Adapter(ABC):
    @abstractmethod
    def adapt(
        self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame
    ) -> Forecaster:
        """Return a (possibly retrained) model in response to `changepoints`.

        Must track every retrain it performs (e.g. self.retrain_count) —
        Step 5 requires reporting retrain count per arm alongside accuracy,
        a retrain is not free.
        """
        raise NotImplementedError
