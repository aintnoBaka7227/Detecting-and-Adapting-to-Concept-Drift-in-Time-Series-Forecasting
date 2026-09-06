"""FROZEN CONTRACT (Sprint 1 — do not change the shape after Checkpoint 1):

    adapter(changepoints, model, data) -> updated model

See docs/interfaces.md for the full rationale.

How to implement an adaptation arm (Sprint 4, Step 5)
-------------------------------------------------------
Add a new file next to this one, e.g. `recent_window.py`, and subclass
`Adapter`. The four arms required by Step 5 (never retrain, retrain on a
schedule, retrain on drift with full history, retrain on drift with a
recent window) are each a separate small class, not branches inside one:

    import pandas as pd
    from drift_lab.adaptation.base import Adapter
    from drift_lab.forecasting.base import Forecaster

    class RecentWindowRetrain(Adapter):
        def __init__(self, window_days: int = 60):
            self.window_days = window_days
            self.retrain_count = 0

        def adapt(self, changepoints, model: Forecaster, data: pd.DataFrame) -> Forecaster:
            if not changepoints:
                return model
            recent = data.tail(self.window_days * 48)  # e.g. half-hourly data
            self.retrain_count += 1
            return model.fit(recent[feature_cols], recent[target_col])

`retrain_count` (or equivalent) must be readable after the fact — Step 5's
four-arm comparison reports retrain cost next to accuracy, not accuracy
alone.
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
