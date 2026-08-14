"""FROZEN CONTRACT (Sprint 1 — do not change the shape after Checkpoint 1):

    uq(point_forecast, calibration_residuals) -> lower bound, upper bound, escalate flag

The escalate flag is returned by `quantify()` itself, not by a separate
top-level "escalation" module — the slide's contract bundles it into UQ.
The Step 6 threshold-sweep analysis (coverage vs. fraction escalated)
belongs in a future evaluation/ module that calls `quantify()` at many
thresholds — it's a report, not part of the frozen interface, and isn't
needed to kickstart Sprint 1.

See docs/interfaces.md for the full rationale.

How to implement a UQ method (Sprint 5, Step 6)
-------------------------------------------------
Add a new file next to this one, e.g. `conformal.py`, and subclass
`UncertaintyQuantifier`:

    import numpy as np
    from drift_forecasting.uncertainty.base import UncertaintyQuantifier, UQResult

    class SplitConformal(UncertaintyQuantifier):
        def __init__(self, coverage: float = 0.90, escalate_threshold: float | None = None):
            self.coverage = coverage
            self.escalate_threshold = escalate_threshold

        def quantify(self, point_forecast, calibration_residuals) -> UQResult:
            q = np.quantile(np.abs(calibration_residuals), self.coverage)
            lower, upper = point_forecast - q, point_forecast + q
            width = upper - lower
            threshold = self.escalate_threshold or np.quantile(width, 0.9)
            escalate = width > threshold
            return UQResult(lower=lower, upper=upper, escalate=escalate)

`calibration_residuals` must only ever be residuals already observed —
the calibration window for the static method, or test-stream-so-far for
the adaptive one. Never future test-period values (Step 7 leakage audit).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class UQResult:
    lower: np.ndarray
    upper: np.ndarray
    escalate: np.ndarray  # bool array, one entry per forecast


class UncertaintyQuantifier(ABC):
    @abstractmethod
    def quantify(self, point_forecast: np.ndarray, calibration_residuals: np.ndarray) -> UQResult:
        """Attach a prediction interval and an escalate flag to each forecast.

        `calibration_residuals` must come only from the calibration window
        (or, for the adaptive variant, from residuals already observed in
        the test stream) — never from future test-period values, or the
        Step 7 leakage audit will fail on this function.
        """
        raise NotImplementedError
