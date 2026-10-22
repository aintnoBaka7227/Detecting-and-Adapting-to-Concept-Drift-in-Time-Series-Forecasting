"""UQ contract, frozen in Sprint 1 and extended additively on 2026-10-15:

    uq.quantify(point_forecast, calibration_residuals) -> UQResult(lower, upper, escalate)
    uq.observe(actual)        # feed back the last quantify call's outcomes

    for t in range(n):
        result = uq.quantify(forecast[t], calibration_residuals)
        uq.observe(actual[t])

A static method (fixed split conformal) ignores `observe`; an adaptive one
(AgACI) updates from it before the next issuance. `escalate` flags an
interval wider than `escalate_threshold`; the threshold sweep (T6, F5) is
evaluation.calculate_escalation_curve over saved intervals.

To add a method, subclass UncertaintyQuantifier and implement `quantify`
(see fixed_split_conformal.py). `calibration_residuals` must be residuals
already observed, never future test values.
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

    def observe(self, actual) -> None:
        """Feed back the outcomes of the most recent `quantify` call, in the
        same order as its forecasts. Static methods ignore them (this
        default); adaptive methods override it to update before the next
        issuance."""
