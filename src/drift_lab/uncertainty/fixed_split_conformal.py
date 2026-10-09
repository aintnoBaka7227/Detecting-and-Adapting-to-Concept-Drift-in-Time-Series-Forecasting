"""Fixed split conformal intervals: one radius, calibrated once.

The radius is the finite-sample conformal quantile of the absolute
calibration residuals |actual - issued forecast|. It is set by the first
calibration and never moves afterwards -- test outcomes are not an input
to this class. Every interval is `point forecast ± radius`.

Lifecycle (the frozen `quantify` contract passes residuals on every call):

    uq = FixedSplitConformal()
    uq.quantify(forecasts, calibration_residuals)   # first call calibrates
    uq.quantify(more_forecasts, anything)           # radius unchanged

`quantify` calibrates only while the instance is uncalibrated; later calls
ignore `calibration_residuals`. Call `calibrate` directly to calibrate
ahead of time, and `reset` (or a new instance) to start again. Use one
instance per forecaster x region x arm x seed x configuration.

The standard split-conformal coverage guarantee assumes exchangeable
scores, which a drifting series does not provide; this is the baseline
the adaptive method is compared with, not a guaranteed 90% interval.
"""

from __future__ import annotations

import math

import numpy as np

from drift_lab.uncertainty.base import UncertaintyQuantifier, UQResult
from drift_lab.uncertainty.conformal_common import (
    absolute_scores,
    conformal_radius,
    escalation_flags,
    point_forecast_array,
    validate_escalate_threshold,
    validate_level,
)


class FixedSplitConformal(UncertaintyQuantifier):
    """Split conformal with a radius fixed at calibration.

    Parameters
    ----------
    alpha : float, default 0.10
        Nominal miscoverage; 0.10 gives a nominal 90% interval.
    escalate_threshold : float | None, default None
        Escalate a forecast when its interval width exceeds this value.
        None disables escalation (all flags False). The width is constant
        here, so a width rule flags every forecast or none.
    """

    name = "fixed_split_conformal"

    def __init__(self, alpha: float = 0.10, escalate_threshold: float | None = None) -> None:
        self.alpha = validate_level(alpha)
        self.escalate_threshold = validate_escalate_threshold(escalate_threshold)
        self._radius: float | None = None
        self._n_calibration = 0

    @property
    def is_calibrated(self) -> bool:
        return self._radius is not None

    @property
    def radius(self) -> float:
        if self._radius is None:
            raise RuntimeError("radius requested before calibrate()")
        return self._radius

    @property
    def n_calibration(self) -> int:
        """Number of calibration scores the radius was taken from."""
        return self._n_calibration

    def calibrate(self, calibration_residuals) -> "FixedSplitConformal":
        """Set the radius from residuals of forecasts issued without access
        to their outcomes. With too few scores for the requested level the
        radius is infinite (strict conformal convention), never capped."""
        scores = absolute_scores(calibration_residuals)
        self._radius = conformal_radius(scores, self.alpha)
        self._n_calibration = len(scores)
        return self

    def reset(self) -> None:
        self._radius = None
        self._n_calibration = 0

    def predict(self, point_forecast) -> UQResult:
        """Bounds for `point_forecast` from the stored radius."""
        if self._radius is None:
            raise RuntimeError("predict() called before calibrate()")
        forecasts = point_forecast_array(point_forecast)
        return UQResult(
            lower=forecasts - self._radius,
            upper=forecasts + self._radius,
            escalate=escalation_flags(len(forecasts), self._radius, self.escalate_threshold),
        )

    def quantify(self, point_forecast: np.ndarray, calibration_residuals: np.ndarray) -> UQResult:
        if self._radius is None:
            self.calibrate(calibration_residuals)
        return self.predict(point_forecast)

    def describe(self) -> dict:
        """Configuration and calibrated state, for run logging."""
        return {
            "uq_method": self.name,
            "alpha": self.alpha,
            "escalate_threshold": self.escalate_threshold,
            "radius": self._radius,
            "radius_is_infinite": self._radius is not None and math.isinf(self._radius),
            "n_calibration": self._n_calibration,
        }
