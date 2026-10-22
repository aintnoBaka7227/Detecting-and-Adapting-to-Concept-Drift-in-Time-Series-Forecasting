"""Split conformal prediction intervals (stateless version).

Idea: a model is trained on one part of the data, then its mistakes on a
separate calibration part are used to decide how wide an interval must be.
If 90% of the calibration mistakes were smaller than q, then
`forecast +/- q` should contain the truth about 90% of the time.

Steps inside `quantify`:

    1. score    = |residual| for every calibration residual
    2. rank     = ceil((n + 1) * coverage)         (finite-sample correction)
    3. q        = the rank-th smallest score       (infinite if rank > n)
    4. interval = [forecast - q, forecast + q]
    5. escalate = interval width > escalate_threshold (never, if no threshold)

Nothing is remembered between calls: the radius is rebuilt from whatever
`calibration_residuals` the caller passes. The caller is responsible for
passing only residuals that were already observed (the calibration window),
never test-period values, or the leakage audit fails.

Because the radius is the same for every forecast, every interval has the
same width, so a width-based escalate rule flags all forecasts or none.

    uq = SplitConformal(coverage=0.90)
    result = uq.quantify(point_forecast, calibration_residuals)
    result.lower, result.upper, result.escalate
"""

from __future__ import annotations

import math

import numpy as np

from drift_lab.uncertainty.base import UncertaintyQuantifier, UQResult


class SplitConformal(UncertaintyQuantifier):
    """Split conformal intervals of the form `forecast +/- q`.

    Parameters
    ----------
    coverage : float, default 0.90
        Target share of true values the interval should contain, strictly
        between 0 and 1.
    escalate_threshold : float | None, default None
        Flag a forecast for review when its interval width exceeds this
        value. None means never escalate.
    """

    name = "split_conformal"

    def __init__(self, coverage: float = 0.90, escalate_threshold: float | None = None) -> None:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must be strictly between 0 and 1.")
        if escalate_threshold is not None and not escalate_threshold >= 0.0:
            raise ValueError("escalate_threshold must be a non-negative number or None.")
        self.coverage = float(coverage)
        self.escalate_threshold = None if escalate_threshold is None else float(escalate_threshold)

    def radius(self, calibration_residuals: np.ndarray) -> float:
        """The conformal radius q for these calibration residuals.

        Returns infinity when there are too few residuals to support the
        requested coverage (rank > n); the radius is never capped.
        """
        scores = np.abs(np.asarray(calibration_residuals, dtype=float))
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("calibration_residuals must be a non-empty 1-D array.")
        if not np.isfinite(scores).all():
            raise ValueError("calibration_residuals must be finite.")

        n = scores.size
        # round() first so float noise such as 90.00000000000001 cannot add one
        rank = math.ceil(round((n + 1) * self.coverage, 9))
        if rank > n:
            return math.inf
        return float(np.sort(scores)[rank - 1])

    def quantify(self, point_forecast: np.ndarray, calibration_residuals: np.ndarray) -> UQResult:
        forecast = np.asarray(point_forecast, dtype=float)
        if forecast.ndim != 1:
            raise ValueError("point_forecast must be one-dimensional.")
        if not np.isfinite(forecast).all():
            raise ValueError("point_forecast must be finite.")

        q = self.radius(calibration_residuals)
        lower, upper = forecast - q, forecast + q

        if self.escalate_threshold is None:
            escalate = np.zeros(forecast.shape, dtype=bool)
        else:
            escalate = (upper - lower) > self.escalate_threshold

        return UQResult(lower=lower, upper=upper, escalate=escalate)
