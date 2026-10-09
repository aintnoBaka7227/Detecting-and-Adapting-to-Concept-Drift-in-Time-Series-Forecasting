"""Adaptive conformal inference (ACI) over a rolling buffer of issued errors.

Two things adapt, and both only ever change *future* intervals:

    buffer          the latest `window_size` absolute errors
                    |actual - originally issued forecast|, seeded from the
                    end of the calibration residuals and then fed by test
                    outcomes as they arrive;
    alpha_current   the miscoverage level the radius is read at, moved by
                    the published ACI rule (Gibbs & Candes 2021)

                        alpha_current += gamma * (target_alpha - miss)

                    so a miss asks for a higher quantile and a covered
                    outcome a lower one.

The radius at issuance is the finite-sample conformal quantile of the
buffer at level 1 - alpha_current. Extreme levels follow the published
conventions and are never clipped:

    alpha_current <= 0                          infinite interval
    alpha_current >= 1                          zero-radius interval
    rank exceeds the number of buffered scores  infinite interval

Lifecycle (the frozen `quantify` contract passes residuals on every call):

    uq = AdaptiveConformalInference(gamma=..., window_size=...)
    issued = uq.quantify(block_forecasts, calibration_residuals)  # seeds once
    ...outcomes for that block arrive...
    uq.observe(actuals, block_forecasts, issued.lower, issued.upper)
    issued = uq.quantify(next_block_forecasts, calibration_residuals)

`quantify` / `predict` never change state: calling them twice without an
`observe` in between returns the same bounds. `quantify` seeds the buffer
only while the instance is uncalibrated, so repeated calls cannot reset
the run. Every forecast in one call shares the radius available at that
issuance; feed a block's outcomes to `observe` only once they would
really have been known, and before the next issuance. `observe` judges
each outcome against the bounds that were actually issued for it, never
against recomputed ones.

`gamma` and `window_size` have no defaults: choose them on historical
validation data, freeze them, and log them. Use one instance per
forecaster x region x arm x seed x configuration (and per horizon when
horizons are evaluated separately); a forecaster retrain does not reset
this state. This is ACI around a rolling split-conformal score buffer,
not EnbPI.
"""

from __future__ import annotations

import math
from collections import deque

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


class AdaptiveConformalInference(UncertaintyQuantifier):
    """ACI with a rolling buffer of absolute issued-forecast errors.

    Parameters
    ----------
    gamma : float
        ACI learning rate; finite and positive.
    window_size : int
        Buffer length W in scores. On a complete half-hourly single-horizon
        stream W = 48 * days.
    target_alpha : float, default 0.10
        Nominal miscoverage the feedback rule steers towards.
    escalate_threshold : float | None, default None
        Escalate a forecast when its interval width exceeds this value.
        None disables escalation (all flags False).
    """

    name = "adaptive_conformal_inference"

    def __init__(
        self,
        gamma: float,
        window_size: int,
        target_alpha: float = 0.10,
        escalate_threshold: float | None = None,
    ) -> None:
        gamma = float(gamma)
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be a finite positive number.")
        if isinstance(window_size, bool) or int(window_size) != window_size or window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        self.gamma = gamma
        self.window_size = int(window_size)
        self.target_alpha = validate_level(target_alpha, "target_alpha")
        self.escalate_threshold = validate_escalate_threshold(escalate_threshold)
        self._buffer: deque[float] | None = None
        self._alpha_current = self.target_alpha
        self._feedback_log: list[dict] = []

    @property
    def is_calibrated(self) -> bool:
        return self._buffer is not None

    @property
    def alpha_current(self) -> float:
        return self._alpha_current

    @property
    def buffer(self) -> np.ndarray:
        """The buffered scores, oldest first (a copy)."""
        if self._buffer is None:
            raise RuntimeError("buffer requested before calibrate()")
        return np.array(self._buffer, dtype=float)

    @property
    def radius(self) -> float:
        """Radius the next issuance would use."""
        if self._buffer is None:
            raise RuntimeError("radius requested before calibrate()")
        return conformal_radius(np.array(self._buffer, dtype=float), self._alpha_current)

    @property
    def feedback_log(self) -> list[dict]:
        """One record per observed outcome, in the order processed: the
        issued bounds, the miss, and the state the update left behind."""
        return list(self._feedback_log)

    def calibrate(self, calibration_residuals) -> "AdaptiveConformalInference":
        """Seed the buffer with the latest min(W, n) calibration scores, in
        chronological order, and set alpha_current back to target_alpha."""
        scores = absolute_scores(calibration_residuals)
        self._buffer = deque(scores[-self.window_size :].tolist(), maxlen=self.window_size)
        self._alpha_current = self.target_alpha
        self._feedback_log = []
        return self

    def reset(self) -> None:
        self._buffer = None
        self._alpha_current = self.target_alpha
        self._feedback_log = []

    def predict(self, point_forecast) -> UQResult:
        """Bounds for one issuance, from the current buffer and
        alpha_current. Does not change state."""
        if self._buffer is None:
            raise RuntimeError("predict() called before calibrate()")
        forecasts = point_forecast_array(point_forecast)
        radius = self.radius
        return UQResult(
            lower=forecasts - radius,
            upper=forecasts + radius,
            escalate=escalation_flags(len(forecasts), radius, self.escalate_threshold),
        )

    def quantify(self, point_forecast: np.ndarray, calibration_residuals: np.ndarray) -> UQResult:
        if self._buffer is None:
            self.calibrate(calibration_residuals)
        return self.predict(point_forecast)

    def observe(self, actual, issued_forecast, issued_lower, issued_upper, timestamps=None):
        """Feed back outcomes that have now arrived, oldest first.

        For each outcome, in order: check it against the bounds originally
        issued for it (an outcome equal to a bound is covered), append
        |actual - issued forecast| to the buffer, then apply the ACI
        update. Pass each outcome exactly once. Returns the 0/1 miss
        indicators.
        """
        if self._buffer is None:
            raise RuntimeError("observe() called before calibrate()")

        actual = np.asarray(actual, dtype=float)
        forecast = np.asarray(issued_forecast, dtype=float)
        lower = np.asarray(issued_lower, dtype=float)
        upper = np.asarray(issued_upper, dtype=float)

        if actual.ndim != 1 or not (actual.shape == forecast.shape == lower.shape == upper.shape):
            raise ValueError(
                "actual, issued_forecast, issued_lower and issued_upper must be "
                "one-dimensional and the same length."
            )
        if not (np.isfinite(actual).all() and np.isfinite(forecast).all()):
            raise ValueError("actual and issued_forecast must be finite.")
        if np.isnan(lower).any() or np.isnan(upper).any():
            raise ValueError("issued bounds must not contain missing values.")
        if (lower > upper).any():
            raise ValueError("issued_lower must not exceed issued_upper.")
        if timestamps is None:
            timestamps = [None] * len(actual)
        elif len(timestamps) != len(actual):
            raise ValueError("timestamps must be the same length as actual.")

        misses = np.zeros(len(actual), dtype=int)
        for i in range(len(actual)):
            miss = int(actual[i] < lower[i] or actual[i] > upper[i])
            self._buffer.append(abs(actual[i] - forecast[i]))
            self._alpha_current += self.gamma * (self.target_alpha - miss)
            misses[i] = miss
            self._feedback_log.append(
                {
                    "timestamp": timestamps[i],
                    "issued_lower": lower[i],
                    "issued_upper": upper[i],
                    "miss": miss,
                    "alpha_current": self._alpha_current,
                    "radius": self.radius,
                    "buffer_size": len(self._buffer),
                }
            )
        return misses

    def describe(self) -> dict:
        """Configuration and current state, for run logging."""
        return {
            "uq_method": self.name,
            "target_alpha": self.target_alpha,
            "gamma": self.gamma,
            "window_size": self.window_size,
            "escalate_threshold": self.escalate_threshold,
            "alpha_current": self._alpha_current,
            "buffer_size": 0 if self._buffer is None else len(self._buffer),
            "n_feedback": len(self._feedback_log),
        }
