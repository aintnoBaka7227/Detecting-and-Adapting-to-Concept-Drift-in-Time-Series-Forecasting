"""AgACI: online aggregation of ACI candidates over a rolling error buffer.

Zaffran et al. (2022), "Adaptive Conformal Predictions for Time Series",
Section 4 / Algorithm 1, on top of the Gibbs & Candes (2021) ACI update.
Three things adapt, and all of them only ever change *future* intervals:

    buffer      the latest `window_size` absolute errors
                |actual - originally issued forecast|, shared by every
                candidate, seeded from the end of the calibration
                residuals and fed once per test outcome;
    alpha_k     one miscoverage level per learning rate gamma_k, moved by

                    alpha_k += gamma_k * (target_alpha - miss_k)

                where miss_k is judged against candidate k's *own*
                originally issued interval. gamma_k = 0 keeps the level at
                target_alpha, but its radius still follows the buffer;
    weights     two Bernstein Online Aggregation states, one over the
                candidates' lower bounds scored at alpha / 2 and one over
                their upper bounds scored at 1 - alpha / 2. They are run
                by the R package opera, as in the paper (opera_boa.py), so
                R and opera must be installed.

At issuance candidate k's radius is the finite-sample conformal quantile
of the buffer at level 1 - alpha_k, and the issued interval is

    lower = sum_k lower_weight_k * (point_forecast - radius_k)
    upper = sum_k upper_weight_k * (point_forecast + radius_k)

The two weight vectors are separate, so the interval can be asymmetric
around the point forecast.

Infinite candidates. A candidate whose level has fallen to alpha_k <= 0, or
whose rank exceeds the buffer, has an infinite radius (never clipped; see
conformal_common.conformal_radius). It cannot be averaged, so for the
aggregation only it is replaced by a finite threshold radius M: the
largest absolute error among the calibration scores and the scores now in
the buffer. M uses only outcomes already observed and is never smaller
than a finite candidate's radius. Every replaced bound stays on its own
side of the point forecast, so the issued bounds cannot cross. The
candidate's feedback still uses its original (infinite) interval.

Lifecycle, one instance per forecaster x region x arm x seed x
configuration (a forecaster retrain does not reset it):

    uq = AgACI()
    issued = uq.quantify(forecasts, calibration_residuals)  # seeds once
    ...the outcomes of those forecasts arrive...
    uq.observe(actuals)                  # feeds back the last quantify call
    issued = uq.quantify(next_forecasts, calibration_residuals)

This is the shared `quantify` / `observe` loop of base.py. `observe` can
also be given an issuance explicitly (`uq.observe(actuals, issued)`), e.g.
after `predict`.

`predict` never changes state; `quantify` only records its issuance as the
one `observe` feeds back. Every forecast in one call
shares the candidate radii and weights available at that issuance. The
project issues one forecast per call (a half-hour) and feeds its outcome
back before the next, so each interval uses every outcome up to the
previous half-hour; passing several forecasts at once delays feedback to
the end of that block instead.
`observe` takes the issuance it is feeding back, processes its outcomes in
order, and refuses to process the same issuance twice. With one candidate
there is nothing to aggregate and R is not needed.

`window_size` defaults to a provisional 7 days and `gammas` to a grid the paper examines; neither is tuned to AEMO.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from drift_lab.uncertainty.base import UncertaintyQuantifier, UQResult
from drift_lab.uncertainty.conformal_common import (
    absolute_scores,
    conformal_radius,
    point_forecast_array,
    validate_escalate_threshold,
    validate_level,
)
from drift_lab.uncertainty.opera_boa import OperaBOA, opera_backend

# 7 days of half-hourly scores. Provisional: not yet selected on validation data.
PROVISIONAL_WINDOW_SIZE = 48 * 7

# Learning rates examined in Zaffran et al. (2022). A published experimental
# grid, not one selected for AEMO.
PAPER_GAMMA_GRID = (0.0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07)


@dataclass
class AgACIIssuance(UQResult):
    """One issuance: the final bounds plus everything feedback needs.

    `candidate_radius` holds each candidate's original radius (may be
    infinite) and `aggregation_radius` the same with infinite entries
    replaced by the threshold; both have one entry per candidate, shared by
    every forecast of the issuance.
    """

    point_forecast: np.ndarray
    candidate_radius: np.ndarray
    aggregation_radius: np.ndarray
    lower_weights: np.ndarray
    upper_weights: np.ndarray
    processed: bool = False


class AgACI(UncertaintyQuantifier):
    """Aggregated ACI with a shared rolling buffer of issued-forecast errors.

    Parameters
    ----------
    gammas : sequence of float, default PAPER_GAMMA_GRID
        ACI learning rates, one candidate each; finite, non-negative and
        distinct.
    window_size : int, default 336 (7 days)
        Buffer length W in scores. The default is provisional.
    target_alpha : float, default 0.10
        Nominal miscoverage. The endpoints are aggregated at
        target_alpha / 2 and 1 - target_alpha / 2.
    escalate_threshold : float | None, default None
        Escalate a forecast when its interval width exceeds this value.
        None disables escalation (all flags False).
    """

    name = "agaci_rolling_calibration"

    def __init__(
        self,
        gammas=PAPER_GAMMA_GRID,
        window_size: int = PROVISIONAL_WINDOW_SIZE,
        target_alpha: float = 0.10,
        escalate_threshold: float | None = None,
    ) -> None:
        gammas = np.asarray(gammas, dtype=float)
        if gammas.ndim != 1 or gammas.size == 0:
            raise ValueError("gammas must be a nonempty one-dimensional sequence.")
        if not np.isfinite(gammas).all() or (gammas < 0).any():
            raise ValueError("gammas must be finite and non-negative.")
        if len(np.unique(gammas)) != len(gammas):
            raise ValueError("gammas must be distinct.")
        if isinstance(window_size, bool) or int(window_size) != window_size or window_size < 1:
            raise ValueError("window_size must be a positive integer.")

        self.gammas = gammas
        self.window_size = int(window_size)
        self.target_alpha = validate_level(target_alpha, "target_alpha")
        self.escalate_threshold = validate_escalate_threshold(escalate_threshold)
        self.reset()

    @property
    def is_calibrated(self) -> bool:
        return self._buffer is not None

    @property
    def alphas(self) -> np.ndarray:
        """Each candidate's current miscoverage level (a copy)."""
        return self._alphas.copy()

    @property
    def buffer(self) -> np.ndarray:
        """The shared buffered scores, oldest first (a copy)."""
        self._require_calibrated("buffer")
        return np.array(self._buffer, dtype=float)

    @property
    def candidate_radius(self) -> np.ndarray:
        """Each candidate's radius at the next issuance; may be infinite."""
        scores = self.buffer
        return np.array([conformal_radius(scores, alpha) for alpha in self._alphas])

    @property
    def threshold_radius(self) -> float:
        """Finite radius M standing in for an infinite candidate."""
        self._require_calibrated("threshold_radius")
        return max(self._calibration_max, max(self._buffer))

    @property
    def lower_weights(self) -> np.ndarray:
        self._require_calibrated("lower_weights")
        return self._lower_aggregator.weights

    @property
    def upper_weights(self) -> np.ndarray:
        self._require_calibrated("upper_weights")
        return self._upper_aggregator.weights

    def calibrate(self, calibration_residuals) -> "AgACI":
        """Seed the buffer with the latest min(W, n) calibration scores, in
        chronological order, and start every level and weight afresh."""
        scores = absolute_scores(calibration_residuals)
        self.reset()
        self._buffer = deque(scores[-self.window_size :].tolist(), maxlen=self.window_size)
        self._calibration_max = float(scores.max())
        self._lower_aggregator = OperaBOA(len(self.gammas), self.target_alpha / 2)
        self._upper_aggregator = OperaBOA(len(self.gammas), 1 - self.target_alpha / 2)
        return self

    def reset(self) -> None:
        self._buffer: deque[float] | None = None
        self._calibration_max = math.nan
        self._alphas = np.full(len(self.gammas), self.target_alpha)
        self._lower_aggregator: OperaBOA | None = None
        self._upper_aggregator: OperaBOA | None = None
        self._n_feedback = 0
        self._n_replaced = np.zeros(len(self.gammas), dtype=int)
        self._last_issuance: AgACIIssuance | None = None

    def predict(self, point_forecast) -> AgACIIssuance:
        """Bounds for one issuance, from the current buffer, levels and
        weights. Does not change state."""
        self._require_calibrated("predict()")
        forecasts = point_forecast_array(point_forecast)

        candidate_radius = self.candidate_radius
        aggregation_radius = np.where(
            np.isfinite(candidate_radius), candidate_radius, self.threshold_radius
        )
        lower_weights, upper_weights = self.lower_weights, self.upper_weights
        lower = forecasts - float(aggregation_radius @ lower_weights)
        upper = forecasts + float(aggregation_radius @ upper_weights)
        if (lower > upper).any():
            raise RuntimeError("aggregated bounds crossed; this should be impossible.")

        if self.escalate_threshold is None:
            escalate = np.zeros(len(forecasts), dtype=bool)
        else:
            escalate = (upper - lower) > self.escalate_threshold
        return AgACIIssuance(
            lower=lower,
            upper=upper,
            escalate=escalate,
            point_forecast=forecasts,
            candidate_radius=candidate_radius,
            aggregation_radius=aggregation_radius,
            lower_weights=lower_weights,
            upper_weights=upper_weights,
        )

    def quantify(self, point_forecast: np.ndarray, calibration_residuals: np.ndarray) -> AgACIIssuance:
        if self._buffer is None:
            self.calibrate(calibration_residuals)
        self._last_issuance = self.predict(point_forecast)
        return self._last_issuance

    def observe(self, actual, issued: AgACIIssuance | None = None) -> np.ndarray:
        """Feed back the outcomes of one issuance, oldest first.

        Without `issued`, feeds back the most recent `quantify` call (the
        shared interface loop); pass `issued` to feed back a `predict`.

        Every candidate's level is updated from that candidate's own
        original interval (an outcome on a bound is covered, and an
        infinite interval always covers); the lower and upper aggregators
        are updated from the replacement endpoints; and each
        |actual - issued forecast| is appended to the shared buffer once.
        Returns the 0/1 misses of the issued interval.
        """
        self._require_calibrated("observe()")
        if issued is None:
            issued = self._last_issuance
            if issued is None:
                raise ValueError("observe() needs a prior quantify() call or an issuance.")
        if issued.processed:
            raise ValueError("this issuance has already been fed back.")

        actual = np.asarray(actual, dtype=float)
        forecasts = issued.point_forecast
        if actual.shape != forecasts.shape:
            raise ValueError("actual must have one outcome per issued forecast.")
        if not np.isfinite(actual).all():
            raise ValueError("actual must be finite.")
        if issued.candidate_radius.shape != self.gammas.shape:
            raise ValueError("issuance does not match this instance's candidates.")

        # One row per outcome, one column per candidate.
        outcomes, centres = actual[:, None], forecasts[:, None]
        miss = (outcomes < centres - issued.candidate_radius) | (
            outcomes > centres + issued.candidate_radius
        )
        self._alphas += self.gammas * (self.target_alpha * len(actual) - miss.sum(axis=0))
        self._lower_aggregator.update(centres - issued.aggregation_radius, actual)
        self._upper_aggregator.update(centres + issued.aggregation_radius, actual)
        self._buffer.extend(np.abs(actual - forecasts).tolist())

        issued.processed = True
        self._n_feedback += len(actual)
        self._n_replaced += len(actual) * np.isinf(issued.candidate_radius)
        return ((actual < issued.lower) | (actual > issued.upper)).astype(int)

    def describe(self) -> dict:
        """Configuration and current state, for run logging.
        `n_replaced` counts, per candidate, the fed-back forecasts whose
        infinite endpoints were replaced by the threshold."""
        calibrated = self._buffer is not None
        return {
            "uq_method": self.name,
            "target_alpha": self.target_alpha,
            "gammas": self.gammas.tolist(),
            "window_size": self.window_size,
            "escalate_threshold": self.escalate_threshold,
            "aggregation": "BOA, gradient trick, pinball; "
            + (opera_backend() if len(self.gammas) > 1 else "single candidate, no aggregation"),
            "threshold_rule": "max(calibration scores, buffered scores)",
            "alphas": self._alphas.tolist(),
            "lower_weights": self.lower_weights.tolist() if calibrated else None,
            "upper_weights": self.upper_weights.tolist() if calibrated else None,
            "buffer_size": len(self._buffer) if calibrated else 0,
            "n_feedback": self._n_feedback,
            "n_replaced": self._n_replaced.tolist(),
        }

    def _require_calibrated(self, what: str) -> None:
        if self._buffer is None:
            raise RuntimeError(f"{what} requested before calibrate()")
