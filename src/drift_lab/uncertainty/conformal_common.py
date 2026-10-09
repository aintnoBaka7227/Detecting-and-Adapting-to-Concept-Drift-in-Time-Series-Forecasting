"""Shared pieces of the two conformal uncertainty methods.

Nothing here holds state: `fixed_split_conformal.py` and
`adaptive_conformal.py` each own their calibration scores and radius.
"""

from __future__ import annotations

import math

import numpy as np


def absolute_scores(residuals, name: str = "calibration_residuals") -> np.ndarray:
    """Conformal scores |actual - issued forecast|, in the order supplied.

    Raises ValueError unless `residuals` is nonempty, one-dimensional and
    finite.
    """
    scores = np.asarray(residuals, dtype=float)
    if scores.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if scores.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(scores).all():
        raise ValueError(f"{name} must be finite.")
    return np.abs(scores)


def validate_level(alpha: float, name: str = "alpha") -> float:
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError(f"{name} must be strictly between 0 and 1.")
    return alpha


def validate_escalate_threshold(threshold: float | None) -> float | None:
    if threshold is None:
        return None
    threshold = float(threshold)
    if math.isnan(threshold) or threshold < 0:
        raise ValueError("escalate_threshold must be a non-negative number or None.")
    return threshold


def conformal_rank(n_scores: int, alpha: float) -> int:
    """Finite-sample conformal rank ceil((n + 1) * (1 - alpha)).

    The product is rounded to 9 decimals first so that binary floating
    point noise (e.g. 90.00000000000001) cannot push the rank up by one.
    """
    return math.ceil(round((n_scores + 1) * (1 - alpha), 9))


def conformal_radius(scores: np.ndarray, alpha: float) -> float:
    """The rank-th smallest score at miscoverage level `alpha`.

    Conventions (never clipped):
        alpha <= 0                       -> infinite radius
        alpha >= 1                       -> zero radius
        rank exceeds the number of scores -> infinite radius
    """
    if alpha <= 0:
        return math.inf
    if alpha >= 1:
        return 0.0
    rank = conformal_rank(len(scores), alpha)
    if rank > len(scores):
        return math.inf
    return float(np.partition(scores, rank - 1)[rank - 1])


def point_forecast_array(point_forecast) -> np.ndarray:
    """Point forecasts as a finite one-dimensional float array."""
    forecasts = np.asarray(point_forecast, dtype=float)
    if forecasts.ndim != 1:
        raise ValueError("point_forecast must be one-dimensional.")
    if not np.isfinite(forecasts).all():
        raise ValueError("point_forecast must be finite.")
    return forecasts


def escalation_flags(n_forecasts: int, radius: float, threshold: float | None) -> np.ndarray:
    """Escalate when the interval width (2 * radius) exceeds `threshold`.

    `threshold=None` disables the rule: every flag is False. That is a
    placeholder, not an escalation policy.
    """
    if threshold is None:
        return np.zeros(n_forecasts, dtype=bool)
    return np.full(n_forecasts, 2 * radius > threshold, dtype=bool)
