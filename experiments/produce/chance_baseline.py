"""Chance-matching baseline for AEMO Table T2.

For N detections, K events and a matching window of w days over T days,
the expected matches by chance are:

    expected matches = K * (1 - (1 - w/T)^N)

Any result not clearly above that line is not a result.
"""

from __future__ import annotations

from drift_lab.evaluation.evaluation import POINT_WINDOW


def expected_matches_closed_form(
    k_events: int,
    test_days: float,
    n_detections: int,
    window_days: float = POINT_WINDOW.days,
) -> float:
    """expected matches = K * (1 - (1 - w/T)^N)."""
    if test_days <= 0:
        return float("nan")
    return k_events * (1.0 - (1.0 - window_days / test_days) ** n_detections)
