"""Chance-matching baseline for AEMO Table T2.

For N detections spread at random over T days, the chance that at least
one lands in event k's match window of w_k days is 1 - (1 - w_k/T)^N, so
the expected matches by chance are:

    expected matches = sum over events k of (1 - (1 - w_k/T)^N)

Each w_k is that event's own match window (evaluation.calculate_match_window),
so long events, which are easier to hit by chance, raise the bar.

Any result not clearly above that line is not a result.
"""

from __future__ import annotations

from collections.abc import Sequence


def expected_matches_closed_form(
    test_days: float,
    n_detections: int,
    window_days: Sequence[float],
) -> float:
    """expected matches = sum_k (1 - (1 - w_k/T)^N), one w_k per event."""
    if test_days <= 0:
        return float("nan")
    return sum(1.0 - (1.0 - w / test_days) ** n_detections for w in window_days)
