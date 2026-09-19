"""Timestamp-level chance-matching baseline for AEMO Table T2 (req. 10).

Simulates the same number of accepted detections as the real detector,
spread across the exact TEST period with the same 14-day refractory rule
enforced -- by construction, consecutive draws are always >= the
refractory period apart, so evaluation.match_unmatch's own refractory
pass never has anything left to suppress -- then evaluated with the same
shared `evaluation.match_unmatch`. No second matching or refractory
implementation lives here.

Also reports the closed-form approximation used as a sanity check on the
simulation, not a replacement for it:

    expected matches = K * (1 - (1 - w/T)^N)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_lab.evaluation.evaluation import POINT_WINDOW, REFRACTORY_PERIOD, match_unmatch

# 20 fixed, reproducible seeds -- not chosen post hoc.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1, 21))


def _random_accepted_timestamps(
    n: int,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    refractory_period: pd.Timedelta,
    rng: np.random.Generator,
) -> pd.DatetimeIndex:
    """n timestamps in [test_start, test_end], consecutive gaps always
    >= refractory_period, via the standard order-statistics-with-minimum-
    spacing reparametrisation (sort n uniform draws over the remaining
    slack, then add back the mandatory minimum gap at each position)."""
    total_days = (test_end - test_start) / pd.Timedelta(days=1)
    min_gap_days = refractory_period / pd.Timedelta(days=1)
    slack = total_days - (n - 1) * min_gap_days

    if slack < 0:
        raise ValueError(
            f"Cannot place {n} detections >= {min_gap_days:.0f} days apart "
            f"inside a {total_days:.0f}-day window."
        )

    positions = np.sort(rng.uniform(0.0, slack, size=n)) if n > 1 else rng.uniform(0.0, slack, size=n)
    offsets = positions + np.arange(n) * min_gap_days
    return pd.DatetimeIndex([test_start + pd.Timedelta(days=float(o)) for o in offsets])


def chance_matching_baseline(
    n_detections: int,
    events: pd.DataFrame,
    region: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    refractory_period: pd.Timedelta = REFRACTORY_PERIOD,
) -> dict:
    """Mean Tier 1 matches (+ a 95% interval) across `seeds` simulations
    of `n_detections` random accepted detections."""
    if n_detections <= 0:
        return {
            "n_detections": n_detections,
            "n_simulations": len(seeds),
            "mean_tier1_matches": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }

    tier1_matches = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        random_timestamps = _random_accepted_timestamps(
            n_detections, test_start, test_end, refractory_period, rng
        )
        match_results = match_unmatch(random_timestamps, events, region)
        matched = match_results[match_results["label"] == "Match"]
        tier1_matches.append(int((matched["tier"] == 1).sum()))

    values = np.array(tier1_matches, dtype=float)
    ci_low, ci_high = np.percentile(values, [2.5, 97.5])

    return {
        "n_detections": n_detections,
        "n_simulations": len(seeds),
        "mean_tier1_matches": float(values.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
    }


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
