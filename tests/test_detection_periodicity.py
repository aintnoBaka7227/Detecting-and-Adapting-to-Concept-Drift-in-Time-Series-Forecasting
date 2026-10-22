"""Detection periodicity metrics (validation of detector alarm timing)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    calculate_alarm_gaps,
    calculate_gap_cv,
    calculate_gap_share,
    calculate_phase_concentration,
    simulate_random_alarm_days,
)

MONDAY = pd.Timestamp("2020-03-02")


def test_gaps_are_sorted_differences():
    np.testing.assert_allclose(calculate_alarm_gaps([10, 0, 4]), [4, 6])
    assert len(calculate_alarm_gaps([5])) == 0


def test_evenly_spaced_alarms_have_zero_gap_cv():
    assert calculate_gap_cv(np.arange(0, 140, 14)) == 0.0


def test_random_alarms_have_gap_cv_near_one():
    days = np.sort(np.random.default_rng(1).uniform(0, 100_000, 5_000))
    assert calculate_gap_cv(days) == pytest.approx(1.0, abs=0.05)


def test_gap_cv_needs_two_gaps():
    assert math.isnan(calculate_gap_cv([0, 5]))


def test_gap_share_counts_half_open_window():
    assert calculate_gap_share([0, 14, 28, 50], 14, 16) == pytest.approx(2 / 3)


def test_alarms_on_one_weekday_are_concentrated():
    mondays = [MONDAY + pd.Timedelta(weeks=k) for k in range(20)]
    r, p = calculate_phase_concentration(mondays, 7, MONDAY)
    assert r == pytest.approx(1.0)
    assert p < 1e-6


def test_alarms_spread_over_the_week_are_not_concentrated():
    days = [MONDAY + pd.Timedelta(days=k) for k in range(70)]  # every weekday, 10 times
    r, p = calculate_phase_concentration(days, 7, MONDAY)
    assert r == pytest.approx(0.0, abs=1e-9)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_phase_concentration_needs_three_alarms():
    assert all(math.isnan(v) for v in calculate_phase_concentration([MONDAY, MONDAY], 7, MONDAY))


def test_random_baseline_respects_span_and_min_gap():
    sims = simulate_random_alarm_days(10, 400, min_gap_days=14, n_sim=500, seed=3)
    assert sims.shape == (500, 10)
    assert sims.min() >= 0 and sims.max() <= 400
    assert np.diff(sims, axis=1).min() >= 14 - 1e-9


def test_random_baseline_rejects_impossible_spacing():
    with pytest.raises(ValueError):
        simulate_random_alarm_days(10, 100, min_gap_days=14)
