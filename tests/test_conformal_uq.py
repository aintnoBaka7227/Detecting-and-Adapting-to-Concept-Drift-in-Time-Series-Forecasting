"""Fixed split conformal and adaptive conformal inference (Step 6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drift_lab.evaluation import calculate_interval_pinball_loss, calculate_pinball_loss
from drift_lab.uncertainty import (
    AdaptiveConformalInference,
    FixedSplitConformal,
    UncertaintyQuantifier,
)
from drift_lab.uncertainty.conformal_common import conformal_radius, conformal_rank

# 19 scores at alpha = 0.10: rank = ceil(20 * 0.9) = 18 -> the 18th smallest.
RESIDUALS_19 = np.array([-3, 7, 1, -12, 5, 9, -2, 4, 8, -6, 10, 11, -13, 14, 15, -16, 17, 18, -19.0])


# --- shared quantile rule ---------------------------------------------------


def test_conformal_rank_is_not_pushed_up_by_float_noise():
    assert conformal_rank(99, 0.10) == 90
    assert conformal_rank(19, 0.10) == 18


def test_conformal_radius_extreme_levels_are_not_clipped():
    scores = np.array([1.0, 2.0, 3.0])
    assert conformal_radius(scores, 0.0) == math.inf
    assert conformal_radius(scores, -0.2) == math.inf
    assert conformal_radius(scores, 1.0) == 0.0
    assert conformal_radius(scores, 1.3) == 0.0
    # rank = ceil(4 * 0.9) = 4 > 3 scores -> infinite, not the maximum.
    assert conformal_radius(scores, 0.10) == math.inf
    # rank = ceil(4 * 0.5) = 2 -> second smallest.
    assert conformal_radius(scores, 0.50) == 2.0


# --- FixedSplitConformal ----------------------------------------------------


def test_fixed_is_an_uncertainty_quantifier():
    assert isinstance(FixedSplitConformal(), UncertaintyQuantifier)
    assert isinstance(AdaptiveConformalInference(gamma=0.01, window_size=5), UncertaintyQuantifier)


def test_fixed_radius_matches_the_hand_computed_order_statistic():
    uq = FixedSplitConformal(alpha=0.10).calibrate(RESIDUALS_19)
    assert sorted(np.abs(RESIDUALS_19))[18 - 1] == 18.0
    assert uq.radius == 18.0
    assert uq.n_calibration == 19


def test_fixed_bounds_are_symmetric_with_constant_width():
    uq = FixedSplitConformal()
    forecasts = np.array([100.0, 250.0, -40.0])
    first = uq.quantify(forecasts, RESIDUALS_19)
    again = uq.quantify(forecasts, RESIDUALS_19)

    np.testing.assert_allclose(first.lower, forecasts - 18.0)
    np.testing.assert_allclose(first.upper, forecasts + 18.0)
    np.testing.assert_allclose(first.upper - first.lower, 36.0)
    np.testing.assert_array_equal(first.lower, again.lower)
    assert first.escalate.dtype == bool and not first.escalate.any()


def test_fixed_radius_is_set_once_and_later_residuals_are_ignored():
    uq = FixedSplitConformal()
    uq.quantify([1.0], RESIDUALS_19)
    uq.quantify([1.0], RESIDUALS_19 * 100)  # e.g. residuals that include test outcomes
    assert uq.radius == 18.0

    uq.reset()
    assert not uq.is_calibrated
    with pytest.raises(RuntimeError):
        uq.predict([1.0])


def test_fixed_insufficient_calibration_gives_an_infinite_radius():
    # 5 scores: rank = ceil(6 * 0.9) = 6 > 5.
    uq = FixedSplitConformal().calibrate([1.0, 2.0, 3.0, 4.0, 5.0])
    assert uq.radius == math.inf
    assert uq.describe()["radius_is_infinite"]
    result = uq.predict([10.0])
    assert result.lower[0] == -math.inf and result.upper[0] == math.inf


@pytest.mark.parametrize(
    "residuals",
    [[], [[1.0, 2.0]], [1.0, np.nan], [1.0, np.inf]],
)
def test_fixed_rejects_invalid_calibration(residuals):
    with pytest.raises(ValueError):
        FixedSplitConformal().calibrate(residuals)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_fixed_rejects_invalid_alpha(alpha):
    with pytest.raises(ValueError):
        FixedSplitConformal(alpha=alpha)


def test_fixed_rejects_non_finite_forecasts():
    uq = FixedSplitConformal().calibrate(RESIDUALS_19)
    with pytest.raises(ValueError):
        uq.predict([1.0, np.nan])


def test_fixed_escalation_threshold_flags_all_or_none():
    wide = FixedSplitConformal(escalate_threshold=30.0).quantify([1.0, 2.0], RESIDUALS_19)
    narrow = FixedSplitConformal(escalate_threshold=40.0).quantify([1.0, 2.0], RESIDUALS_19)
    assert wide.escalate.all()  # width 36 > 30
    assert not narrow.escalate.any()  # width 36 <= 40


# --- AdaptiveConformalInference ---------------------------------------------


def _aci(**kwargs):
    params = {"gamma": 0.01, "window_size": 19}
    params.update(kwargs)
    return AdaptiveConformalInference(**params).calibrate(RESIDUALS_19)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gamma": 0.0, "window_size": 10},
        {"gamma": -0.01, "window_size": 10},
        {"gamma": math.inf, "window_size": 10},
        {"gamma": 0.01, "window_size": 0},
        {"gamma": 0.01, "window_size": 2.5},
        {"gamma": 0.01, "window_size": 10, "target_alpha": 1.0},
    ],
)
def test_aci_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AdaptiveConformalInference(**kwargs)


def test_aci_seeds_the_buffer_with_the_latest_scores_in_order():
    uq = AdaptiveConformalInference(gamma=0.01, window_size=4).calibrate(RESIDUALS_19)
    np.testing.assert_array_equal(uq.buffer, [16.0, 17.0, 18.0, 19.0])
    assert uq.alpha_current == 0.10


def test_aci_first_radius_equals_fixed_when_the_window_holds_all_scores():
    assert _aci().radius == FixedSplitConformal().calibrate(RESIDUALS_19).radius == 18.0


def test_aci_miss_lowers_alpha_and_cover_raises_it():
    missed = _aci()
    missed.observe([200.0], [100.0], [82.0], [118.0])
    assert missed.alpha_current == pytest.approx(0.10 + 0.01 * (0.10 - 1))  # 0.091

    covered = _aci()
    covered.observe([105.0], [100.0], [82.0], [118.0])
    assert covered.alpha_current == pytest.approx(0.10 + 0.01 * (0.10 - 0))  # 0.101


def test_aci_outcome_on_a_bound_counts_as_covered():
    uq = _aci()
    misses = uq.observe([82.0, 118.0], [100.0, 100.0], [82.0, 82.0], [118.0, 118.0])
    assert misses.tolist() == [0, 0]


def test_aci_prediction_does_not_change_state():
    uq = _aci()
    before = (uq.alpha_current, uq.buffer.tolist())
    first = uq.quantify([100.0, 200.0], RESIDUALS_19)
    second = uq.quantify([100.0, 200.0], RESIDUALS_19 * 5)  # must not re-seed
    assert (uq.alpha_current, uq.buffer.tolist()) == before
    np.testing.assert_array_equal(first.lower, second.lower)
    np.testing.assert_array_equal(first.upper, second.upper)


def test_aci_feedback_uses_the_issued_bounds_not_recomputed_ones():
    uq = _aci()
    issued = uq.predict([100.0])
    # The outcome sits outside the issued interval [82, 118]. Its own error
    # (60) enters the buffer afterwards, but must not rescue the miss.
    misses = uq.observe([160.0], [100.0], issued.lower, issued.upper)
    assert misses.tolist() == [1]
    assert uq.buffer[-1] == 60.0


def test_aci_buffer_rolls_and_keeps_issued_errors():
    uq = AdaptiveConformalInference(gamma=0.01, window_size=3).calibrate([1.0, 2.0, 3.0])
    uq.observe([10.0, 20.0], [6.0, 27.0], [0.0, 0.0], [50.0, 50.0])
    np.testing.assert_array_equal(uq.buffer, [3.0, 4.0, 7.0])


def test_aci_each_outcome_updates_once_and_is_logged():
    uq = _aci()
    uq.observe([105.0, 300.0], [100.0, 100.0], [82.0, 82.0], [118.0, 118.0], timestamps=["t1", "t2"])
    assert uq.alpha_current == pytest.approx(0.10 + 0.01 * 0.10 + 0.01 * (0.10 - 1))
    log = uq.feedback_log
    assert [r["miss"] for r in log] == [0, 1]
    assert [r["timestamp"] for r in log] == ["t1", "t2"]
    assert log[0]["alpha_current"] == pytest.approx(0.101)
    assert log[0]["buffer_size"] == 19 and log[0]["issued_lower"] == 82.0
    assert uq.describe()["n_feedback"] == 2


def test_aci_extreme_levels_follow_the_conventions():
    # Repeated misses with a large gamma drive alpha_current below zero.
    low = AdaptiveConformalInference(gamma=0.5, window_size=19).calibrate(RESIDUALS_19)
    low.observe([999.0], [0.0], [-1.0], [1.0])
    assert low.alpha_current < 0
    assert low.radius == math.inf
    assert np.isinf(low.predict([5.0]).upper).all()

    # Repeated covers drive it above one: zero-radius interval.
    high = AdaptiveConformalInference(gamma=5.0, window_size=19).calibrate(RESIDUALS_19)
    high.observe([0.0, 0.0], [0.0, 0.0], [-1.0, -1.0], [1.0, 1.0])
    assert high.alpha_current >= 1
    result = high.predict([5.0])
    assert result.lower[0] == result.upper[0] == 5.0


def test_aci_rank_overflow_gives_an_infinite_radius():
    uq = AdaptiveConformalInference(gamma=0.01, window_size=5).calibrate(RESIDUALS_19)
    assert uq.radius == math.inf  # rank = ceil(6 * 0.9) = 6 > 5 buffered scores


def test_aci_observe_accepts_infinite_issued_bounds():
    uq = _aci()
    assert uq.observe([1e9], [0.0], [-math.inf], [math.inf]).tolist() == [0]


def test_aci_observe_rejects_bad_feedback():
    uq = _aci()
    with pytest.raises(ValueError):
        uq.observe([1.0, 2.0], [1.0], [0.0], [2.0])
    with pytest.raises(ValueError):
        uq.observe([np.nan], [1.0], [0.0], [2.0])
    with pytest.raises(ValueError):
        uq.observe([1.0], [1.0], [3.0], [2.0])
    with pytest.raises(RuntimeError):
        AdaptiveConformalInference(gamma=0.01, window_size=5).observe([1.0], [1.0], [0.0], [2.0])


def test_aci_reinitialising_resets_all_state():
    uq = _aci()
    uq.observe([300.0], [100.0], [82.0], [118.0])
    uq.calibrate(RESIDUALS_19)
    assert uq.alpha_current == 0.10
    assert uq.feedback_log == []
    np.testing.assert_array_equal(uq.buffer, np.abs(RESIDUALS_19))

    uq.reset()
    assert not uq.is_calibrated


def test_aci_instances_do_not_share_state():
    a, b = _aci(), _aci()
    a.observe([300.0], [100.0], [82.0], [118.0])
    assert b.alpha_current == 0.10
    assert len(b.feedback_log) == 0
    np.testing.assert_array_equal(b.buffer, np.abs(RESIDUALS_19))


def test_aci_escalation_follows_the_current_width():
    uq = _aci(escalate_threshold=30.0)
    assert uq.predict([1.0, 2.0]).escalate.all()  # width 36 > 30
    assert not _aci(escalate_threshold=40.0).predict([1.0]).escalate.any()
    assert not _aci().predict([1.0]).escalate.any()  # disabled


# --- pinball loss with infinite bounds --------------------------------------


def test_pinball_loss_is_infinite_not_nan_for_infinite_bounds():
    assert calculate_pinball_loss([1.0, 2.0], [-math.inf, 0.0], 0.05) == math.inf
    assert calculate_pinball_loss([1.0, 2.0], [math.inf, 3.0], 0.95) == math.inf
    assert calculate_interval_pinball_loss([1.0], [-math.inf], [math.inf]) == math.inf


def test_pinball_loss_finite_values_are_unchanged():
    # q=0.05, errors +2 and -1: (0.05 * 2 + 0.95 * 1) / 2
    assert calculate_pinball_loss([3.0, 1.0], [1.0, 2.0], 0.05) == pytest.approx(0.525)
