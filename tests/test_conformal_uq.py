"""Fixed split conformal and the interval metrics (Step 6). AgACI: test_agaci.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    calculate_empirical_coverage,
    calculate_interval_pinball_loss,
    calculate_mean_interval_width,
    calculate_pinball_loss,
    calculate_rolling_coverage,
    calculate_uq_metrics_by_regime,
    calculate_worst_rolling_coverage,
)
from drift_lab.uncertainty import (
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


# --- pinball loss with infinite bounds --------------------------------------


def test_pinball_loss_is_infinite_not_nan_for_infinite_bounds():
    assert calculate_pinball_loss([1.0, 2.0], [-math.inf, 0.0], 0.05) == math.inf
    assert calculate_pinball_loss([1.0, 2.0], [math.inf, 3.0], 0.95) == math.inf
    assert calculate_interval_pinball_loss([1.0], [-math.inf], [math.inf]) == math.inf


def test_pinball_loss_finite_values_are_unchanged():
    # q=0.05, errors +2 and -1: (0.05 * 2 + 0.95 * 1) / 2
    assert calculate_pinball_loss([3.0, 1.0], [1.0, 2.0], 0.05) == pytest.approx(0.525)


# --- coverage and width (T5) ------------------------------------------------


def test_coverage_and_width_accept_infinite_bounds_and_reject_missing():
    assert calculate_empirical_coverage([1e9, 5.0], [-math.inf, 0.0], [math.inf, 1.0]) == 0.5
    assert calculate_mean_interval_width([-math.inf, 0.0], [math.inf, 1.0]) == math.inf
    with pytest.raises(ValueError):
        calculate_empirical_coverage([1.0], [np.nan], [2.0])


def test_coverage_compares_by_position_not_index_label():
    frame = pd.DataFrame({"y": [0.0, 5.0, 0.0, 5.0], "lo": [-1.0] * 4, "hi": [1.0] * 4})
    kept = frame[frame["y"] > 1]  # index labels 1 and 3
    assert calculate_empirical_coverage(kept["y"].to_numpy(), kept["lo"], kept["hi"]) == 0.0


def test_rolling_coverage_matches_hand_computed_windows():
    y = [0.0, 0.0, 5.0, 5.0, 0.0]  # covered: T T F F T
    rolling = calculate_rolling_coverage(y, [-1.0] * 5, [1.0] * 5, window=2)
    assert np.isnan(rolling.iloc[0])
    assert rolling.iloc[1:].tolist() == [1.0, 0.5, 0.0, 0.5]
    assert calculate_worst_rolling_coverage(y, [-1.0] * 5, [1.0] * 5, window=2) == 0.0


def test_worst_rolling_coverage_only_counts_windows_inside_the_regime():
    y = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0]  # covered: T T F F T T
    labels = ["pre-drift", "pre-drift", "drift", "drift", "drift", "post-drift"]
    args = (y, [-1.0] * 6, [1.0] * 6)
    assert calculate_worst_rolling_coverage(*args, 2, labels, "drift") == 0.0
    assert calculate_worst_rolling_coverage(*args, 2, labels, "pre_drift") == 1.0
    # one post-drift row cannot hold a 2-observation window
    assert math.isnan(calculate_worst_rolling_coverage(*args, 2, labels, "post_drift"))
    assert math.isnan(calculate_worst_rolling_coverage(*args, window=48))


def test_uq_metrics_by_regime_pool_within_the_regime():
    y = [0.0, 0.0, 5.0, 5.0, 0.0, 0.0]  # covered: T T F F T T
    labels = ["pre-drift", "pre-drift", "drift", "drift", "drift", "post-drift"]
    metrics = calculate_uq_metrics_by_regime(y, [-1.0] * 6, [1.0] * 6, labels, window=2)
    assert metrics["drift"] == {
        "coverage": pytest.approx(1 / 3),
        "mean_interval_width": 2.0,
        "worst_rolling_coverage": 0.0,
        "n_observations": 3,
    }
    assert metrics["pre_drift"]["coverage"] == 1.0
    assert math.isnan(metrics["post_drift"]["worst_rolling_coverage"])  # one row, no window

