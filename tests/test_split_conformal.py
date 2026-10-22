from __future__ import annotations

import math

import numpy as np
import pytest

from drift_lab.uncertainty.base import UncertaintyQuantifier
from drift_lab.uncertainty.split_conformal import SplitConformal


def test_is_an_uncertainty_quantifier():
    assert isinstance(SplitConformal(), UncertaintyQuantifier)


def test_radius_is_the_rank_th_smallest_absolute_residual():
    # n = 9, coverage 0.8 -> rank ceil(10 * 0.8) = 8 -> 8th smallest of 1..9
    residuals = np.array([-9, 8, -7, 6, -5, 4, -3, 2, -1], dtype=float)
    assert SplitConformal(coverage=0.8).radius(residuals) == 8.0


def test_float_noise_does_not_push_the_rank_up():
    # n = 99, coverage 0.9 -> (99 + 1) * 0.9 = 90 (not 91)
    residuals = np.arange(1, 100, dtype=float)
    assert SplitConformal(coverage=0.9).radius(residuals) == 90.0


def test_too_few_residuals_gives_an_infinite_radius():
    assert math.isinf(SplitConformal(coverage=0.9).radius(np.array([1.0, 2.0, 3.0])))


def test_interval_is_forecast_plus_minus_radius():
    uq = SplitConformal(coverage=0.8)
    out = uq.quantify(np.array([100.0, 200.0]), np.arange(1.0, 10.0))
    np.testing.assert_allclose(out.lower, [92.0, 192.0])
    np.testing.assert_allclose(out.upper, [108.0, 208.0])
    assert out.escalate.dtype == bool and not out.escalate.any()


def test_escalate_flags_everything_when_width_exceeds_threshold():
    residuals = np.arange(1.0, 10.0)  # radius 8, width 16
    flagged = SplitConformal(coverage=0.8, escalate_threshold=15.0).quantify(np.zeros(3), residuals)
    kept = SplitConformal(coverage=0.8, escalate_threshold=16.0).quantify(np.zeros(3), residuals)
    assert flagged.escalate.all()
    assert not kept.escalate.any()


def test_stateless_between_calls():
    uq = SplitConformal(coverage=0.8)
    small = uq.quantify(np.zeros(1), np.arange(1.0, 10.0))
    large = uq.quantify(np.zeros(1), 10 * np.arange(1.0, 10.0))
    assert large.upper[0] == 10 * small.upper[0]


def test_coverage_is_close_to_target_on_exchangeable_data():
    rng = np.random.default_rng(0)
    calibration = rng.normal(0, 1, 2000)
    truth = rng.normal(0, 1, 20000)  # forecast 0, so residual = truth
    out = SplitConformal(coverage=0.9).quantify(np.zeros_like(truth), calibration)
    covered = ((truth >= out.lower) & (truth <= out.upper)).mean()
    assert covered == pytest.approx(0.9, abs=0.01)


@pytest.mark.parametrize("coverage", [0.0, 1.0, -0.1, 1.5])
def test_bad_coverage_is_rejected(coverage):
    with pytest.raises(ValueError):
        SplitConformal(coverage=coverage)


def test_bad_inputs_are_rejected():
    uq = SplitConformal()
    with pytest.raises(ValueError):
        uq.quantify(np.zeros(2), np.array([]))
    with pytest.raises(ValueError):
        uq.quantify(np.zeros(2), np.array([1.0, np.nan]))
    with pytest.raises(ValueError):
        uq.quantify(np.array([1.0, np.inf]), np.ones(10))
    with pytest.raises(ValueError):
        uq.quantify(np.zeros((2, 2)), np.ones(10))
    with pytest.raises(ValueError):
        SplitConformal(escalate_threshold=-1.0)
