"""AgACI and its BOA endpoint aggregator (Step 6).

Tests marked `requires_opera` are skipped on a machine without R and opera.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drift_lab.uncertainty import (
    AgACI,
    FixedSplitConformal,
    OperaBOA,
    UncertaintyQuantifier,
    UQResult,
    opera_available,
)
from drift_lab.uncertainty.agaci import PAPER_GAMMA_GRID

# 19 scores at alpha = 0.10: rank = ceil(20 * 0.9) = 18 -> the 18th smallest.
RESIDUALS_19 = np.array([-3, 7, 1, -12, 5, 9, -2, 4, 8, -6, 10, 11, -13, 14, 15, -16, 17, 18, -19.0])


# Everything that aggregates more than one candidate runs opera in R.
requires_opera = pytest.mark.skipif(not opera_available(), reason="needs R with the opera package")


# --- BOA: opera in R ---------------------------------------------------------


def _expert_stream(n=240, k=5, seed=3):
    rng = np.random.default_rng(seed)
    y = 8000 + rng.normal(scale=300, size=n)
    experts = 8000 - np.abs(rng.normal(scale=200, size=(n, k))) * np.linspace(0.5, 3, k)
    return experts, y


@requires_opera
def test_opera_first_update_matches_the_hand_computed_weights():
    boa = OperaBOA(2, tau=0.05)
    np.testing.assert_allclose(boa.weights, [0.5, 0.5])
    assert boa.aggregate([0.0, 10.0])[0] == 5.0
    boa.update([[0.0, 10.0]], [2.0])
    # y < prediction: gradient = 1 - 0.05, regret = 0.95 * (5 - [0, 10]) = [4.75, -4.75];
    # eta_inv2 = 2.2 * 4.75^2 for both, so the regularised regrets differ by 9.5.
    expected = 1 / (1 + math.exp(-9.5 / math.sqrt(2.2 * 4.75**2)))
    np.testing.assert_allclose(boa.weights, [expected, 1 - expected])


@requires_opera
def test_opera_block_feedback_leaves_the_same_state_as_one_at_a_time():
    experts, y = _expert_stream(n=97)  # the last block has a single outcome
    blocks, steps = OperaBOA(experts.shape[1], 0.05), OperaBOA(experts.shape[1], 0.05)
    for start in range(0, len(y), 48):
        blocks.update(experts[start : start + 48], y[start : start + 48])
    for row, outcome in zip(experts, y):
        steps.update([row], [outcome])
    np.testing.assert_allclose(blocks.weights, steps.weights, rtol=1e-9, atol=1e-12)
    assert blocks.n_updates == steps.n_updates == 97


def test_opera_boa_with_one_expert_needs_no_r_session():
    boa = OperaBOA(1, tau=0.05)
    boa.update([[3.0], [4.0]], [1.0, 2.0])
    assert boa.weights.tolist() == [1.0]


@pytest.mark.parametrize("kwargs", [{"n_experts": 0, "tau": 0.5}, {"n_experts": 2, "tau": 1.0}])
def test_opera_boa_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        OperaBOA(**kwargs)


@requires_opera
def test_opera_boa_rejects_bad_input():
    boa = OperaBOA(2, tau=0.05)
    with pytest.raises(ValueError):
        boa.update([[0.0, math.inf]], [1.0])
    with pytest.raises(ValueError):
        boa.update([[0.0, 1.0]], [1.0, 2.0])


# --- AgACI ------------------------------------------------------------------


def _agaci(**kwargs):
    params = {"window_size": 19}
    params.update(kwargs)
    return AgACI(**params).calibrate(RESIDUALS_19)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gammas": []},
        {"gammas": [0.01, -0.01]},
        {"gammas": [0.01, math.inf]},
        {"gammas": [0.01, 0.01]},
        {"window_size": 0},
        {"target_alpha": 1.0},
    ],
)
def test_agaci_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        AgACI(**kwargs)


@requires_opera
def test_agaci_is_an_uncertainty_quantifier_with_one_candidate_per_gamma():
    uq = _agaci()
    assert isinstance(uq, UncertaintyQuantifier)
    assert uq.gammas.tolist() == list(PAPER_GAMMA_GRID)
    np.testing.assert_array_equal(uq.alphas, [0.10] * len(PAPER_GAMMA_GRID))


@requires_opera
def test_agaci_first_issuance_equals_fixed_conformal():
    forecasts = np.array([100.0, 200.0])
    issued = _agaci().quantify(forecasts, RESIDUALS_19)
    fixed = FixedSplitConformal().calibrate(RESIDUALS_19).predict(forecasts)
    assert isinstance(issued, UQResult)
    np.testing.assert_allclose(issued.lower, fixed.lower)
    np.testing.assert_allclose(issued.upper, fixed.upper)


@requires_opera
def test_agaci_prediction_does_not_change_state():
    uq = _agaci()
    first = uq.predict([5.0])
    second = uq.predict([5.0])
    assert first.lower[0] == second.lower[0] and first.upper[0] == second.upper[0]
    assert uq.describe()["n_feedback"] == 0
    np.testing.assert_array_equal(uq.buffer, np.abs(RESIDUALS_19))


@requires_opera
def test_agaci_each_candidate_updates_from_its_own_gamma():
    uq = _agaci(gammas=[0.0, 0.01, 0.05])
    issued = uq.predict([0.0])  # every candidate radius is 18
    assert uq.observe([999.0], issued).tolist() == [1]
    np.testing.assert_allclose(uq.alphas, [0.10, 0.10 - 0.01 * 0.9, 0.10 - 0.05 * 0.9])


def test_agaci_outcome_on_a_bound_counts_as_covered():
    uq = _agaci(gammas=[0.01])
    assert uq.observe([18.0], uq.predict([0.0])).tolist() == [0]
    assert uq.alphas[0] == pytest.approx(0.10 + 0.01 * 0.1)


@requires_opera
def test_agaci_appends_each_outcome_to_the_shared_buffer_once():
    uq = _agaci(window_size=4)
    np.testing.assert_array_equal(uq.buffer, [16.0, 17.0, 18.0, 19.0])
    uq.observe([7.0, -3.0], uq.predict([5.0, 5.0]))
    np.testing.assert_array_equal(uq.buffer, [18.0, 19.0, 2.0, 8.0])
    assert uq.describe()["n_feedback"] == 2


@requires_opera
def test_agaci_refuses_to_feed_back_an_issuance_twice():
    uq = _agaci()
    issued = uq.predict([5.0])
    uq.observe([5.0], issued)
    with pytest.raises(ValueError):
        uq.observe([5.0], issued)


@requires_opera
def test_agaci_observe_rejects_bad_feedback():
    uq = _agaci()
    with pytest.raises(ValueError):
        uq.observe([1.0, 2.0], uq.predict([5.0]))
    with pytest.raises(ValueError):
        uq.observe([np.nan], uq.predict([5.0]))


@requires_opera
def test_agaci_block_shares_the_radii_and_weights_available_at_issuance():
    issued = _agaci().predict(np.arange(48.0))
    np.testing.assert_allclose(issued.upper - issued.lower, 36.0)  # 2 * 18 for all 48


@requires_opera
def test_agaci_replaces_infinite_candidates_with_the_finite_threshold():
    uq = _agaci(gammas=[0.0, 0.5])
    uq.observe([999.0], uq.predict([0.0]))  # a miss drives the gamma = 0.5 level below zero
    assert uq.alphas[1] < 0

    issued = uq.predict([10.0])
    assert issued.candidate_radius[1] == math.inf
    assert issued.aggregation_radius[1] == uq.threshold_radius == 999.0
    assert np.isfinite(issued.lower).all() and np.isfinite(issued.upper).all()
    assert issued.lower[0] <= 10.0 <= issued.upper[0]

    # the infinite candidate's own interval covers anything: its level rises
    before = uq.alphas[1]
    uq.observe([1e9], issued)
    assert uq.alphas[1] == pytest.approx(before + 0.5 * 0.10)
    assert uq.describe()["n_replaced"] == [0, 1]


@requires_opera
def test_agaci_bounds_stay_ordered_around_the_forecast_under_shift():
    rng = np.random.default_rng(1)
    uq = AgACI(window_size=96).calibrate(rng.normal(scale=10, size=500))
    for day in range(60):
        forecasts = rng.normal(loc=1000, scale=50, size=48)
        issued = uq.predict(forecasts)
        assert np.isfinite(issued.lower).all() and np.isfinite(issued.upper).all()
        assert (issued.lower <= forecasts).all() and (forecasts <= issued.upper).all()
        scale = 10 if day < 20 else 60  # errors grow sixfold from day 20
        uq.observe(forecasts + rng.normal(scale=scale, size=48), issued)
    assert uq.lower_weights.sum() == pytest.approx(1.0)
    assert uq.upper_weights.sum() == pytest.approx(1.0)


@requires_opera
def test_agaci_instances_do_not_share_state_and_recalibration_resets():
    first, second = _agaci(), _agaci()
    first.observe([999.0], first.predict([0.0]))
    np.testing.assert_array_equal(second.alphas, [0.10] * len(PAPER_GAMMA_GRID))
    first.calibrate(RESIDUALS_19)
    np.testing.assert_array_equal(first.alphas, [0.10] * len(PAPER_GAMMA_GRID))
    assert first.describe()["n_feedback"] == 0


@requires_opera
def test_agaci_escalation_follows_the_issued_width():
    assert _agaci(escalate_threshold=30.0).predict([1.0, 2.0]).escalate.all()  # width 36 > 30
    assert not _agaci(escalate_threshold=40.0).predict([1.0]).escalate.any()
    assert not _agaci().predict([1.0]).escalate.any()  # disabled
