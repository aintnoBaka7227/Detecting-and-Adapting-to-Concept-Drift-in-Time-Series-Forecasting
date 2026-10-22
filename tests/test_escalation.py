"""Escalation metrics (Step 6 / T6, F5): width-threshold review sweep."""

from __future__ import annotations

import math

import numpy as np
import pytest

from drift_lab.evaluation import (
    calculate_coverage_at_review_rate,
    calculate_escalation_curve,
    calculate_escalation_curves_by_regime,
    calculate_escalation_operating_points,
    calculate_nonconformity_score,
    calculate_retained_coverage,
    calculate_review_rate,
    calculate_review_rate_for_coverage,
)
from drift_lab.uncertainty import AgACI, FixedSplitConformal

RESIDUALS_19 = np.array([-3, 7, 1, -12, 5, 9, -2, 4, 8, -6, 10, 11, -13, 14, 15, -16, 17, 18, -19.0])

# Five intervals around 0 with widths 2, 4, 4, 6, 8; the widest two miss.
ACTUAL = np.array([0.5, 1.0, -1.5, 10.0, 10.0])
LOWER = np.array([-1.0, -2.0, -2.0, -3.0, -4.0])
UPPER = np.array([1.0, 2.0, 2.0, 3.0, 4.0])


def _row(curve, threshold):
    return curve[curve["threshold"] == threshold].iloc[0]


# --- single threshold ----------------------------------------------------------


def test_review_rate_counts_strictly_wider_intervals():
    assert calculate_review_rate(LOWER, UPPER, 4) == 0.4
    assert calculate_review_rate(LOWER, UPPER, 8) == 0.0


def test_retained_coverage_is_coverage_among_kept_forecasts():
    assert calculate_retained_coverage(ACTUAL, LOWER, UPPER, 4) == 1.0
    assert calculate_retained_coverage(ACTUAL, LOWER, UPPER, 8) == pytest.approx(3 / 5)
    assert math.isnan(calculate_retained_coverage(ACTUAL, LOWER, UPPER, 1))


# --- the curve -------------------------------------------------------------------


def test_curve_grid_is_every_distinct_width_plus_review_everything():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    assert curve["threshold"].tolist() == [-math.inf, 2, 4, 6, 8]
    assert curve["review_rate"].tolist() == [1.0, 0.8, 0.4, 0.2, 0.0]


def test_curve_counts_and_coverage_by_hand():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    no_review = _row(curve, 8)
    assert (no_review["n_retained"], no_review["n_reviewed"]) == (5, 0)
    assert no_review["retained_coverage"] == pytest.approx(3 / 5)
    assert _row(curve, 4)["retained_coverage"] == 1.0
    assert _row(curve, 6)["retained_coverage"] == pytest.approx(3 / 4)


def test_curve_agrees_with_the_single_threshold_metrics():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    for row in curve.iloc[1:].itertuples():
        assert row.review_rate == calculate_review_rate(LOWER, UPPER, row.threshold)
        assert row.retained_coverage == calculate_retained_coverage(
            ACTUAL, LOWER, UPPER, row.threshold
        )


def test_review_everything_has_undefined_coverage_not_100_percent():
    first = calculate_escalation_curve(ACTUAL, LOWER, UPPER).iloc[0]
    assert first["n_retained"] == 0
    assert math.isnan(first["retained_coverage"])


def test_tied_widths_are_reviewed_together():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    assert 0.6 not in curve["review_rate"].tolist()


def test_float_noise_in_a_constant_width_is_one_operating_point():
    centre = np.array([7000.1, 7100.2, 7200.3])
    lower, upper = centre - 1644.2, centre + 1644.2
    assert len(np.unique(upper - lower)) > 1  # the noise this guards against
    curve = calculate_escalation_curve([7000, 7100, 7200], lower, upper)
    assert curve["review_rate"].tolist() == [1.0, 0.0]


def test_explicit_thresholds_are_used_as_given():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER, thresholds=[3, 100])
    assert curve["threshold"].tolist() == [-math.inf, 3, 100]
    assert curve["review_rate"].tolist() == [1.0, 0.8, 0.0]


def test_curve_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="lower_bound must not exceed"):
        calculate_escalation_curve([0.0], [1.0], [0.0])


# --- agreement with the UQ classes' own escalate flag ----------------------------


def _reviewed_at(curve, threshold):
    return int(curve.loc[curve["threshold"] == threshold, "n_reviewed"].iloc[0])


def test_curve_matches_agaci_escalate_flags_at_the_same_threshold():
    rng = np.random.default_rng(0)
    forecast = 100 + rng.normal(scale=5, size=60)
    actual = forecast + rng.normal(scale=8, size=60)
    lower, upper = np.empty(60), np.empty(60)
    uq = AgACI(gammas=(0.01,), window_size=19)  # one candidate: no R needed
    for t in range(60):
        issued = uq.quantify(forecast[t : t + 1], RESIDUALS_19)
        lower[t], upper[t] = issued.lower[0], issued.upper[0]
        uq.observe(actual[t : t + 1])
    tau = float(np.median(np.round(upper - lower, 6)))

    flagged = AgACI(gammas=(0.01,), window_size=19, escalate_threshold=tau)
    flags = np.empty(60, dtype=bool)
    for t in range(60):
        issued = flagged.quantify(forecast[t : t + 1], RESIDUALS_19)
        np.testing.assert_allclose([issued.lower[0], issued.upper[0]], [lower[t], upper[t]])
        flags[t] = issued.escalate[0]
        flagged.observe(actual[t : t + 1])

    curve = calculate_escalation_curve(actual, lower, upper, thresholds=[tau])
    assert 0 < flags.sum() < 60
    assert _reviewed_at(curve, tau) == flags.sum()


@pytest.mark.parametrize("tau", [10.0, 60.0])
def test_curve_matches_fixed_conformal_escalate_flags(tau):
    forecast = np.array([100.0, 120.0, 90.0])
    issued = FixedSplitConformal(escalate_threshold=tau).quantify(forecast, RESIDUALS_19)
    curve = calculate_escalation_curve(forecast, issued.lower, issued.upper, thresholds=[tau])
    assert _reviewed_at(curve, tau) == issued.escalate.sum()


# --- regimes ---------------------------------------------------------------------


def test_regimes_share_the_full_stream_grid():
    labels = ["pre-drift", "drift", "drift", "post-drift", "post-drift"]
    curves = calculate_escalation_curves_by_regime(ACTUAL, LOWER, UPPER, labels)
    assert curves["regime"].unique().tolist() == ["pre_drift", "drift", "post_drift"]
    grids = curves.groupby("regime", sort=False)["threshold"].apply(list)
    assert all(grid == [-math.inf, 2, 4, 6, 8] for grid in grids)
    drift = curves[curves["regime"] == "drift"]
    assert drift["n_observations"].unique().tolist() == [2]


def test_regime_labels_are_validated():
    with pytest.raises(ValueError, match="same length"):
        calculate_escalation_curves_by_regime(ACTUAL, LOWER, UPPER, ["drift"])
    with pytest.raises(ValueError, match="unsupported regimes"):
        calculate_escalation_curves_by_regime(ACTUAL, LOWER, UPPER, ["stable"] * 5)


# --- T6 operating points -----------------------------------------------------------


def test_review_rate_for_coverage_is_the_smallest_reaching_rate():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    assert calculate_review_rate_for_coverage(curve, 0.75) == 0.2
    assert calculate_review_rate_for_coverage(curve, 0.90) == 0.4
    assert calculate_review_rate_for_coverage(curve, 0.50) == 0.0


def test_target_is_unreachable_when_no_retained_set_reaches_it():
    curve = calculate_escalation_curve([10.0, 10.0], [-1.0, -2.0], [1.0, 2.0])
    assert math.isnan(calculate_review_rate_for_coverage(curve, 0.90))


def test_min_retained_excludes_tiny_retained_sets():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    assert math.isnan(calculate_review_rate_for_coverage(curve, 0.90, min_retained=4))
    with pytest.raises(ValueError):
        calculate_review_rate_for_coverage(curve, 0.90, min_retained=0)


def test_budget_uses_the_largest_rate_not_above_it():
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER)
    assert calculate_coverage_at_review_rate(curve, 0.30) == (pytest.approx(3 / 4), 0.2)
    assert calculate_coverage_at_review_rate(curve, 0.10) == (pytest.approx(3 / 5), 0.0)


def test_operating_points_keys():
    points = calculate_escalation_operating_points(calculate_escalation_curve(ACTUAL, LOWER, UPPER))
    assert points["review_rate_for_coverage_0.90"] == 0.4
    assert points["review_rate_for_coverage_0.95"] == 0.4
    assert points["coverage_at_review_0.20"] == pytest.approx(3 / 4)
    assert points["actual_review_at_0.05"] == 0.0
    assert points["n_operating_points"] == 5


# --- nonconformity score ---------------------------------------------------------


def test_nonconformity_score_uses_only_past_errors():
    y = np.array([10.0, 12.0, 9.0, 15.0])
    pred = np.array([11.0, 10.0, 10.0, 10.0])  # |errors| = 1, 2, 1, 5
    score = calculate_nonconformity_score(y, pred, lookback=1)
    assert math.isnan(score[0])
    np.testing.assert_allclose(score[1:], [1.0, 2.0, 1.0])  # forecast t sees error t-1 only
    np.testing.assert_allclose(calculate_nonconformity_score(y, pred, lookback=2)[2:], [1.5, 1.5])


def test_nonconformity_score_does_not_move_with_the_current_outcome():
    y = np.array([10.0, 12.0, 9.0, 15.0])
    pred = np.array([11.0, 10.0, 10.0, 10.0])
    changed = y.copy()
    changed[3] = 1e6  # the outcome of the scored forecast itself
    np.testing.assert_array_equal(
        calculate_nonconformity_score(y, pred, lookback=2),
        calculate_nonconformity_score(changed, pred, lookback=2),
    )


def test_nonconformity_score_carries_in_history():
    score = calculate_nonconformity_score(
        [10.0, 12.0], [11.0, 10.0], lookback=2, history_true=[5.0, 7.0], history_pred=[2.0, 7.0]
    )
    # history |errors| 3, 0; stream |errors| 1, 2
    np.testing.assert_allclose(score, [1.5, 0.5])


def test_nonconformity_score_rejects_bad_lookback():
    with pytest.raises(ValueError, match="lookback"):
        calculate_nonconformity_score([1.0], [1.0], lookback=0)


def test_curve_sweeps_any_score_against_the_issued_intervals():
    # Score ranks the two misses highest, unlike width.
    score = np.array([1.0, 1.0, 1.0, 9.0, 8.0])
    curve = calculate_escalation_curve(ACTUAL, LOWER, UPPER, score=score)
    assert curve["threshold"].tolist() == [-math.inf, 1.0, 8.0, 9.0]
    assert _row(curve, 1.0)["review_rate"] == 0.4
    assert _row(curve, 1.0)["retained_coverage"] == 1.0  # both misses reviewed


def test_constant_width_intervals_get_a_full_curve_from_a_varying_score():
    centre = np.arange(6, dtype=float)
    lower, upper = centre - 1, centre + 1
    curve = calculate_escalation_curve(centre, lower, upper, score=[1, 2, 3, 4, 5, 6])
    assert curve["review_rate"].nunique() == 7


def test_score_must_be_finite_and_aligned():
    with pytest.raises(ValueError, match="one value per forecast"):
        calculate_escalation_curve(ACTUAL, LOWER, UPPER, score=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        calculate_escalation_curve(ACTUAL, LOWER, UPPER, score=[1.0, np.nan, 1.0, 1.0, 1.0])


def test_regimes_share_the_score_grid():
    labels = ["pre-drift", "drift", "drift", "post-drift", "post-drift"]
    score = np.array([5.0, 1.0, 2.0, 3.0, 4.0])
    curves = calculate_escalation_curves_by_regime(ACTUAL, LOWER, UPPER, labels, score=score)
    grids = curves.groupby("regime", sort=False)["threshold"].apply(list)
    assert all(grid == [-math.inf, 1.0, 2.0, 3.0, 4.0, 5.0] for grid in grids)
