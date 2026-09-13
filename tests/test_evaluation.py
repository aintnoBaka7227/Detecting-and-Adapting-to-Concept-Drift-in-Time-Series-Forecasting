"""Tests for the shared metric module, mirroring evaluation/."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    calculate_mae,
    calculate_rolling_mae,
    evaluate_detections,
    match_detections_to_events,
)

# --- forecast metrics -------------------------------------------------------

def test_mae_basic():
    assert calculate_mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(2 / 3)


def test_mae_rejects_length_mismatch_and_nan():
    with pytest.raises(ValueError):
        calculate_mae([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        calculate_mae([1.0, np.nan], [1.0, 2.0])


def test_rolling_mae_warmup_then_constant():
    y_true = np.full(400, 100.0)
    y_pred = np.full(400, 95.0)  # constant absolute error of 5
    curve = calculate_rolling_mae(y_true, y_pred)  # default window 336

    assert curve.iloc[:335].isna().all()
    assert curve.iloc[335:].to_numpy() == pytest.approx(5.0)


def test_rolling_mae_matches_manual_rolling():
    rng = np.random.default_rng(0)
    y_true = rng.normal(1000, 50, 1000)
    y_pred = y_true + rng.normal(0, 10, 1000)
    manual = pd.Series(np.abs(y_true - y_pred)).rolling(window=336).mean()
    np.testing.assert_allclose(
        calculate_rolling_mae(y_true, y_pred).to_numpy(), manual.to_numpy()
    )


# --- detection metrics ----------------------------------------------------

def test_sudden_match_delay_and_false_alarm():
    result = evaluate_detections(
        detected_changepoints=[1010, 1800],
        true_changepoints=[1000],
        n_observations=2000,
        drift_type="sudden",
        tolerance=100,
    )
    assert result["detection_delay"] == pytest.approx(10.0)
    assert result["missed_detections"] == 0
    assert result["false_alarms_per_10000"] == pytest.approx(1 / 2000 * 10000)


def test_sudden_missed_when_outside_tolerance():
    result = evaluate_detections([1500], [1000], 2000, "sudden", tolerance=100)
    assert result["missed_detections"] == 1
    assert np.isnan(result["detection_delay"])
    assert result["false_alarms_per_10000"] == pytest.approx(1 / 2000 * 10000)


def test_no_drift_makes_every_detection_a_false_alarm():
    result = evaluate_detections([300, 900], [], 2000, "none")
    assert result["missed_detections"] == 0
    assert np.isnan(result["detection_delay"])
    assert result["false_alarms_per_10000"] == pytest.approx(2 / 2000 * 10000)


def test_gradual_matched_from_start_within_window():
    # truth = [drift_start, drift_end]; a detection inside the ramp matches it.
    result = evaluate_detections([1300], [1000, 2000], 5000, "gradual", tolerance=336)
    assert result["missed_detections"] == 0
    assert result["detection_delay"] == pytest.approx(300.0)  # from drift_start


def test_recurring_matches_each_event_once():
    result = evaluate_detections(
        [1100, 4100, 4200], [1000, 4000], 6000, "recurring", tolerance=500
    )
    assert result["missed_detections"] == 0
    assert result["detection_delay"] == pytest.approx((100 + 100) / 2)
    assert result["false_alarms_per_10000"] == pytest.approx(1 / 6000 * 10000)


def test_validation_rejects_out_of_range_and_wrong_count():
    with pytest.raises(ValueError):
        evaluate_detections([50], [9999], 2000, "sudden")
    with pytest.raises(ValueError):
        evaluate_detections([50], [100, 200], 2000, "sudden")  # sudden needs 1


# --- real-data (documented-event) detection ------------------------------
# Tolerances are passed explicitly: the module defaults are an evaluation
# design choice the team is still tuning, so these test the matching logic,
# not whatever DOCUMENTED_POINT_TOLERANCE / _PERIOD_GRACE currently are.

POINT_TOL = pd.Timedelta(days=14)
PERIOD_GRACE = pd.Timedelta(days=60)


def _events(rows):
    return pd.DataFrame(
        rows, columns=["event_id", "start_date", "end_date", "date_precision"]
    )


def _match(detected, events, **kw):
    kw.setdefault("point_tolerance", POINT_TOL)
    kw.setdefault("period_grace", PERIOD_GRACE)
    return match_detections_to_events(detected, events, **kw)


def test_point_event_matched_within_tolerance_with_delay():
    events = _events([("E1", "2020-03-01", "2020-03-01", "day")])
    result = _match(["2020-03-13"], events)  # 12 days later, tolerance 14

    assert result["n_matched"] == 1
    assert result["matched"][0]["event_id"] == "E1"
    assert result["matched"][0]["delay_days"] == pytest.approx(12.0)
    assert result["mean_delay_days"] == pytest.approx(12.0)
    assert result["n_unmatched_events"] == 0
    assert result["n_unmatched_detections"] == 0
    assert result["precision"] == pytest.approx(1.0)


def test_point_event_missed_when_detection_too_late():
    events = _events([("E1", "2020-03-01", "2020-03-01", "day")])
    result = _match(["2020-04-01"], events)  # 31 days > tolerance 14

    assert result["n_matched"] == 0
    assert result["unmatched_events"] == ["E1"]
    assert result["n_unmatched_detections"] == 1
    assert np.isnan(result["mean_delay_days"])
    assert result["precision"] == pytest.approx(0.0)


def test_detection_before_event_does_not_match():
    events = _events([("E1", "2020-03-01", "2020-03-01", "day")])
    result = _match(["2020-02-25"], events)

    assert result["n_matched"] == 0
    assert result["n_unmatched_events"] == 1
    assert result["n_unmatched_detections"] == 1


def test_period_event_matched_mid_interval_delay_from_start():
    # A months-long trend: detection lands 106 days after the start, inside
    # the interval, so it matches regardless of grace.
    events = _events([("SOLAR", "2020-12-01", "2021-06-30", "date_range")])
    result = _match(["2021-03-17"], events)

    assert result["n_matched"] == 1
    assert result["matched"][0]["delay_days"] == pytest.approx(106.0)


def test_period_event_matched_within_grace_after_end():
    events = _events([("E1", "2020-04-01", "2020-05-17", "date_range")])
    result = _match(["2020-06-30"], events)  # 44 days past end, grace 60

    assert result["n_matched"] == 1


def test_each_detection_and_event_used_at_most_once():
    events = _events(
        [
            ("A", "2020-01-01", "2020-01-01", "day"),
            ("B", "2020-01-05", "2020-01-05", "day"),
        ]
    )
    # One detection in A's window, one shared by both, one spare.
    result = _match(["2020-01-03", "2020-01-08", "2020-06-01"], events)

    assert result["n_matched"] == 2
    assert {m["event_id"] for m in result["matched"]} == {"A", "B"}
    assert [str(ts.date()) for ts in result["unmatched_detections"]] == ["2020-06-01"]


def test_no_detections_leaves_every_event_unmatched():
    events = _events(
        [
            ("A", "2020-01-01", "2020-01-01", "day"),
            ("B", "2021-01-01", "2021-01-31", "month"),
        ]
    )
    result = match_detections_to_events([], events)

    assert result["n_matched"] == 0
    assert result["n_unmatched_events"] == 2
    assert result["n_unmatched_detections"] == 0
    assert np.isnan(result["precision"])


def test_missing_event_column_raises():
    bad = pd.DataFrame({"event_id": ["E1"], "start_date": ["2020-01-01"]})
    with pytest.raises(ValueError):
        match_detections_to_events(["2020-01-02"], bad)


# --- guard ----------------------------------------------------------------

def test_no_metric_code_outside_evaluation_module():
    repo = Path(__file__).resolve().parents[1]
    roots = [repo / "src" / "drift_lab", repo / "experiments"]
    allowed = {"drift_lab/evaluation/evaluation.py"}
    offenders = []
    for root in roots:
        for path in root.rglob("*.py"):
            rel = path.relative_to(
                repo / "src" if root.name == "drift_lab" else repo
            ).as_posix()
            if rel in allowed:
                continue
            text = path.read_text()
            if ".rolling(" in text or ".ewm(" in text:
                offenders.append(rel)
    assert not offenders, f"metric code outside evaluation.py: {offenders}"
