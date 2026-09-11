"""Tests for the shared metric module, mirroring evaluation/."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    assign_regime,
    build_event_windows,
    calculate_event_metrics,
    calculate_mae,
    calculate_rolling_mae,
    evaluate_aemo_detections,
    evaluate_detections,
    match_detections_to_events,
    match_unmatch,
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


# --- build_event_windows --------------------------------------------------


def _aemo_events(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "start_date",
            "end_date",
            "date_precision",
            "region",
        ],
    )


def test_build_event_windows_point_event():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1", pre_drift_days=7, post_drift_days=7)

    assert len(windows) == 1
    row = windows.iloc[0]
    assert row["pre_drift_start"] == pd.Timestamp("2020-02-23")
    assert row["drift_start"] == pd.Timestamp("2020-03-01")
    assert row["drift_end"] == pd.Timestamp("2020-03-01")
    assert row["post_drift_end"] == pd.Timestamp("2020-03-08")


def test_build_event_windows_range_event():
    events = _aemo_events(
        [
            ("E1", "2020-04-01", "2020-05-17", "date_range", "NEM"),
        ]
    )
    windows = build_event_windows(events, "NEM", pre_drift_days=7, post_drift_days=7)

    assert len(windows) == 1
    row = windows.iloc[0]
    assert row["pre_drift_start"] == pd.Timestamp("2020-03-25")
    assert row["drift_start"] == pd.Timestamp("2020-04-01")
    assert row["drift_end"] == pd.Timestamp("2020-05-17")
    assert row["post_drift_end"] == pd.Timestamp("2020-05-24")


def test_build_event_windows_filters_by_region():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-06-01", "2020-06-01", "day", "NSW1"),
        ]
    )
    windows = build_event_windows(events, "SA1")

    assert len(windows) == 1
    assert windows.iloc[0]["event_id"] == "E1"


def test_build_event_windows_sorted_by_start():
    events = _aemo_events(
        [
            ("E2", "2020-06-01", "2020-06-01", "day", "SA1"),
            ("E1", "2020-01-01", "2020-01-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")

    assert list(windows["event_id"]) == ["E1", "E2"]


# --- assign_regime --------------------------------------------------------


def test_assign_regime_detection_during_drift():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-03-01"], windows)

    assert len(labels) == 1
    assert labels.iloc[0]["regime"] == "drift"
    assert labels.iloc[0]["event_id"] == "E1"


def test_assign_regime_detection_in_pre_drift():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-02-25"], windows)

    assert len(labels) == 1
    assert labels.iloc[0]["regime"] == "pre_drift"


def test_assign_regime_detection_in_post_drift():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-03-05"], windows)

    assert len(labels) == 1
    assert labels.iloc[0]["regime"] == "post_drift"


def test_assign_regime_detection_outside_all_windows():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-06-01"], windows)

    assert len(labels) == 0


def test_assign_regime_no_detections():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime([], windows)

    assert len(labels) == 0


def test_assign_regime_overlapping_events():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-03-10", "2020-03-10", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-03-05"], windows)

    assert len(labels) == 2
    assert set(labels["event_id"]) == {"E1", "E2"}
    assert set(labels["regime"]) == {"post_drift", "pre_drift"}


def test_assign_regime_range_event():
    events = _aemo_events(
        [
            ("E1", "2020-04-01", "2020-05-17", "date_range", "NEM"),
        ]
    )
    windows = build_event_windows(events, "NEM")
    labels = assign_regime(["2020-04-15"], windows)

    assert len(labels) == 1
    assert labels.iloc[0]["regime"] == "drift"


# --- match_unmatch --------------------------------------------------------


def test_match_unmatch_point_event_matched():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch(
        ["2020-03-03"], events, "SA1", tolerance=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Match"
    assert result.iloc[0]["matched_event_id"] == "E1"
    assert result.iloc[0]["delay_days"] == pytest.approx(2.0)


def test_match_unmatch_point_event_unmatched_too_late():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch(
        ["2020-03-10"], events, "SA1", tolerance=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Unmatch"
    assert result.iloc[0]["matched_event_id"] is None
    assert pd.isna(result.iloc[0]["delay_days"])


def test_match_unmatch_pre_event_detection_is_unmatched():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch(
        ["2020-02-25"], events, "SA1", tolerance=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Unmatch"


def test_match_unmatch_one_to_one_chronological():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-03-05", "2020-03-05", "day", "SA1"),
        ]
    )
    # Mar 3 is inside E1's window [Mar 1, Mar 8] but before E2 starts;
    # Mar 7 is inside both windows, but E2 can only take the first unused
    # one (Mar 3 is already taken by E1).
    result = match_unmatch(
        ["2020-03-03", "2020-03-07"],
        events,
        "SA1",
        tolerance=pd.Timedelta(days=7),
    )

    assert result[result["label"] == "Match"]["matched_event_id"].tolist() == [
        "E1",
        "E2",
    ]


def test_match_unmatch_one_detection_per_event():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch(
        ["2020-03-02", "2020-03-03"],
        events,
        "SA1",
        tolerance=pd.Timedelta(days=7),
    )

    matched = result[result["label"] == "Match"]
    assert len(matched) == 1
    assert matched.iloc[0]["matched_event_id"] == "E1"
    assert len(result[result["label"] == "Unmatch"]) == 1


def test_match_unmatch_range_event_matched_within_window():
    events = _aemo_events(
        [
            ("E1", "2020-04-01", "2020-05-17", "date_range", "NEM"),
        ]
    )
    result = match_unmatch(
        ["2020-05-20"], events, "NEM", tolerance=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Match"
    assert result.iloc[0]["matched_event_id"] == "E1"


def test_match_unmatch_no_detections():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch([], events, "SA1")

    assert len(result) == 0


def test_match_unmatch_different_tolerance():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    # 3-day tolerance: detection 5 days later is unmatched
    result3 = match_unmatch(
        ["2020-03-06"], events, "SA1", tolerance=pd.Timedelta(days=3)
    )
    assert result3.iloc[0]["label"] == "Unmatch"

    # 7-day tolerance: same detection is matched
    result7 = match_unmatch(
        ["2020-03-06"], events, "SA1", tolerance=pd.Timedelta(days=7)
    )
    assert result7.iloc[0]["label"] == "Match"


# --- evaluate_aemo_detections ---------------------------------------------


def test_evaluate_aemo_detections_returns_all_keys():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = evaluate_aemo_detections(["2020-03-03"], events, "SA1")

    assert "match_results" in result
    assert "regime_labels" in result
    assert "metrics" in result
    m = result["metrics"]
    assert m["n_total_detections"] == 1
    assert m["n_matched_events"] == 1
    assert m["n_unmatched_events"] == 0
    assert m["n_matched_detections"] == 1
    assert m["n_unmatched_detections"] == 0
    assert m["precision"] == pytest.approx(1.0)
    assert m["event_recall"] == pytest.approx(1.0)


def test_evaluate_aemo_detections_no_match():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = evaluate_aemo_detections(["2020-06-01"], events, "SA1")
    m = result["metrics"]

    assert m["n_matched_events"] == 0
    assert m["n_unmatched_events"] == 1
    assert m["n_unmatched_detections"] == 1
    assert m["precision"] == pytest.approx(0.0)
    assert m["event_recall"] == pytest.approx(0.0)


# --- calculate_event_metrics ----------------------------------------------


def test_calculate_event_metrics_basic():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-06-01", "2020-06-01", "day", "SA1"),
        ]
    )
    m = calculate_event_metrics(
        ["2020-03-03", "2020-06-05", "2020-09-01"],
        events,
        "SA1",
        tolerance=pd.Timedelta(days=7),
    )

    assert m["n_total_detections"] == 3
    assert m["n_matched_events"] == 2
    assert m["n_unmatched_events"] == 0
    assert m["n_matched_detections"] == 2
    assert m["n_unmatched_detections"] == 1
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["event_recall"] == pytest.approx(1.0)


def test_calculate_event_metrics_no_detections():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    m = calculate_event_metrics([], events, "SA1")

    assert m["n_total_detections"] == 0
    assert m["n_matched_events"] == 0
    assert m["n_unmatched_events"] == 1
    assert np.isnan(m["precision"])
    assert m["event_recall"] == pytest.approx(0.0)


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
