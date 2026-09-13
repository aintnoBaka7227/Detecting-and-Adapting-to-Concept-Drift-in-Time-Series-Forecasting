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
    # half-open [drift_start, drift_end): a "day" event occupies the full 24h
    # of its start date, so drift_end is midnight of the *next* day.
    assert row["drift_end"] == pd.Timestamp("2020-03-02")
    assert row["post_drift_end"] == pd.Timestamp("2020-03-09")


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
    # half-open [drift_start, drift_end): occupies the full 24h of end_date too.
    assert row["drift_end"] == pd.Timestamp("2020-05-18")
    assert row["post_drift_end"] == pd.Timestamp("2020-05-25")


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


@pytest.mark.xfail(
    reason="assign_regime's post_drift window is unbounded (accepted gap, "
    "team decision) -- a detection this far past the only event still gets "
    "labelled post_drift/unassigned instead of falling back to pre_drift",
)
def test_assign_regime_detection_outside_all_windows_is_pre_drift():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-06-01"], windows)

    assert len(labels) == 1
    assert labels.iloc[0]["regime"] == "pre_drift"
    assert labels.iloc[0]["event_id"] == "unassigned"


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
    # 2020-03-05 is simultaneously in E1's post-drift tail and E2's
    # pre-drift lead-up (pre_drift_days=7 reaches back to 2020-03-03).
    # pre_drift outranks post_drift globally, so E2's pre_drift wins --
    # regardless of E1 being the temporally nearer event.
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-03-10", "2020-03-10", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-03-05"], windows)

    assert len(labels) == 1
    assert labels.iloc[0]["event_id"] == "E2"
    assert labels.iloc[0]["regime"] == "pre_drift"


def test_assign_regime_each_timestamp_at_most_once():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-03-10", "2020-03-10", "day", "SA1"),
        ]
    )
    windows = build_event_windows(events, "SA1")
    labels = assign_regime(["2020-03-05", "2020-03-05"], windows)

    assert len(labels) == 1


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
        ["2020-03-03"], events, "SA1", point_window=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Match"
    assert result.iloc[0]["event_id"] == "E1"
    assert result.iloc[0]["delay_days"] == pytest.approx(2.0)


def test_match_unmatch_point_event_unmatched_too_late():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch(
        ["2020-03-10"], events, "SA1", point_window=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Unmatch"
    assert result.iloc[0]["event_id"] is None
    assert pd.isna(result.iloc[0]["delay_days"])


def test_match_unmatch_pre_event_detection_is_unmatched():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = match_unmatch(
        ["2020-02-25"], events, "SA1", point_window=pd.Timedelta(days=7)
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
        point_window=pd.Timedelta(days=7),
    )

    assert result[result["label"] == "Match"]["event_id"].tolist() == [
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
        point_window=pd.Timedelta(days=7),
    )

    matched = result[result["label"] == "Match"]
    assert len(matched) == 1
    assert matched.iloc[0]["event_id"] == "E1"
    assert len(result[result["label"] == "Unmatch"]) == 1


def test_match_unmatch_range_event_matched_within_window():
    events = _aemo_events(
        [
            ("E1", "2020-04-01", "2020-05-17", "date_range", "NEM"),
        ]
    )
    result = match_unmatch(
        ["2020-05-20"], events, "NEM", interval_grace=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Match"
    assert result.iloc[0]["event_id"] == "E1"


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
        ["2020-03-06"], events, "SA1", point_window=pd.Timedelta(days=3)
    )
    assert result3.iloc[0]["label"] == "Unmatch"

    # 7-day tolerance: same detection is matched
    result7 = match_unmatch(
        ["2020-03-06"], events, "SA1", point_window=pd.Timedelta(days=7)
    )
    assert result7.iloc[0]["label"] == "Match"


def _tiered_events(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "event_id",
            "start_date",
            "end_date",
            "date_precision",
            "region",
            "tier",
        ],
    )


def test_match_unmatch_tier1_preferred_over_tier2():
    # E2 (Tier 2) starts earlier and would win under pure chronology; the
    # tier ordering must give the detection to the later Tier 1 event E1.
    events = _tiered_events(
        [
            ("E2", "2020-03-01", "2020-03-01", "day", "SA1", 2),
            ("E1", "2020-03-05", "2020-03-05", "day", "SA1", 1),
        ]
    )
    result = match_unmatch(
        ["2020-03-06"], events, "SA1", point_window=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Match"
    assert result.iloc[0]["event_id"] == "E1"
    assert result.iloc[0]["tier"] == 1


def test_match_unmatch_tier2_event_still_counts_as_match():
    events = _tiered_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1", 2),
        ]
    )
    result = match_unmatch(
        ["2020-03-03"], events, "SA1", point_window=pd.Timedelta(days=7)
    )

    assert len(result) == 1
    assert result.iloc[0]["label"] == "Match"
    assert result.iloc[0]["event_id"] == "E1"
    assert result.iloc[0]["tier"] == 2


def test_match_unmatch_rejects_invalid_tier():
    events = _tiered_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1", 3),
        ]
    )
    with pytest.raises(ValueError):
        match_unmatch(["2020-03-03"], events, "SA1")


def test_match_unmatch_output_is_tier_ranked():
    # Tier-2 event comes first chronologically, but the output must list
    # the Tier-1 match first, then the Tier-2 match, then Unmatch.
    events = _tiered_events(
        [
            ("E2", "2020-03-01", "2020-03-01", "day", "SA1", 2),
            ("E1", "2020-03-05", "2020-03-05", "day", "SA1", 1),
        ]
    )
    result = match_unmatch(
        ["2020-03-03", "2020-03-06", "2021-01-01"],
        events,
        "SA1",
        point_window=pd.Timedelta(days=7),
    )

    assert result.iloc[0]["tier"] == 1
    assert result.iloc[0]["event_id"] == "E1"
    assert result.iloc[1]["tier"] == 2
    assert result.iloc[1]["event_id"] == "E2"
    assert result.iloc[2]["label"] == "Unmatch"
    assert pd.isna(result.iloc[2]["tier"])


def test_match_unmatch_missing_column_raises():
    bad = pd.DataFrame({"event_id": ["E1"], "start_date": ["2020-01-01"]})
    with pytest.raises(ValueError):
        match_unmatch(["2020-01-02"], bad, "SA1")


def test_match_unmatch_rejects_end_before_start():
    events = _aemo_events(
        [
            ("E1", "2020-03-10", "2020-03-01", "date_range", "SA1"),
        ]
    )
    with pytest.raises(ValueError):
        match_unmatch(["2020-03-05"], events, "SA1")


# --- evaluate_aemo_detections ---------------------------------------------


def test_evaluate_aemo_detections_returns_all_keys():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
        ]
    )
    result = evaluate_aemo_detections(["2020-03-03"], events, "SA1")

    assert "match_results" in result
    assert "regime_results" in result
    assert "event_metrics" in result
    m = result["event_metrics"]
    assert m["n_total_detections"] == 1
    assert m["n_matched_t1"] == 0
    assert m["n_matched_t2"] == 1
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
    m = result["event_metrics"]

    assert m["n_matched_t1"] == 0
    assert m["n_matched_t2"] == 0
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
    match_results = match_unmatch(
        ["2020-03-03", "2020-06-05", "2020-09-01"],
        events,
        "SA1",
        point_window=pd.Timedelta(days=7),
    )
    m = calculate_event_metrics(match_results, events, "SA1")

    assert m["n_total_detections"] == 3
    assert m["n_matched_t1"] == 0
    assert m["n_matched_t2"] == 2
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
    match_results = match_unmatch([], events, "SA1")
    m = calculate_event_metrics(match_results, events, "SA1")

    assert m["n_total_detections"] == 0
    assert m["n_matched_t1"] == 0
    assert m["n_matched_t2"] == 0
    assert m["n_unmatched_events"] == 1
    assert np.isnan(m["precision"])
    assert m["event_recall"] == pytest.approx(0.0)


def test_calculate_event_metrics_counts_by_tier():
    events = _aemo_events(
        [
            ("E1", "2020-03-01", "2020-03-01", "day", "SA1"),
            ("E2", "2020-06-01", "2020-06-01", "day", "SA1"),
            ("E3", "2020-09-01", "2020-09-01", "day", "SA1"),
        ]
    )
    events["tier"] = [1, 2, 2]
    match_results = match_unmatch(
        ["2020-03-03", "2020-06-05"],
        events,
        "SA1",
        point_window=pd.Timedelta(days=7),
    )
    m = calculate_event_metrics(match_results, events, "SA1")

    assert m["n_matched_t1"] == 1
    assert m["n_matched_t2"] == 1
    assert m["n_matched_detections"] == 2
    assert m["n_unmatched_events"] == 1
    assert m["event_recall"] == pytest.approx(2 / 3)


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
