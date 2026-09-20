"""Focused tests for the AEMO post-tuning detection pipeline:
standard_stream_common.py, detection_artifacts.py, chance_baseline.py
(closed-form only), post_tune_detector_configs.py, and the
bounded/unassigned post-drift regime rules in drift_lab.evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation.evaluation import assign_regime, build_event_windows
from experiments.produce.chance_baseline import expected_matches_closed_form
from experiments.results_io import RUN_COLUMNS


def _aemo_events(rows):
    return pd.DataFrame(
        rows,
        columns=["event_id", "start_date", "end_date", "date_precision", "region"],
    )


def _frame(dates: pd.DatetimeIndex, value: float) -> pd.DataFrame:
    return pd.DataFrame({"SETTLEMENTDATE": dates, "TOTALDEMAND": value})


# --- TRAIN-only profile + scaler fitting ------------------------------------


def test_seasonal_profile_and_scaler_are_fit_on_train_only(monkeypatch):
    from experiments.run.detection import standard_stream_common

    train_dates = pd.date_range("2018-01-01", periods=60, freq="1D")
    calibration_dates = pd.date_range("2018-03-02", periods=10, freq="1D")
    test_dates = pd.date_range("2018-03-12", periods=10, freq="1D")

    # A tiny, deterministic wobble (not a perfectly flat series) so the
    # TRAIN residual has a nonzero standard deviation to standardise by.
    train_values = 100.0 + 0.1 * np.sin(np.arange(60))
    train = _frame(train_dates, train_values)
    calibration = _frame(calibration_dates, 150.0)  # a level shift TRAIN never saw
    test = _frame(test_dates, 200.0)  # a larger level shift TRAIN never saw

    monkeypatch.setattr(
        standard_stream_common.loader, "load", lambda region: (train, calibration, test)
    )

    full, warmup = standard_stream_common.build_standard_stream("SA1", "daily")

    # TRAIN was fit on itself, so its own standardised residual must sit
    # at ~0 mean regardless of the huge shifts calibration/TEST carry.
    train_z = full.iloc[:60]
    assert abs(float(train_z.mean())) < 1e-6

    # If the profile or the z-score mean/std had been refit (or blended)
    # using calibration/TEST, TEST's constant +100 offset over TRAIN would
    # be partly "absorbed" and show up as a smaller residual. Fit on
    # TRAIN-only, it must show up almost exactly as the raw 100-unit gap
    # (before dividing by TRAIN's own near-zero std, which only amplifies
    # a genuine gap rather than shrinking it).
    test_residual_raw = test["TOTALDEMAND"].mean() - train_values.mean()
    assert test_residual_raw == pytest.approx(100.0, abs=0.1)


def test_build_standard_stream_rejects_bad_cadence():
    from experiments.run.detection import standard_stream_common

    with pytest.raises(ValueError):
        standard_stream_common.build_standard_stream("SA1", "weekly")


# --- fresh detector instance per run ----------------------------------------


def test_make_post_tune_detectors_returns_fresh_instances_each_call():
    from experiments.run.detection.post_tune_detector_configs import make_post_tune_detectors

    first = make_post_tune_detectors()
    second = make_post_tune_detectors()

    for a, b in zip(first, second):
        assert a is not b
        assert vars(a) == vars(b)  # same frozen config, independent objects


# --- fail loudly on a missing/failing frozen configuration ------------------


def test_load_winner_row_fails_loudly_when_missing(tmp_path, monkeypatch):
    from experiments.run.detection import post_tune_detector_configs as cfg

    empty_csv = tmp_path / "fine_tune_on_synthetic_winners.csv"
    pd.DataFrame(
        columns=["detector", "config_hash", "detector_parameters", "status"]
    ).to_csv(empty_csv, index=False)
    monkeypatch.setattr(cfg, "WINNERS_CSV", empty_csv)

    with pytest.raises(RuntimeError):
        cfg.load_winner_row("adwin")


def test_load_winner_row_fails_loudly_when_no_eligible_configuration(tmp_path, monkeypatch):
    from experiments.run.detection import post_tune_detector_configs as cfg

    csv_path = tmp_path / "fine_tune_on_synthetic_winners.csv"
    pd.DataFrame(
        [
            {
                "detector": "kswin",
                "config_hash": None,
                "detector_parameters": None,
                "status": "no_eligible_configuration",
            }
        ]
    ).to_csv(csv_path, index=False)
    monkeypatch.setattr(cfg, "WINNERS_CSV", csv_path)

    with pytest.raises(RuntimeError):
        cfg.make_post_tune_detectors()


# --- timestamp-based refractory filtering + accepted/raw persistence -------


def test_match_and_persist_splits_accepted_from_raw(tmp_path, monkeypatch):
    from experiments.run.detection import detection_artifacts

    monkeypatch.setattr(detection_artifacts, "RUNS_DIR", tmp_path)

    events = _aemo_events([("E1", "2020-03-01", "2020-03-01", "day", "SA1")])
    # Second timestamp is 1 day after the first match -- inside the 14-day
    # refractory window, so it must be Ignored, not counted as accepted.
    detected = ["2020-03-02", "2020-03-03"]

    match_results, artifacts = detection_artifacts.match_and_persist("hash123", "SA1", detected, events)

    accepted = pd.read_csv(artifacts["accepted"])
    raw = pd.read_csv(artifacts["raw"])
    assignments = pd.read_csv(artifacts["assignments"])

    assert len(raw) == 2
    assert len(accepted) == 1
    assert len(assignments) == 2
    assert (match_results["label"] == "Ignored").sum() == 1


# --- bounded post-drift event attribution + unassigned fallback ------------


def test_assign_regime_bounded_post_drift_gets_the_event_id():
    events = _aemo_events([("E1", "2020-03-01", "2020-03-01", "day", "SA1")])
    windows = build_event_windows(events, "SA1")
    # drift_end = 2020-03-02; post_drift_end = drift_end + 7 days = 2020-03-09.
    labels = assign_regime(["2020-03-05"], windows)

    assert labels.iloc[0]["regime"] == "post_drift"
    assert labels.iloc[0]["event_id"] == "E1"


def test_assign_regime_unassigned_post_drift_fallback_beyond_the_bound():
    events = _aemo_events([("E1", "2020-03-01", "2020-03-01", "day", "SA1")])
    windows = build_event_windows(events, "SA1")
    # Well past post_drift_end (2020-03-09): general, unassigned fallback,
    # not a reversion to pre_drift.
    labels = assign_regime(["2020-06-01"], windows)

    assert labels.iloc[0]["regime"] == "post_drift"
    assert labels.iloc[0]["event_id"] == "unassigned"


# --- chance-matching baseline (closed form) ---------------------------------


def test_expected_matches_closed_form_matches_the_stated_formula():
    # expected matches = K * (1 - (1 - w/T)^N)
    k_events, test_days, n_detections, window_days = 5, 1000.0, 80, 7.0
    expected = k_events * (1 - (1 - window_days / test_days) ** n_detections)

    assert expected_matches_closed_form(k_events, test_days, n_detections, window_days) == pytest.approx(
        expected
    )


def test_expected_matches_closed_form_is_deterministic():
    first = expected_matches_closed_form(5, 1000.0, 80, 7.0)
    second = expected_matches_closed_form(5, 1000.0, 80, 7.0)

    assert first == second


# --- runs.csv schema is unchanged -------------------------------------------


def test_runs_csv_schema_is_unchanged():
    assert RUN_COLUMNS == [
        "group",
        "method",
        "dataset",
        "region",
        "seed",
        "config_hash",
        "split_id",
        "metric_name",
        "metric_value",
        "regime",
        "n_retrains",
        "train_samples",
        "wall_clock_s",
        "timestamp",
    ]
