"""Tests for the shared metric module, mirroring evaluation/."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    calculate_mae,
    calculate_rolling_mae,
    evaluate_detections,
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
