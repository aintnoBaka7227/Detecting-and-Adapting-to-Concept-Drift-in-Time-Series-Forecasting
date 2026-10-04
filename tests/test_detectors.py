"""Behavior tests for the three native project drift detectors."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series

CASES = [
    (ADWINDetector(), "adwin"),
    (KSWINDetector(), "kswin"),
    (PageHinkleyDetector(), "page_hinkley"),
]


@pytest.mark.parametrize("detector,name", CASES)
def test_detector_exposes_name_and_positional_indices(detector, name):
    y, _ = make_series("sudden", n=6000, noise=1.0, seed=1)
    assert detector.name == name
    detected = detector.detect(y)
    assert all(0 <= index < len(y) for index in detected)
    assert detected == sorted(set(detected))


@pytest.mark.parametrize("detector,name", CASES)
def test_detects_the_sudden_changepoint(detector, name):
    y, changepoints = make_series("sudden", n=20000, noise=1.0, seed=2)
    result = evaluate_detections(
        detector.detect(y),
        changepoints,
        n_observations=len(y),
        drift_type="sudden",
        tolerance=500,
    )
    assert result["missed_detections"] == 0
    assert np.isfinite(result["detection_delay"])


def test_detect_accepts_series_and_array_alike():
    y, _ = make_series("sudden", n=3000, noise=1.0, seed=1)
    detector = ADWINDetector()
    assert detector.detect(pd.Series(y)) == detector.detect(y)


def test_no_drift_series_produces_few_false_alarms():
    y, changepoints = make_series("none", n=20000, noise=1.0, seed=3)
    assert changepoints == []
    result = evaluate_detections(
        ADWINDetector().detect(y),
        changepoints,
        n_observations=len(y),
        drift_type="none",
    )
    assert result["false_alarms_per_10000"] < 5.0
