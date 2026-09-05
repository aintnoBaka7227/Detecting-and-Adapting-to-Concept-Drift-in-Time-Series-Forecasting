"""The three river-backed drift detectors, mirroring detection/."""

import numpy as np
import pandas as pd
import pytest
from river import drift

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series

CASES = [
    (ADWINDetector(), lambda: drift.ADWIN(delta=0.002), "adwin"),
    (
        KSWINDetector(),
        lambda: drift.KSWIN(alpha=0.005, window_size=200, stat_size=50, seed=42),
        "kswin",
    ),
    (
        PageHinkleyDetector(),
        lambda: drift.PageHinkley(min_instances=30, delta=0.005, threshold=50),
        "page_hinkley",
    ),
]


def _raw_river(factory, y) -> list[int]:
    detector = factory()
    flagged = []
    for i, x in enumerate(y):
        detector.update(float(x))
        if detector.drift_detected:
            flagged.append(i)
    return flagged


@pytest.mark.parametrize("detector,factory,name", CASES)
def test_wrapper_matches_raw_river_and_exposes_name(detector, factory, name):
    y, _ = make_series("sudden", n=6000, noise=1.0, seed=1)
    assert detector.name == name
    assert detector.detect(y) == _raw_river(factory, y)


@pytest.mark.parametrize("detector,factory,name", CASES)
def test_detects_the_sudden_changepoint(detector, factory, name):
    y, changepoints = make_series("sudden", n=20000, noise=1.0, seed=2)
    result = evaluate_detections(
        detector.detect(y), changepoints, n_observations=len(y),
        drift_type="sudden", tolerance=500,
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
        ADWINDetector().detect(y), changepoints, n_observations=len(y), drift_type="none"
    )
    assert result["false_alarms_per_10000"] < 5.0
