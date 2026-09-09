"""KSWIN-specific tests."""

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.base import detect_with_river
from drift_lab.detection.kswin import KSWINDetector


def test_kswin_name_is_stable():
    detector = KSWINDetector()

    assert detector.name == "kswin"


def test_kswin_default_config():
    detector = KSWINDetector()

    assert detector.alpha == 0.005
    assert detector.window_size == 100
    assert detector.stat_size == 30
    assert detector.seed == 42

    public_config = {
        key: value for key, value in vars(detector).items() if not key.startswith("_")
    }

    assert public_config == {
        "alpha": 0.005,
        "window_size": 100,
        "stat_size": 30,
        "seed": 42,
    }


def test_kswin_detect_returns_positional_indices():
    detector = KSWINDetector()

    stream = np.concatenate(
        [
            np.zeros(500),
            np.full(500, 5.0),
        ]
    )

    detected = detector.detect(stream)

    assert isinstance(detected, list)
    assert all(isinstance(index, int) for index in detected)
    assert all(0 <= index < len(stream) for index in detected)


def test_kswin_accepts_pandas_series():
    detector = KSWINDetector()

    stream = pd.Series(
        np.concatenate(
            [
                np.zeros(500),
                np.full(500, 5.0),
            ]
        )
    )

    detected = detector.detect(stream)

    assert isinstance(detected, list)


def test_kswin_wrapper_matches_river():
    stream = np.concatenate(
        [
            np.zeros(500),
            np.full(500, 5.0),
        ]
    )

    project_detected = KSWINDetector(
        alpha=0.005,
        window_size=100,
        stat_size=30,
        seed=42,
    ).detect(stream)

    river_detected = detect_with_river(
        drift.KSWIN(
            alpha=0.005,
            window_size=100,
            stat_size=30,
            seed=42,
        ),
        stream,
    )

    assert project_detected == river_detected
