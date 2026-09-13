"""ADWIN-specific tests."""

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.base import detect_with_river


def test_adwin_name_is_stable():
    detector = ADWINDetector()

    assert detector.name == "adwin"


def test_adwin_default_config():
    detector = ADWINDetector()

    assert detector.delta == 0.002

    public_config = {
        key: value for key, value in vars(detector).items() if not key.startswith("_")
    }

    assert public_config == {"delta": 0.002}


def test_adwin_detect_returns_positional_indices():
    detector = ADWINDetector()

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


def test_adwin_accepts_pandas_series():
    detector = ADWINDetector()

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


def test_adwin_wrapper_matches_river():
    stream = np.concatenate(
        [
            np.zeros(500),
            np.full(500, 5.0),
        ]
    )

    project_detected = ADWINDetector(delta=0.002).detect(stream)

    river_detected = detect_with_river(
        drift.ADWIN(delta=0.002),
        stream,
    )

    assert project_detected == river_detected
