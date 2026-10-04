"""ADWIN-specific tests."""

import numpy as np
import pandas as pd

from drift_lab.detection.adwin import ADWINDetector


def test_adwin_name_is_stable():
    detector = ADWINDetector()

    assert detector.name == "adwin"


def test_adwin_default_config():
    detector = ADWINDetector()

    assert detector.delta == 0.002
    assert detector.max_window_size == 2048
    assert detector.cooldown == 1024

    public_config = {
        key: value for key, value in vars(detector).items() if not key.startswith("_")
    }

    assert public_config == {
        "delta": 0.002,
        "max_window_size": 2048,
        "cooldown": 1024,
    }


def test_adwin_detect_returns_positional_indices():
    detector = ADWINDetector(max_window_size=512)

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
    assert detected
    assert detected[0] >= 500
    assert detected[0] < 700


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

    assert detected == detector.detect(stream.to_numpy())


def test_adwin_is_deterministic_for_each_stream():
    stream = np.concatenate([np.zeros(500), np.full(500, 5.0)])
    detector = ADWINDetector()

    assert detector.detect(stream) == detector.detect(stream)
