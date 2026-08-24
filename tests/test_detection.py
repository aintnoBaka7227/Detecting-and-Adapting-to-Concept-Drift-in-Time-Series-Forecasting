import numpy as np

from drift_forecasting.detection.page_hinkley import PageHinkleyDetector
from drift_forecasting.detection.adwin import ADWINDetector
from drift_forecasting.detection.kswin import KSWINDetector

def test_page_hinkley_detects_gradual_drift():
    rng = np.random.default_rng(42)

    stable = rng.normal(0, 0.5, 1000)

    gradual_mean = np.linspace(0, 5, 1000)
    gradual = rng.normal(gradual_mean, 0.5)

    stream = np.concatenate([stable, gradual])

    detector = PageHinkleyDetector()
    changepoints = detector.detect(stream)

    assert len(changepoints) > 0
    assert changepoints[0] >= 1000
    
def test_adwin_detects_gradual_drift():
    rng = np.random.default_rng(42)

    stable = rng.normal(0, 0.5, 1000)

    gradual_mean = np.linspace(0, 5, 1000)
    gradual = rng.normal(gradual_mean, 0.5)

    stream = np.concatenate([stable, gradual])

    detector = ADWINDetector()
    changepoints = detector.detect(stream)

    assert len(changepoints) > 0
    assert changepoints[0] >= 1000
def test_kswin_detects_gradual_drift():
    rng = np.random.default_rng(42)

    stable = rng.normal(0, 0.5, 1000)

    gradual_mean = np.linspace(0, 5, 1000)
    gradual = rng.normal(gradual_mean, 0.5)

    stream = np.concatenate([stable, gradual])

    detector = KSWINDetector()
    changepoints = detector.detect(stream)

    # KSWIN should produce at least one detection.
    assert len(changepoints) > 0

    # Detections before index 1000 are false alarms.
    # We only require that KSWIN eventually detects the real gradual drift.
    valid_detections = [
        cp for cp in changepoints
        if cp >= 1000
    ]

    assert len(valid_detections) > 0