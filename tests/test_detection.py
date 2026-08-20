import numpy as np

from drift_forecasting.detection.page_hinkley import PageHinkleyDetector


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