import numpy as np
from river import drift

from drift_forecasting.detection.base import DriftDetector


class ADWINDetector(DriftDetector):
    """ADWIN drift detector wrapper."""

    def __init__(self, delta: float = 0.002):
        self.delta = delta

    def detect(self, stream: np.ndarray) -> list[int]:
        detector = drift.ADWIN(delta=self.delta)

        changepoints: list[int] = []

        for index, value in enumerate(stream):
            detector.update(float(value))

            if detector.drift_detected:
                changepoints.append(index)

        return changepoints