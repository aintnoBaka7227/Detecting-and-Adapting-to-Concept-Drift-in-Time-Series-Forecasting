import numpy as np
from river import drift

from drift_forecasting.detection.base import DriftDetector


class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley detector for gradual concept drift."""

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        mode: str = "both",
    ):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode

    def detect(self, stream: np.ndarray) -> list[int]:
        detector = drift.PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
            mode=self.mode,
        )

        changepoints: list[int] = []

        for index, value in enumerate(stream):
            detector.update(float(value))

            if detector.drift_detected:
                changepoints.append(index)

        return changepoints