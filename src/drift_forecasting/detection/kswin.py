import numpy as np
from river import drift

from drift_forecasting.detection.base import DriftDetector


class KSWINDetector(DriftDetector):
    """KSWIN drift detector wrapper."""

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = 42,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

    def detect(self, stream: np.ndarray) -> list[int]:
        detector = drift.KSWIN(
            alpha=self.alpha,
            window_size=self.window_size,
            stat_size=self.stat_size,
            seed=self.seed,
        )

        changepoints: list[int] = []

        for index, value in enumerate(stream):
            detector.update(float(value))

            if detector.drift_detected:
                changepoints.append(index)

        return changepoints