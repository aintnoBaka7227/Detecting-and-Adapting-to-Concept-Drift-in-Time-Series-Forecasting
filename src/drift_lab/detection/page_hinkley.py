"""Page-Hinkley drift detector (river) behind the DriftDetector interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.base import DriftDetector, detect_with_river


class PageHinkleyDetector(DriftDetector):
    name = "page_hinkley"

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
    ) -> None:
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        return detect_with_river(
            drift.PageHinkley(
                min_instances=self.min_instances,
                delta=self.delta,
                threshold=self.threshold,
            ),
            stream,
        )
