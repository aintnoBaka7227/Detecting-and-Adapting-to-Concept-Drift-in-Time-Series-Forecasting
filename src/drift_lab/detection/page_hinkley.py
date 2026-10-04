"""Native two-sided Page-Hinkley detector behind the project interface."""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_lab.detection.base import DriftDetector, stream_values


class PageHinkleyDetector(DriftDetector):
    name = "page_hinkley"

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
    ) -> None:
        if min_instances < 1:
            raise ValueError("min_instances must be positive")
        if delta < 0.0:
            raise ValueError("delta must be non-negative")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        values = stream_values(stream)
        changepoints: list[int] = []
        count = 0
        mean = 0.0
        cumulative_up = 0.0
        minimum_up = 0.0
        cumulative_down = 0.0
        maximum_down = 0.0

        for index, value in enumerate(values):
            count += 1
            mean += (float(value) - mean) / count
            cumulative_up += float(value) - mean - self.delta
            minimum_up = min(minimum_up, cumulative_up)
            cumulative_down += float(value) - mean + self.delta
            maximum_down = max(maximum_down, cumulative_down)

            if count < self.min_instances:
                continue

            if (
                cumulative_up - minimum_up > self.threshold
                or maximum_down - cumulative_down > self.threshold
            ):
                changepoints.append(index)
                count = 0
                mean = 0.0
                cumulative_up = 0.0
                minimum_up = 0.0
                cumulative_down = 0.0
                maximum_down = 0.0

        return changepoints
