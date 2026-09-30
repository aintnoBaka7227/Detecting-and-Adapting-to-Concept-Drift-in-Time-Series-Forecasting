"""Page-Hinkley drift detector (river) behind the DriftDetector interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.base import DriftDetector, detect_with_river


class PageHinkleyDetector(DriftDetector):
    """`threshold` (lambda) is the test value that fires a drift; `delta`
    the per-step change to tolerate; `min_instances` how many values after
    a reset before it may fire; `alpha` the forgetting factor on the
    cumulative sums (memory ~ 1 / (1 - alpha) steps; 1.0 = no forgetting);
    `mode` which direction of mean shift to watch. Defaults are river's."""

    name = "page_hinkley"

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 1 - 0.0001,
        mode: str = "both",
    ) -> None:
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.mode = mode

    def _river_detector(self) -> drift.PageHinkley:
        return drift.PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
            mode=self.mode,
        )

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        return detect_with_river(self._river_detector(), stream)

    def statistic_trace(self, stream: np.ndarray | pd.Series) -> tuple[list[int], np.ndarray]:
        """Replay `detect` and also return the Page-Hinkley test value at
        every step -- for display only. river computes it inside `update`
        without storing it, so it is rebuilt from the same running sums:
        max(increase, decrease) cumulative deviation, compared against
        `threshold`. NaN until `min_instances` values have been seen since
        the last reset."""
        detector = self._river_detector()
        values = stream.to_numpy() if isinstance(stream, pd.Series) else np.asarray(stream)
        statistic = np.full(len(values), np.nan)
        flagged: list[int] = []
        for i, x in enumerate(values):
            detector.update(float(x))
            if detector.drift_detected:
                flagged.append(i)
            if detector._x_mean.n >= self.min_instances:
                statistic[i] = max(
                    detector._sum_increase - detector._min_increase,
                    detector._max_decrease - detector._sum_decrease,
                )
        return flagged, statistic
