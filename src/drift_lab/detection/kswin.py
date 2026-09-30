"""KSWIN drift detector (river) behind the DriftDetector interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.base import DriftDetector, detect_with_river


class KSWINDetector(DriftDetector):
    name = "kswin"

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

    def _river_detector(self) -> drift.KSWIN:
        return drift.KSWIN(
            alpha=self.alpha,
            window_size=self.window_size,
            stat_size=self.stat_size,
            seed=self.seed,
        )

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        return detect_with_river(self._river_detector(), stream)

    def p_value_trace(self, stream: np.ndarray | pd.Series) -> tuple[list[int], np.ndarray]:
        """Replay `detect` and also return the KS-test p-value at every
        step -- for display only. NaN wherever no test ran (the sliding
        window is still filling, e.g. after each drift reset). Same seed,
        same stream -> the same flagged indices as `detect`."""
        detector = self._river_detector()
        values = stream.to_numpy() if isinstance(stream, pd.Series) else np.asarray(stream)
        p_values = np.full(len(values), np.nan)
        flagged: list[int] = []
        for i, x in enumerate(values):
            detector.update(float(x))
            if detector.drift_detected:
                flagged.append(i)
            # A test ran this step iff the window was full (on drift it is
            # truncated to the most recent `stat_size` values afterwards).
            if detector.drift_detected or len(detector.window) >= self.window_size:
                p_values[i] = detector.p_value
        return flagged, p_values
