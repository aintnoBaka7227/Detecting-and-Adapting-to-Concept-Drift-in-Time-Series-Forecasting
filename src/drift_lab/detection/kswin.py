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
        window_size: int = 200,
        stat_size: int = 50,
        seed: int = 42,
    ) -> None:
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        return detect_with_river(
            drift.KSWIN(
                alpha=self.alpha,
                window_size=self.window_size,
                stat_size=self.stat_size,
                seed=self.seed,
            ),
            stream,
        )
