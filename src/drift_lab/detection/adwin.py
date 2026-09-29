"""ADWIN drift detector (river) behind the DriftDetector interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.base import DriftDetector, detect_with_river


class ADWINDetector(DriftDetector):
    """`delta` sets the confidence of each cut; `clock` how often (in
    observations) a cut is checked; `min_window_length` the shortest
    sub-window a cut may compare. Defaults are river's."""

    name = "adwin"

    def __init__(
        self,
        delta: float = 0.002,
        clock: int = 32,
        min_window_length: int = 5,
    ) -> None:
        self.delta = delta
        self.clock = clock
        self.min_window_length = min_window_length

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        return detect_with_river(
            drift.ADWIN(
                delta=self.delta,
                clock=self.clock,
                min_window_length=self.min_window_length,
            ),
            stream,
        )
