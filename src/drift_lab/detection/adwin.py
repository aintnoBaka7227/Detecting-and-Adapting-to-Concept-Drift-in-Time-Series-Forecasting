"""ADWIN drift detector (river) behind the DriftDetector interface."""

from __future__ import annotations

import numpy as np
import pandas as pd
from river import drift

from drift_lab.detection.base import DriftDetector, detect_with_river


class ADWINDetector(DriftDetector):
    name = "adwin"

    def __init__(self, delta: float = 0.002) -> None:
        self.delta = delta

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        return detect_with_river(drift.ADWIN(delta=self.delta), stream)
