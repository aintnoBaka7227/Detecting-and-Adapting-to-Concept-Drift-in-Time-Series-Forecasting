"""Detection contract, frozen in Sprint 1:

    detector(stream) -> changepoint indices

Each detector subclasses DriftDetector (e.g. adwin.py). Wrapping an online
detector (river's ADWIN, Page-Hinkley, KSWIN) is fine; the class only has to
expose one batch `detect`, so every detector is scored the same way on the
synthetic benchmark before AEMO.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DriftDetector(ABC):
    name: str = ""

    @abstractmethod
    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        """Return positional indices into `stream` flagged as changepoints.

        Run on the synthetic benchmark first (Step 4) and scored against
        its known changepoints before ever touching AEMO. Implementations
        may only use values up to each index when deciding on it — no
        full-series statistics — otherwise the "detects changes as they
        happen" story is fiction.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Drop any internal state so the detector can be reused. Default no-op."""


def detect_with_river(river_detector, stream: np.ndarray | pd.Series) -> list[int]:
    """Feed `stream` through a fresh river drift detector one value at a time,
    returning the positional indices where it flags a change."""
    values = stream.to_numpy() if isinstance(stream, pd.Series) else np.asarray(stream)
    flagged: list[int] = []
    for i, x in enumerate(values):
        river_detector.update(float(x))
        if river_detector.drift_detected:
            flagged.append(i)
    return flagged
