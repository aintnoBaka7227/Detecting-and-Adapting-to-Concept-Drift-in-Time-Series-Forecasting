"""FROZEN CONTRACT (Sprint 1 — do not change the shape after Checkpoint 1):

    detector(stream) -> changepoint indices

See docs/interfaces.md for the full rationale.

How to implement a detector
---------------------------
Subclass `DriftDetector` and process observations in order:

    import numpy as np
    from drift_lab.detection.base import DriftDetector

    class ADWINDetector(DriftDetector):
        def __init__(self, delta: float = 0.002):
            self.delta = delta

        def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
            values = stream_values(stream)
            # Update online state once per value and record flagged indices.
            ...
            return changepoint_indices

The public batch-style method keeps every detector comparable on the
synthetic benchmark before it is evaluated on AEMO.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class DriftDetector(ABC):
    name: str = ""
    implementation: str = "native_v2"

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


def stream_values(stream: np.ndarray | pd.Series) -> np.ndarray:
    """Convert a detector input to a finite, one-dimensional float array."""
    values = stream.to_numpy(dtype=float) if isinstance(stream, pd.Series) else np.asarray(stream, dtype=float)
    if values.ndim != 1:
        raise ValueError("stream must be one-dimensional")
    if not np.isfinite(values).all():
        raise ValueError("stream must contain only finite values")
    return values


def detect_with_river(river_detector, stream: np.ndarray | pd.Series) -> list[int]:
    """Adapt a River-style online detector to the project batch interface."""
    values = stream.to_numpy() if isinstance(stream, pd.Series) else np.asarray(stream)
    flagged: list[int] = []
    for i, x in enumerate(values):
        river_detector.update(float(x))
        if river_detector.drift_detected:
            flagged.append(i)
    return flagged
