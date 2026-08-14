"""FROZEN CONTRACT (Sprint 1 — do not change the shape after Checkpoint 1):

    detector(stream) -> changepoint indices

See docs/interfaces.md for the full rationale.

How to implement a detector (Sprint 3, Step 4)
-----------------------------------------------
Add a new file next to this one, e.g. `adwin.py`, and subclass
`DriftDetector`:

    import numpy as np
    from drift_forecasting.detection.base import DriftDetector

    class ADWINDetector(DriftDetector):
        def __init__(self, delta: float = 0.002):
            self.delta = delta

        def detect(self, stream: np.ndarray) -> list[int]:
            # e.g. feed `stream` through river.drift.ADWIN one value at a
            # time and collect the indices where it flags a change.
            ...
            return changepoint_indices

Wrapping an online/streaming detector (river's ADWIN, Page-Hinkley,
KSWIN, ...) is fine internally — the class just has to expose this one
batch-style method so every detector can be run and scored the same way
against the synthetic benchmark (Step 4) before ever touching AEMO.
"""

from abc import ABC, abstractmethod

import numpy as np


class DriftDetector(ABC):
    @abstractmethod
    def detect(self, stream: np.ndarray) -> list[int]:
        """Return indices into `stream` flagged as changepoints.

        Run on the synthetic benchmark first (Step 4) and scored against
        its known changepoints before ever touching AEMO. Implementations
        may only use values up to each index when deciding on it — no
        full-series statistics — otherwise the "detects changes as they
        happen" story is fiction.
        """
        raise NotImplementedError
