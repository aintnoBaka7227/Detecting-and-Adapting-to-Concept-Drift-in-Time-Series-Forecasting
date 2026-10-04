"""Native adaptive-window drift detector behind the DriftDetector interface."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from drift_lab.detection.base import DriftDetector, stream_values


class ADWINDetector(DriftDetector):
    name = "adwin"

    _CHECK_INTERVAL = 16
    _MIN_SUBWINDOW_SIZE = 32

    def __init__(
        self,
        delta: float = 0.002,
        max_window_size: int = 2048,
        cooldown: int = 1024,
    ) -> None:
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be between 0 and 1")
        if max_window_size < 2 * self._MIN_SUBWINDOW_SIZE:
            raise ValueError("max_window_size must be at least 64")
        if cooldown < 0:
            raise ValueError("cooldown must be non-negative")
        self.delta = delta
        self.max_window_size = max_window_size
        self.cooldown = cooldown

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        values = stream_values(stream)
        window: deque[float] = deque(maxlen=self.max_window_size)
        changepoints: list[int] = []
        cooldown_until = -1

        for index, value in enumerate(values):
            window.append(float(value))

            if index < cooldown_until or (index + 1) % self._CHECK_INTERVAL:
                continue

            changed = False
            while len(window) >= 2 * self._MIN_SUBWINDOW_SIZE:
                cut = self._significant_cut(window)
                if cut is None:
                    break
                for _ in range(cut):
                    window.popleft()
                changed = True

            if changed:
                changepoints.append(index)
                cooldown_until = index + self.cooldown

        return changepoints

    def _significant_cut(self, window: deque[float]) -> int | None:
        values = np.fromiter(window, dtype=float, count=len(window))
        size = len(values)
        cuts = np.arange(
            self._MIN_SUBWINDOW_SIZE,
            size - self._MIN_SUBWINDOW_SIZE + 1,
            self._MIN_SUBWINDOW_SIZE // 2,
        )
        if cuts.size == 0:
            return None

        prefix = np.cumsum(values)
        total = prefix[-1]
        left_means = prefix[cuts - 1] / cuts
        right_means = (total - prefix[cuts - 1]) / (size - cuts)
        variance = float(np.var(values, ddof=1))
        effective_size = cuts * (size - cuts) / size

        log_term = math.log(2.0 * math.log(size) / self.delta)
        epsilon = np.sqrt(2.0 * variance * log_term / effective_size)
        epsilon += 2.0 * log_term / (3.0 * effective_size)

        significant = np.flatnonzero(np.abs(left_means - right_means) > epsilon)
        return int(cuts[significant[0]]) if significant.size else None
