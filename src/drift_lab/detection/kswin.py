"""Native KS-window drift detector behind the DriftDetector interface."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pandas as pd

from drift_lab.detection.base import DriftDetector, stream_values


class KSWINDetector(DriftDetector):
    name = "kswin"

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int = 42,
    ) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if stat_size < 2:
            raise ValueError("stat_size must be at least 2")
        if window_size < 2 * stat_size:
            raise ValueError("window_size must be at least twice stat_size")
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed

    def detect(self, stream: np.ndarray | pd.Series) -> list[int]:
        values = stream_values(stream)
        rng = np.random.default_rng(self.seed)
        window: deque[float] = deque(maxlen=self.window_size)
        changepoints: list[int] = []

        for index, value in enumerate(values):
            window.append(float(value))
            if len(window) < self.window_size or (index + 1) % self.stat_size:
                continue

            values_in_window = np.fromiter(window, dtype=float, count=len(window))
            reference_pool = values_in_window[:-self.stat_size]
            recent = values_in_window[-self.stat_size:]
            sample_indices = rng.choice(
                len(reference_pool),
                size=self.stat_size,
                replace=False,
            )
            reference = reference_pool[sample_indices]

            if self._ks_statistic(reference, recent) > self._critical_value():
                changepoints.append(index)
                window = deque(recent.tolist(), maxlen=self.window_size)

        return changepoints

    def _critical_value(self) -> float:
        effective_size = self.stat_size / 2.0
        return math.sqrt(-0.5 * math.log(self.alpha / 2.0) / effective_size)

    @staticmethod
    def _ks_statistic(reference: np.ndarray, recent: np.ndarray) -> float:
        support = np.sort(np.concatenate((reference, recent)))
        reference_cdf = np.searchsorted(np.sort(reference), support, side="right") / len(reference)
        recent_cdf = np.searchsorted(np.sort(recent), support, side="right") / len(recent)
        return float(np.max(np.abs(reference_cdf - recent_cdf)))
