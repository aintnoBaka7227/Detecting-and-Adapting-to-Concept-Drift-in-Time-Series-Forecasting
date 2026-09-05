"""Synthetic drift benchmark — the one generator in the codebase.

`make_series(kind, n, noise, seed)` returns `(y, changepoints)` in memory.
`changepoints` matches the coordinate system `drift_lab.evaluation` expects:
`none` -> `[]`; `sudden` -> `[cp]`; `recurring` -> `[cp1, cp2]` (two point
events); `gradual` -> `[drift_start, drift_end]` (one transition interval).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Kind = Literal["none", "sudden", "gradual", "recurring"]

_PERIOD = 48  # half-hourly data -> 48 steps/day
_BASE_LEVEL = 20.0
_BASE_AMP = 5.0
_SHIFT = 8.0
_GRADUAL_WIDTH = 1000


def make_series(
    kind: Kind = "sudden",
    n: int = 20000,
    noise: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, list[int]]:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = _BASE_LEVEL + _BASE_AMP * np.sin(2 * np.pi * t / _PERIOD)

    if kind == "none":
        y = seasonal
        changepoints: list[int] = []

    elif kind == "sudden":
        cp = n // 4
        y = seasonal + np.where(t >= cp, _SHIFT, 0.0)
        changepoints = [cp]

    elif kind == "gradual":
        start = n // 4
        width = min(_GRADUAL_WIDTH, n // 4)
        ramp = np.clip((t - start) / width, 0.0, 1.0)
        y = seasonal + ramp * _SHIFT
        changepoints = [start, start + width]

    elif kind == "recurring":
        # A -> B -> A: the old pattern returns partway through.
        cp1, cp2 = n // 3, (2 * n) // 3
        in_b = (t >= cp1) & (t < cp2)
        y = seasonal + np.where(in_b, _SHIFT, 0.0)
        changepoints = [cp1, cp2]

    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    return y + rng.normal(0, noise, n), changepoints
