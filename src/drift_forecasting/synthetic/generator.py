"""Synthetic drift benchmark generator — Step 2."""

from typing import Literal

import numpy as np


DriftKind = Literal[
    "sudden",
    "gradual",
    "recurring",
    "none",
]


def make_series(
    kind: DriftKind,
    n: int,
    noise: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, list[int]]:

    rng = np.random.default_rng(seed)
    t = np.arange(n)

    period = 48
    base_level = 20.0
    base_amp = 5.0

    if kind == "none":
        y = base_level + base_amp * np.sin(
            2 * np.pi * t / period
        )
        y = y + rng.normal(0, noise, n)

        changepoints = []

    elif kind == "sudden":
        cp = n // 2

        y = base_level + base_amp * np.sin(
            2 * np.pi * t / period
        )

        y = y + np.where(
            t >= cp,
            8.0,
            0.0,
        )

        y = y + rng.normal(0, noise, n)

        changepoints = [cp]

    elif kind == "gradual":
        cp = n // 2
        width = min(1000, n // 4)

        w = np.clip(
            (t - cp) / width,
            0.0,
            1.0,
        )

        pre = (
            base_level
            + base_amp * np.sin(2 * np.pi * t / period)
        )

        post = (
            base_level
            + base_amp * np.sin(2 * np.pi * t / period)
            + 8.0
        )

        y = (1 - w) * pre + w * post
        y = y + rng.normal(0, noise, n)

        changepoints = [cp]

    elif kind == "recurring":
        cp1 = n // 3
        cp2 = (2 * n) // 3

        regime = np.where(
            (t >= cp1) & (t < cp2),
            1,
            0,
        )

        y = base_level + base_amp * np.sin(
            2 * np.pi * t / period
        )

        y = y + np.where(
            regime == 1,
            8.0,
            0.0,
        )

        y = y + rng.normal(0, noise, n)

        changepoints = [cp1, cp2]

    else:
        raise ValueError(
            f"Unknown drift kind: {kind!r}"
        )

    return y, changepoints