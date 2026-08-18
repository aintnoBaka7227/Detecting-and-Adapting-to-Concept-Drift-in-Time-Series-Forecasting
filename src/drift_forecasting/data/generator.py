"""
Synthetic Data Generator for Concept Drift Detection
=====================================================

Implements make_series(), matching the interface frozen in the project
brief:

    make_series(kind, n, noise) -> (y, changepoints)

- y            : numpy array of shape (n,), the generated series
- changepoints : list[int], the true indices where drift begins
                 (ground truth, used to score detectors)

Supported kind values
----------------------
- "none"      : no drift at all (control case — a detector should find
                 nothing here)
- "sudden"    : an instant step change
- "gradual"   : the new pattern is interpolated in over ~1000 steps
- "recurring" : the series alternates back and forth between two
                 regimes (old pattern returns)

This mirrors the minimal generator shown in the brief:

    t < 5000   y = 20 + 5*sin(2*pi*t/48) + noise
    t >= 5000  y = 20 + 5*sin(2*pi*t/48) + 8 + noise
    t >= 9000  y = 20 + 9*sin(2*pi*t/48) + noise
    changepoints = [5000, 9000]

but is generalised so `n` and drift locations scale with series length.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Literal

Kind = Literal["none", "sudden", "gradual", "recurring"]


def make_series(
    kind: Kind = "sudden",
    n: int = 20000,
    noise: float = 1.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Generate a synthetic time series with known concept drift.

    Parameters
    ----------
    kind : {"none", "sudden", "gradual", "recurring"}
        The type of drift to inject.
    n : int
        Number of time steps to generate.
    noise : float
        Standard deviation of additive Gaussian noise.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    y : np.ndarray, shape (n,)
        The generated series.
    changepoints : list[int]
        True indices where drift begins (ground truth).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)

    period = 48  # half-hourly data -> 48 steps/day, matches the brief's example
    base_level = 20.0
    base_amp = 5.0

    if kind == "none":
        y = base_level + base_amp * np.sin(2 * np.pi * t / period)
        y = y + rng.normal(0, noise, n)
        changepoints: List[int] = []

    elif kind == "sudden":
        cp = n // 2
        y = base_level + base_amp * np.sin(2 * np.pi * t / period)
        y = y + np.where(t >= cp, 8.0, 0.0)
        y = y + rng.normal(0, noise, n)
        changepoints = [cp]

    elif kind == "gradual":
        cp = n // 2
        width = min(1000, n // 4)  # interpolate over ~1000 steps, per the brief
        w = np.clip((t - cp) / width, 0.0, 1.0)
        pre = base_level + base_amp * np.sin(2 * np.pi * t / period)
        post = base_level + base_amp * np.sin(2 * np.pi * t / period) + 8.0
        y = (1 - w) * pre + w * post
        y = y + rng.normal(0, noise, n)
        changepoints = [cp]

    elif kind == "recurring":
        # A -> B -> A: old pattern returns partway through
        cp1 = n // 3
        cp2 = (2 * n) // 3
        regime = np.where((t >= cp1) & (t < cp2), 1, 0)  # 1 = B, 0 = A
        y = base_level + base_amp * np.sin(2 * np.pi * t / period)
        y = y + np.where(regime == 1, 8.0, 0.0)
        y = y + rng.normal(0, noise, n)
        changepoints = [cp1, cp2]

    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    return y, changepoints


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=False)
    kinds: List[Kind] = ["none", "sudden", "gradual", "recurring"]

    for ax, kind in zip(axes, kinds):
        y, changepoints = make_series(kind=kind, n=20000, noise=1.0, seed=7)
        ax.plot(y, linewidth=0.5, color="#3b5b6b")
        for cp in changepoints:
            ax.axvline(cp, color="#c0392b", linestyle="--", alpha=0.8)
        ax.set_title(f"kind = '{kind}'  |  changepoints = {changepoints}")
        ax.set_ylabel("y")

    axes[-1].set_xlabel("t")
    fig.tight_layout()

    # Save next to this script, not to a hardcoded absolute path,
    # so it works on any machine.
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_examples.png")
    fig.savefig(out_path, dpi=130)
    print(f"Saved preview to {out_path}")

    # Sanity check matching the brief's required test:
    # a detector on a no-drift series should find nothing.
    y_none, cps_none = make_series(kind="none", n=20000, noise=1.0, seed=1)
    assert cps_none == [], "no-drift series should report zero changepoints"
    print("Sanity check passed: kind='none' has zero changepoints.")

    y, cps = make_series(kind="sudden", n=20000, noise=1.0, seed=42)
    print(f"\nExample: kind='sudden', n=20000 -> changepoints={cps}")
    print(f"y.shape = {y.shape}, y[:5] = {y[:5]}")


