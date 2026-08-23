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


def save_series_to_csv(
    kind: Kind,
    n: int,
    noise: float,
    seed: int | None,
    out_dir: str,
) -> str:
    """
    Generate one series with make_series() and save it as a CSV file.

    This is a convenience wrapper for teammates who'd rather load a CSV
    (e.g. into pandas, Excel, or another language) than call make_series()
    directly in Python. It does NOT change make_series() itself — the
    frozen interface stays exactly as agreed.

    The CSV has two columns:
        t              : time index (0 .. n-1)
        y              : the generated series value
        is_changepoint : 1 at a true drift index, 0 otherwise (so the
                          ground truth travels with the data even outside
                          Python)

    Parameters
    ----------
    kind, n, noise, seed : same as make_series()
    out_dir : str
        Folder to save the CSV into. Created if it doesn't exist.

    Returns
    -------
    str : the full path of the saved CSV file.
    """
    import csv

    y, changepoints = make_series(kind=kind, n=n, noise=noise, seed=seed)
    changepoint_set = set(changepoints)

    os.makedirs(out_dir, exist_ok=True)
    filename = f"series_{kind}_n{n}_seed{seed}.csv"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "y", "is_changepoint"])
        for t_idx, y_val in enumerate(y):
            writer.writerow([t_idx, y_val, 1 if t_idx in changepoint_set else 0])

    return out_path


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

    # Save one CSV per drift type, next to this script, in a "csv_output" folder.
    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_output")
    print(f"\nSaving CSVs to {csv_dir} ...")
    for kind in kinds:
        path = save_series_to_csv(kind=kind, n=20000, noise=1.0, seed=7, out_dir=csv_dir)
        print(f"  {kind:<10} -> {path}")