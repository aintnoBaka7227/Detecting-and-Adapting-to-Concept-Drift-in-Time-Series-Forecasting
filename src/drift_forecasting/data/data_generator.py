"""
Synthetic Data Generator for Concept Drift Detection
=====================================================

Provides make_series(): generates a synthetic time series with known,
labelled drift points/regions, so drift-detection algorithms can be
evaluated against ground truth (not just eyeballed).

Supported drift types
----------------------
- "sudden"      : abrupt regime change at a single point (step change)
- "gradual"     : probabilistic blend between old/new regime over a window
- "incremental" : smooth, continuous ramp from old regime to new regime
- "recurring"   : periodic switching back and forth between two regimes
- "none"        : no drift at all (useful as a control/baseline series)

Each "regime" is defined by a signal generating function, e.g. a base
level, trend, seasonality amplitude, and noise level. Drift = the
underlying regime parameters change over time.

Usage
-----
    from data_generator import make_series

    df = make_series(
        n_samples=2000,
        drift_type="gradual",
        drift_point=1000,
        drift_width=150,
        seed=42,
    )
    # df columns: ['t', 'value', 'regime', 'is_drift_region']
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal


DriftType = Literal["sudden", "gradual", "incremental", "recurring", "none"]


@dataclass
class Regime:
    """Defines the statistical behaviour of a time series segment."""
    level: float = 0.0          # baseline mean
    trend: float = 0.0          # linear drift per timestep
    seasonality_amp: float = 0.0
    seasonality_period: int = 50
    noise_std: float = 1.0

    def generate(self, t: np.ndarray) -> np.ndarray:
        """Generate raw signal values for the given time indices."""
        seasonal = self.seasonality_amp * np.sin(2 * np.pi * t / self.seasonality_period)
        trend_component = self.trend * t
        return self.level + trend_component + seasonal


def _default_regime_a() -> Regime:
    return Regime(level=0.0, trend=0.0, seasonality_amp=2.0, seasonality_period=50, noise_std=1.0)


def _default_regime_b() -> Regime:
    # A "post-drift" regime: shifted mean, steeper trend, more noise.
    return Regime(level=8.0, trend=0.01, seasonality_amp=2.0, seasonality_period=50, noise_std=1.6)


def make_series(
    n_samples: int = 2000,
    drift_type: DriftType = "sudden",
    drift_point: Optional[int] = None,
    drift_width: int = 100,
    n_recurring_cycles: int = 4,
    regime_a: Optional[Regime] = None,
    regime_b: Optional[Regime] = None,
    noise_std: Optional[float] = None,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic univariate time series with a labelled concept drift.

    Parameters
    ----------
    n_samples : int
        Total number of time steps to generate.
    drift_type : {"sudden", "gradual", "incremental", "recurring", "none"}
        The kind of concept drift to inject.
    drift_point : int, optional
        Index at which drift begins. Defaults to the midpoint of the series.
        Ignored for "recurring" (which uses n_recurring_cycles instead).
    drift_width : int
        Width (in timesteps) of the transition window for "gradual" and
        "incremental" drift. Ignored for "sudden" and "none".
    n_recurring_cycles : int
        For "recurring" drift only: how many times the series alternates
        between regime_a and regime_b.
    regime_a : Regime, optional
        Pre-drift regime. Uses a sensible default if not provided.
    regime_b : Regime, optional
        Post-drift regime. Uses a sensible default if not provided.
    noise_std : float, optional
        If provided, overrides the noise_std of both regimes uniformly.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns:
        t                : integer time index
        value             : the generated series value
        regime            : "A" or "B" (which regime generated this point;
                             for gradual/incremental this reflects the
                             dominant regime after blending/interpolation)
        is_drift_region   : bool, True during the drift transition window
                             (for "sudden", only the exact drift point is True)
        true_drift_point  : bool, True only at the/each canonical drift
                             onset index (useful as the single "ground truth"
                             marker drift detectors are scored against)
    """
    rng = np.random.default_rng(seed)

    regime_a = regime_a or _default_regime_a()
    regime_b = regime_b or _default_regime_b()

    if noise_std is not None:
        regime_a = Regime(**{**regime_a.__dict__, "noise_std": noise_std})
        regime_b = Regime(**{**regime_b.__dict__, "noise_std": noise_std})

    t = np.arange(n_samples)

    if drift_point is None:
        drift_point = n_samples // 2

    if drift_type == "none":
        values = regime_a.generate(t) + rng.normal(0, regime_a.noise_std, n_samples)
        regime_labels = np.array(["A"] * n_samples)
        is_drift_region = np.zeros(n_samples, dtype=bool)
        true_drift_point = np.zeros(n_samples, dtype=bool)

    elif drift_type == "sudden":
        base_a = regime_a.generate(t)
        base_b = regime_b.generate(t)
        values = np.where(t < drift_point, base_a, base_b)
        values = values + rng.normal(
            0,
            np.where(t < drift_point, regime_a.noise_std, regime_b.noise_std),
        )
        regime_labels = np.where(t < drift_point, "A", "B")
        is_drift_region = np.zeros(n_samples, dtype=bool)
        is_drift_region[drift_point] = True
        true_drift_point = is_drift_region.copy()

    elif drift_type == "incremental":
        # Smooth linear ramp from regime A to regime B over drift_width steps.
        base_a = regime_a.generate(t)
        base_b = regime_b.generate(t)
        w = np.clip((t - drift_point) / drift_width, 0.0, 1.0)
        values = (1 - w) * base_a + w * base_b
        noise_std_t = (1 - w) * regime_a.noise_std + w * regime_b.noise_std
        values = values + rng.normal(0, noise_std_t)
        regime_labels = np.where(w < 0.5, "A", "B")
        is_drift_region = (t >= drift_point) & (t < drift_point + drift_width)
        true_drift_point = np.zeros(n_samples, dtype=bool)
        true_drift_point[drift_point] = True

    elif drift_type == "gradual":
        # Probabilistic blend: each point independently drawn from A or B,
        # with P(B) ramping from 0 to 1 across the drift window. This
        # mimics real gradual drift (intermittent old/new behaviour)
        # rather than a smooth deterministic ramp.
        base_a = regime_a.generate(t)
        base_b = regime_b.generate(t)
        p_b = np.clip((t - drift_point) / drift_width, 0.0, 1.0)
        draw_b = rng.random(n_samples) < p_b
        values = np.where(draw_b, base_b, base_a)
        values = values + rng.normal(
            0,
            np.where(draw_b, regime_b.noise_std, regime_a.noise_std),
        )
        regime_labels = np.where(draw_b, "B", "A")
        is_drift_region = (t >= drift_point) & (t < drift_point + drift_width)
        true_drift_point = np.zeros(n_samples, dtype=bool)
        true_drift_point[drift_point] = True

    elif drift_type == "recurring":
        cycle_len = n_samples // n_recurring_cycles
        base_a = regime_a.generate(t)
        base_b = regime_b.generate(t)
        cycle_idx = (t // cycle_len) % 2
        values = np.where(cycle_idx == 0, base_a, base_b)
        values = values + rng.normal(
            0,
            np.where(cycle_idx == 0, regime_a.noise_std, regime_b.noise_std),
        )
        regime_labels = np.where(cycle_idx == 0, "A", "B")
        # A drift event happens at every cycle boundary.
        boundaries = np.arange(cycle_len, n_samples, cycle_len)
        is_drift_region = np.zeros(n_samples, dtype=bool)
        is_drift_region[boundaries] = True
        true_drift_point = is_drift_region.copy()

    else:
        raise ValueError(f"Unknown drift_type: {drift_type!r}")

    df = pd.DataFrame(
        {
            "t": t,
            "value": values,
            "regime": regime_labels,
            "is_drift_region": is_drift_region,
            "true_drift_point": true_drift_point,
        }
    )
    return df


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=False)
    configs = [
        ("none", {}),
        ("sudden", {"drift_point": 1000}),
        ("gradual", {"drift_point": 900, "drift_width": 250}),
        ("incremental", {"drift_point": 900, "drift_width": 250}),
        ("recurring", {"n_recurring_cycles": 4}),
    ]

    for ax, (dtype, kwargs) in zip(axes, configs):
        df = make_series(n_samples=2000, drift_type=dtype, seed=7, **kwargs)
        ax.plot(df["t"], df["value"], linewidth=0.8, color="#3b5b6b")
        drift_idx = df.index[df["true_drift_point"]]
        for idx in drift_idx:
            ax.axvline(df["t"][idx], color="#c0392b", linestyle="--", alpha=0.7)
        ax.set_title(f"drift_type = '{dtype}'")
        ax.set_ylabel("value")

    axes[-1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig("/home/claude/drift_examples.png", dpi=130)
    print("Saved preview to drift_examples.png")

    # Quick sanity check / example of intended usage
    example = make_series(n_samples=2000, drift_type="sudden", drift_point=1000, seed=42)
    print(example.head())
    print(f"\nTotal rows: {len(example)}, drift point flagged at t=", 
          example.loc[example["true_drift_point"], "t"].tolist())
