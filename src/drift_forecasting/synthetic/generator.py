

from __future__ import annotations

import os
import numpy as np
from typing import List, Tuple, Literal

Kind = Literal["none", "sudden", "gradual", "recurring"]


def make_series(
    kind: Kind = "sudden",
    n: int = 20000,
    noise: float = 1.0,
    seed: int | None = None,
) -> Tuple[np.ndarray, List[int]]:
  
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
    import matplotlib.pyplot as plt

    # Sanity check matching the brief's required test:
    # a detector on a no-drift series should find nothing.
    # This is a one-off check, NOT part of the 15-output batch below —
    # 'none' is a test case, not a seeded dataset variant.
    y_none, cps_none = make_series(kind="none", n=20000, noise=1.0, seed=1)
    assert cps_none == [], "no-drift series should report zero changepoints"
    print("Sanity check passed: kind='none' has zero changepoints.")

    # Team-agreed seed list — use these 5 for every experiment all semester,
    # so results are comparable across every table (T1-T7) and both groups.
    SEEDS = [1, 2, 3, 4, 5]
    OUTPUT_KINDS: List[Kind] = ["sudden", "gradual", "recurring"]

    csv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_output")
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plot_output")
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\nGenerating {len(OUTPUT_KINDS) * len(SEEDS)} (graph + CSV) pairs "
          f"for kinds={OUTPUT_KINDS}, seeds={SEEDS} ...\n")

    for kind in OUTPUT_KINDS:
        for seed in SEEDS:
            y, changepoints = make_series(kind=kind, n=20000, noise=1.0, seed=seed)

            # --- CSV ---
            csv_path = save_series_to_csv(kind=kind, n=20000, noise=1.0, seed=seed, out_dir=csv_dir)

            # --- Graph ---
            fig, ax = plt.subplots(figsize=(10, 3.5))
            ax.plot(y, linewidth=0.5, color="#3b5b6b")
            for cp in changepoints:
                ax.axvline(cp, color="#c0392b", linestyle="--", alpha=0.8)
                y_at_cp = y[cp]
                ax.plot(cp, y_at_cp, marker="o", color="#c0392b", markersize=5, zorder=5)
                ax.annotate(
                    f"({cp}, {y_at_cp:.2f})",
                    xy=(cp, y_at_cp),
                    xytext=(8, 10),
                    textcoords="offset points",
                    fontsize=8,
                    color="#c0392b",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c0392b", alpha=0.85),
                )
            ax.set_title(f"kind='{kind}', seed={seed}  |  changepoints={changepoints}")
            ax.set_xlabel("t")
            ax.set_ylabel("y")
            fig.tight_layout()

            plot_path = os.path.join(plot_dir, f"plot_{kind}_seed{seed}.png")
            fig.savefig(plot_path, dpi=130)
            plt.close(fig)

            print(f"  {kind:<10} seed={seed} -> {os.path.basename(csv_path)}, {os.path.basename(plot_path)}")

    print(f"\nDone. {len(OUTPUT_KINDS) * len(SEEDS)} CSVs in {csv_dir}")
    print(f"      {len(OUTPUT_KINDS) * len(SEEDS)} plots in {plot_dir}")

    # Also keep the original 4-panel overview (none/sudden/gradual/recurring,
    # single seed) as a quick-glance summary figure.
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=False)
    overview_kinds: List[Kind] = ["none", "sudden", "gradual", "recurring"]
    for ax, kind in zip(axes, overview_kinds):
        y, changepoints = make_series(kind=kind, n=20000, noise=1.0, seed=7)
        ax.plot(y, linewidth=0.5, color="#3b5b6b")
        for cp in changepoints:
            ax.axvline(cp, color="#c0392b", linestyle="--", alpha=0.8)
            y_at_cp = y[cp]
            ax.plot(cp, y_at_cp, marker="o", color="#c0392b", markersize=5, zorder=5)
            ax.annotate(
                f"({cp}, {y_at_cp:.2f})",
                xy=(cp, y_at_cp),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=9,
                color="#c0392b",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#c0392b", alpha=0.85),
            )
        ax.set_title(f"kind = '{kind}'  |  changepoints = {changepoints}")
        ax.set_ylabel("y")
    axes[-1].set_xlabel("t")
    fig.tight_layout()
    overview_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_examples.png")
    fig.savefig(overview_path, dpi=130)
    print(f"\nSaved overview figure to {overview_path}")

