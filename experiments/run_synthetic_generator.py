"""Step 2 deliverable: the synthetic drift benchmark, visualised.

One figure per drift type (none / sudden / gradual / recurring), each with
one panel per seed, true changepoints marked. No CSV — `make_series` is
already the single source of this data; these plots are just a look at it.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from drift_lab.config import SEEDS
from drift_lab.synthetic.generator import Kind, make_series
from experiments.results_io import FIGURES_DIR

KINDS: tuple[Kind, ...] = ("none", "sudden", "gradual", "recurring")
N = 20_000
NOISE = 1.0


def plot_kind(kind: Kind) -> None:
    fig, axes = plt.subplots(len(SEEDS), 1, figsize=(12, 2.2 * len(SEEDS)), sharex=True)
    for ax, seed in zip(axes, SEEDS):
        y, changepoints = make_series(kind=kind, n=N, noise=NOISE, seed=seed)
        ax.plot(y, linewidth=0.5, color="#3b5b6b")
        for cp in changepoints:
            ax.axvline(cp, color="#c0392b", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(f"seed={seed}")
        ax.grid(alpha=0.25)

    axes[0].set_title(f"Synthetic benchmark — kind='{kind}'", loc="left")
    axes[-1].set_xlabel("t")
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"synthetic_{kind}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    for kind in KINDS:
        plot_kind(kind)


if __name__ == "__main__":
    main()
