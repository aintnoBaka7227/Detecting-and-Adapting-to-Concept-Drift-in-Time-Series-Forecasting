"""Shared build logic for the synthetic Figure F2 companions.

Two sibling producers read this module:

- produce_figure_f2_synthetic_pre_tune.py   class-default detector configs
                                             (run_pre_tune_on_synthetic.py's rows)
- produce_figure_f2_synthetic_post_tune.py  post-tuning (frozen) configs
                                             (run_post_tune_on_synthetic.py's rows)

Each drift type with an actual changepoint (sudden, gradual, recurring --
"none" has nothing to mark) gets one figure: one panel per seed, the raw
series in the background, the true drift point(s) in blue, and each
detector's matched detection as its own coloured vertical line labelled
with its delay. Matching itself is not reimplemented here -- each panel
calls the same `evaluation.evaluate_detections()` the report tables use,
so this figure's delays can never drift from what T1 reports.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from drift_lab.config import SEEDS
from drift_lab.evaluation.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series
from experiments.results_io import RUNS_CSV, synthetic_detections_path

N = 20_000
NOISE = 1.0

# "none" has no changepoint to mark, so it has no F2 companion.
KINDS = ("sudden", "gradual", "recurring")

DETECTORS = ("adwin", "kswin", "page_hinkley")
DETECTOR_DISPLAY_NAME = {
    "adwin": "ADWIN",
    "kswin": "KSWIN",
    "page_hinkley": "Page-Hinkley",
}
# Same accent colours as produce_figure_f1.py's MODEL_STYLE, for palette
# consistency across the project's figures.
DETECTOR_COLOR = {
    "adwin": "#198754",
    "kswin": "#E67E22",
    "page_hinkley": "#7D3C98",
}
SERIES_COLOR = "#9AA5B1"
TRUE_POINT_COLOR = "#1464A5"


def load_runs() -> pd.DataFrame:
    if not RUNS_CSV.exists():
        raise FileNotFoundError(f"{RUNS_CSV} does not exist. Run the synthetic detector experiment first.")
    return pd.read_csv(RUNS_CSV)


def load_detections(
    runs: pd.DataFrame,
    dataset: str,
    seed: int,
    split_id_filter: Callable[[pd.Series], pd.Series],
) -> dict[str, list[int]]:
    """Persisted detection indices for all three detectors, one (dataset, seed)."""
    detections: dict[str, list[int]] = {}

    for method in DETECTORS:
        rows = runs[
            (runs["dataset"] == dataset) & (runs["seed"] == seed) & (runs["method"] == method)
        ]
        rows = rows[split_id_filter(rows["split_id"])]

        if rows.empty:
            raise RuntimeError(f"No recorded run found for {dataset}, {method}, seed={seed}")

        # runs.csv is append-only: a config changed after re-tuning (or any
        # other re-run) adds a fresh batch of rows rather than replacing the
        # old one, so more than one config_hash can exist here. Keep only
        # the most recent record_run() call's rows.
        latest_timestamp = rows["timestamp"].max()
        rows = rows[rows["timestamp"] == latest_timestamp]

        config_hashes = rows["config_hash"].dropna().unique()
        if len(config_hashes) != 1:
            raise RuntimeError(
                f"Expected one config_hash for {dataset}, {method}, seed={seed}, "
                f"found {config_hashes.tolist()}"
            )

        path = synthetic_detections_path(config_hashes[0], dataset, seed)
        if not path.exists():
            raise FileNotFoundError(f"Persisted detections not found: {path}")

        frame = pd.read_csv(path)
        if "observation" not in frame.columns:
            raise RuntimeError(f"{path} is missing required column: 'observation'")

        detections[method] = frame["observation"].astype(int).tolist()

    return detections


def plot_seed_panel(
    ax: plt.Axes,
    *,
    kind: str,
    seed: int,
    runs: pd.DataFrame,
    split_id_filter: Callable[[pd.Series], pd.Series],
) -> None:
    """Draw one seed's panel: the series, the true drift point(s), and each
    detector's matched detection labelled with its delay."""

    series, true_changepoints = make_series(kind, n=N, noise=NOISE, seed=seed)
    series = np.asarray(series)
    x = np.arange(len(series))

    ax.plot(x, series, color=SERIES_COLOR, linewidth=0.6, alpha=0.9, zorder=1)

    # Gradual's truth is [start, end]; the label goes on start (delay is
    # measured from there too), end is marked but not separately labelled.
    for i, true_point in enumerate(true_changepoints):
        ax.axvline(
            true_point,
            color=TRUE_POINT_COLOR,
            linewidth=1.8 if i == 0 or kind != "gradual" else 1.2,
            linestyle="-" if i == 0 or kind != "gradual" else ":",
            zorder=3,
        )
        if i == 0 or kind != "gradual":
            ax.text(
                true_point,
                1.02,
                f"true: {true_point}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color=TRUE_POINT_COLOR,
            )

    detections = load_detections(runs, f"synthetic_{kind}", seed, split_id_filter)

    label_level = 0
    for method in DETECTORS:
        color = DETECTOR_COLOR[method]
        result = evaluate_detections(
            detected_changepoints=detections[method],
            true_changepoints=true_changepoints,
            n_observations=len(series),
            drift_type=kind,
        )

        if not result["matched_pairs"]:
            ax.text(
                0.995,
                0.95 - 0.12 * label_level,
                f"{DETECTOR_DISPLAY_NAME[method]}: no match",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color=color,
            )
            label_level += 1
            continue

        for pair, delay in zip(result["matched_pairs"], result["delays"]):
            detected_point = pair["detected"]
            ax.axvline(detected_point, color=color, linewidth=1.3, linestyle="--", alpha=0.9, zorder=2)
            ax.text(
                detected_point,
                0.95 - 0.12 * label_level,
                f"{DETECTOR_DISPLAY_NAME[method]} delay: {delay}",
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7.5,
                fontweight="bold",
                color=color,
            )
            label_level += 1

    ax.set_ylabel(f"seed {seed}", fontsize=9)
    ax.grid(alpha=0.15)


def build_figure(
    *,
    kind: str,
    runs: pd.DataFrame,
    split_id_filter: Callable[[pd.Series], pd.Series],
    title: str,
    output: Path,
) -> Path:
    """One figure for `kind`: one panel per seed in SEEDS."""

    fig, axes = plt.subplots(len(SEEDS), 1, figsize=(16, 3.0 * len(SEEDS)), sharex=True)

    for ax, seed in zip(axes, SEEDS):
        plot_seed_panel(ax, kind=kind, seed=seed, runs=runs, split_id_filter=split_id_filter)

    axes[-1].set_xlabel("Observation")

    legend_handles = [
        Line2D([], [], color=TRUE_POINT_COLOR, lw=1.8, label="True drift point"),
        *(
            Line2D([], [], color=DETECTOR_COLOR[m], lw=1.3, ls="--", label=DETECTOR_DISPLAY_NAME[m])
            for m in DETECTORS
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper right",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.995, 1.0),
    )

    fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return output
