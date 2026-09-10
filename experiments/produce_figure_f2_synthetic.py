"""Produce Figure F2 synthetic drift-detection visual companions.

This producer:
- regenerates the deterministic synthetic series for visualisation only;
- reads persisted detector alarm indices;
- uses the current detector configurations recorded in results/runs.csv;
- does not rerun any detector;
- produces sudden, gradual and recurring drift figures.

These figures use the current detector settings and can be regenerated
after detector hyperparameters are optimised.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from drift_lab.synthetic.generator import make_series
from experiments.results_io import (
    FIGURES_DIR,
    RUNS_CSV,
    synthetic_detections_path,
)

SEED = 1
N = 20_000
NOISE = 1.0

# 48 observations per synthetic daily cycle.
# 15 days is used only as a visual smoothing overlay.
SMOOTH_WINDOW = 15 * 48

SUDDEN_CP = 5000
GRADUAL_START = 5000
GRADUAL_END = 6000
RECURRING_CPS = [6666, 13333]

TOLERANCE = 336


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------


def smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    """Return a centred moving average for visualisation only."""

    pad = window // 2

    padded = np.concatenate(
        [
            np.full(pad, values[0]),
            values,
            np.full(pad, values[-1]),
        ]
    )

    kernel = np.ones(window) / window

    return np.convolve(
        padded,
        kernel,
        mode="valid",
    )[: len(values)]


def load_runs() -> pd.DataFrame:
    """Load experiment provenance from results/runs.csv."""

    if not RUNS_CSV.exists():
        raise FileNotFoundError(
            f"{RUNS_CSV} does not exist. Run the synthetic detector experiment first."
        )

    return pd.read_csv(RUNS_CSV)


def load_alarm_data(
    runs: pd.DataFrame,
    dataset: str,
    seed: int = SEED,
) -> dict[str, pd.DataFrame]:
    """Load persisted alarm indices for all three detectors."""

    alarm_data: dict[str, pd.DataFrame] = {}

    for method in [
        "adwin",
        "kswin",
        "page_hinkley",
    ]:
        rows = runs[
            (runs["dataset"] == dataset)
            & (runs["seed"] == seed)
            & (runs["method"] == method)
        ]

        if rows.empty:
            raise RuntimeError(
                f"No recorded run found for {dataset}, {method}, seed={seed}"
            )

        config_hash = rows.iloc[0]["config_hash"]

        path = synthetic_detections_path(
            config_hash,
            dataset,
            seed,
        )

        if not path.exists():
            raise FileNotFoundError(f"Persisted detections not found: {path}")

        alarm_data[method] = pd.read_csv(path)

    return alarm_data


def display_name(method: str) -> str:
    """Readable detector name."""

    names = {
        "adwin": "ADWIN",
        "kswin": "KSWIN",
        "page_hinkley": "Page-Hinkley",
    }

    return names[method]


# ---------------------------------------------------------------------
# Sudden drift
# ---------------------------------------------------------------------


def plot_sudden(
    runs: pd.DataFrame,
) -> None:
    """Produce the synthetic sudden-drift F2 companion."""

    dataset = "synthetic_sudden"

    alarm_data = load_alarm_data(
        runs,
        dataset,
    )

    series, _ = make_series(
        "sudden",
        n=N,
        noise=NOISE,
        seed=SEED,
    )

    series = np.asarray(series)
    x = np.arange(len(series))

    smooth = smooth_series(
        series,
        SMOOTH_WINDOW,
    )

    matched: dict[str, int | None] = {}
    unmatched: dict[str, list[int]] = {}

    for method, df in alarm_data.items():
        alarms = df["observation"].astype(int).tolist()

        candidates = [
            alarm for alarm in alarms if SUDDEN_CP <= alarm <= SUDDEN_CP + TOLERANCE
        ]

        match = candidates[0] if candidates else None

        matched[method] = match

        unmatched[method] = [alarm for alarm in alarms if alarm != match]

    fig, ax = plt.subplots(figsize=(15, 6.5))

    ax.plot(
        x,
        series,
        linewidth=0.6,
        alpha=0.20,
        label="Observed synthetic series",
    )

    ax.plot(
        x,
        smooth,
        linewidth=2.0,
        label="15-day mean (visualisation only)",
    )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    # True changepoint
    ax.axvline(
        SUDDEN_CP,
        linewidth=2.0,
        label="True changepoint",
    )

    ax.text(
        SUDDEN_CP - 120,
        ymin + 0.02 * yrange,
        "TRUE CHANGEPOINT\nobservation 5000",
        ha="right",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    # Matched detector lines
    for detection in matched.values():
        if detection is None:
            continue

        ax.axvline(
            detection,
            linestyle="--",
            linewidth=1.3,
            alpha=0.85,
        )

    matched_lines = ["MATCHED DETECTIONS"]

    for method in [
        "adwin",
        "kswin",
        "page_hinkley",
    ]:
        detection = matched[method]

        if detection is None:
            matched_lines.append(f"{display_name(method)}: no match")
        else:
            delay = detection - SUDDEN_CP

            matched_lines.append(
                f"{display_name(method)}: {detection}  ·  delay {delay} obs"
            )

    ax.annotate(
        "\n".join(matched_lines),
        xy=(
            matched["adwin"],
            smooth[matched["adwin"]],
        ),
        xytext=(
            6200,
            ymax - 0.06 * yrange,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "alpha": 0.12,
        },
    )

    # ADWIN has one clear early false alarm in this representative run.
    if unmatched["adwin"]:
        first_false_alarm = unmatched["adwin"][0]

        ax.axvline(
            first_false_alarm,
            linestyle="--",
            linewidth=1.2,
            alpha=0.80,
        )

        ax.annotate(
            "FALSE ALARM\nADWIN · no true change",
            xy=(
                first_false_alarm,
                smooth[first_false_alarm],
            ),
            xytext=(
                850,
                ymax - 0.07 * yrange,
            ),
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.0,
            },
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="top",
        )

    summary = (
        "Current configuration — unmatched detections\n"
        f"ADWIN: {len(unmatched['adwin'])}  ·  "
        f"KSWIN: {len(unmatched['kswin'])}  ·  "
        f"Page-Hinkley: {len(unmatched['page_hinkley'])}"
    )

    ax.text(
        0.015,
        0.04,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        style="italic",
        bbox={
            "boxstyle": "round,pad=0.3",
            "alpha": 0.10,
        },
    )

    ax.set_title(
        "F2 — Drift Detection on Synthetic Sudden-Drift Series (Seed 1)",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Synthetic target")

    ax.grid(alpha=0.18)

    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()

    output = FIGURES_DIR / "synthetic_sudden_detection.png"

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote {output}")


# ---------------------------------------------------------------------
# Gradual drift
# ---------------------------------------------------------------------


def plot_gradual(
    runs: pd.DataFrame,
) -> None:
    """Produce the synthetic gradual-drift F2 companion."""

    dataset = "synthetic_gradual"

    alarm_data = load_alarm_data(
        runs,
        dataset,
    )

    series, _ = make_series(
        "gradual",
        n=N,
        noise=NOISE,
        seed=SEED,
    )

    series = np.asarray(series)
    x = np.arange(len(series))

    smooth = smooth_series(
        series,
        SMOOTH_WINDOW,
    )

    matched: dict[str, int | None] = {}
    unmatched: dict[str, list[int]] = {}

    for method, df in alarm_data.items():
        alarms = df["observation"].astype(int).tolist()

        candidates = [
            alarm
            for alarm in alarms
            if GRADUAL_START <= alarm <= GRADUAL_END + TOLERANCE
        ]

        match = candidates[0] if candidates else None

        matched[method] = match

        unmatched[method] = [alarm for alarm in alarms if alarm != match]

    fig, ax = plt.subplots(figsize=(15, 6.5))

    ax.plot(
        x,
        series,
        linewidth=0.6,
        alpha=0.20,
        label="Observed synthetic series",
    )

    ax.plot(
        x,
        smooth,
        linewidth=2.0,
        label="15-day mean (visualisation only)",
    )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    # True gradual drift interval
    ax.axvspan(
        GRADUAL_START,
        GRADUAL_END,
        alpha=0.14,
        label="True gradual drift window",
    )

    ax.axvline(
        GRADUAL_START,
        linestyle=":",
        linewidth=1.3,
    )

    ax.axvline(
        GRADUAL_END,
        linestyle=":",
        linewidth=1.3,
    )

    ax.text(
        (GRADUAL_START + GRADUAL_END) / 2,
        ymin + 0.025 * yrange,
        "TRUE GRADUAL DRIFT\n5000–6000",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    # Matched detections
    for detection in matched.values():
        if detection is None:
            continue

        ax.axvline(
            detection,
            linestyle="--",
            linewidth=1.3,
            alpha=0.85,
        )

    matched_lines = ["MATCHED DETECTIONS"]

    for method in [
        "adwin",
        "kswin",
        "page_hinkley",
    ]:
        detection = matched[method]

        if detection is None:
            matched_lines.append(f"{display_name(method)}: no match")
        else:
            delay = detection - GRADUAL_START

            matched_lines.append(
                f"{display_name(method)}: {detection}  ·  delay {delay} obs"
            )

    ax.annotate(
        "\n".join(matched_lines),
        xy=(
            matched["adwin"],
            smooth[matched["adwin"]],
        ),
        xytext=(
            7200,
            ymax - 0.07 * yrange,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "alpha": 0.12,
        },
    )

    summary = (
        "Current configuration — unmatched detections\n"
        f"ADWIN: {len(unmatched['adwin'])}  ·  "
        f"KSWIN: {len(unmatched['kswin'])}  ·  "
        f"Page-Hinkley: {len(unmatched['page_hinkley'])}"
    )

    ax.text(
        0.015,
        0.965,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        style="italic",
        bbox={
            "boxstyle": "round,pad=0.3",
            "alpha": 0.10,
        },
    )

    ax.set_title(
        "F2 — Drift Detection on Synthetic Gradual-Drift Series (Seed 1)",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Synthetic target")

    ax.grid(alpha=0.18)

    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()

    output = FIGURES_DIR / "synthetic_gradual_detection.png"

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote {output}")


# ---------------------------------------------------------------------
# Recurring drift
# ---------------------------------------------------------------------


def plot_recurring(
    runs: pd.DataFrame,
) -> None:
    """Produce the synthetic recurring-drift F2 companion."""

    dataset = "synthetic_recurring"

    alarm_data = load_alarm_data(
        runs,
        dataset,
    )

    series, _ = make_series(
        "recurring",
        n=N,
        noise=NOISE,
        seed=SEED,
    )

    series = np.asarray(series)
    x = np.arange(len(series))

    smooth = smooth_series(
        series,
        SMOOTH_WINDOW,
    )

    matched: dict[str, list[int | None]] = {
        "adwin": [],
        "kswin": [],
        "page_hinkley": [],
    }

    unmatched: dict[str, list[int]] = {}

    for method, df in alarm_data.items():
        alarms = df["observation"].astype(int).tolist()
        used: set[int] = set()

        for cp in RECURRING_CPS:
            candidates = [
                alarm
                for alarm in alarms
                if cp <= alarm <= cp + TOLERANCE and alarm not in used
            ]

            match = candidates[0] if candidates else None

            if match is not None:
                used.add(match)

            matched[method].append(match)

        unmatched[method] = [alarm for alarm in alarms if alarm not in used]

    fig, ax = plt.subplots(figsize=(15, 6.5))

    ax.plot(
        x,
        series,
        linewidth=0.6,
        alpha=0.20,
        label="Observed synthetic series",
    )

    ax.plot(
        x,
        smooth,
        linewidth=2.0,
        label="15-day mean (visualisation only)",
    )

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin

    # True recurring changepoints
    for index, cp in enumerate(RECURRING_CPS):
        ax.axvline(
            cp,
            linewidth=2.0,
            label=("True changepoint" if index == 0 else None),
        )

        ax.text(
            cp - 120,
            ymin + 0.02 * yrange,
            f"TRUE CHANGEPOINT\n{cp}",
            ha="right",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Detection lines
    for detections in matched.values():
        for detection in detections:
            if detection is None:
                continue

            ax.axvline(
                detection,
                linestyle="--",
                linewidth=1.3,
                alpha=0.85,
            )

    # First changepoint
    cp1 = RECURRING_CPS[0]

    cp1_lines = [f"MATCHED DETECTIONS — CP {cp1}"]

    for method in [
        "adwin",
        "kswin",
        "page_hinkley",
    ]:
        detection = matched[method][0]

        if detection is None:
            cp1_lines.append(f"{display_name(method)}: no match")
        else:
            cp1_lines.append(
                f"{display_name(method)}: {detection}  ·  delay {detection - cp1} obs"
            )

    ax.annotate(
        "\n".join(cp1_lines),
        xy=(
            matched["adwin"][0],
            smooth[matched["adwin"][0]],
        ),
        xytext=(
            7600,
            ymax - 0.06 * yrange,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "alpha": 0.12,
        },
    )

    # Second changepoint
    cp2 = RECURRING_CPS[1]

    cp2_lines = [f"MATCHED DETECTIONS — CP {cp2}"]

    for method in [
        "adwin",
        "kswin",
        "page_hinkley",
    ]:
        detection = matched[method][1]

        if detection is None:
            cp2_lines.append(f"{display_name(method)}: no match")
        else:
            cp2_lines.append(
                f"{display_name(method)}: {detection}  ·  delay {detection - cp2} obs"
            )

    ax.annotate(
        "\n".join(cp2_lines),
        xy=(
            matched["adwin"][1],
            smooth[matched["adwin"][1]],
        ),
        xytext=(
            14300,
            ymax - 0.06 * yrange,
        ),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.0,
        },
        fontsize=8.8,
        fontweight="bold",
        ha="left",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.35",
            "alpha": 0.12,
        },
    )

    summary = (
        "Current configuration — unmatched detections\n"
        f"ADWIN: {len(unmatched['adwin'])}  ·  "
        f"KSWIN: {len(unmatched['kswin'])}  ·  "
        f"Page-Hinkley: {len(unmatched['page_hinkley'])}"
    )

    ax.text(
        0.015,
        0.965,
        summary,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        style="italic",
        bbox={
            "boxstyle": "round,pad=0.3",
            "alpha": 0.10,
        },
    )

    ax.set_title(
        "F2 — Drift Detection on Synthetic Recurring-Drift Series (Seed 1)",
        loc="left",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xlabel("Observation")
    ax.set_ylabel("Synthetic target")

    ax.grid(alpha=0.18)

    ax.legend(
        loc="lower right",
        fontsize=8,
        framealpha=0.9,
    )

    fig.tight_layout()

    output = FIGURES_DIR / "synthetic_recurring_detection.png"

    fig.savefig(
        output,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Wrote {output}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    """Produce all synthetic F2 visual companions."""

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs = load_runs()

    plot_sudden(runs)
    plot_gradual(runs)
    plot_recurring(runs)

    print()
    print("Synthetic F2 figures completed.")
    print(
        "Current detector configurations were used. "
        "Regenerate after hyperparameter optimisation."
    )


if __name__ == "__main__":
    main()
