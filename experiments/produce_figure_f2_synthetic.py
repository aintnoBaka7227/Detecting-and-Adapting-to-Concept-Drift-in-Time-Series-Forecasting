from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from drift_lab.synthetic.generator import make_series

# --------------------------------------------------
# Paths
# --------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = REPO_ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"
FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("REPO_ROOT:", REPO_ROOT)
print("RESULTS_DIR:", RESULTS_DIR)
print("RUNS_DIR:", RUNS_DIR)
print("FIGURES_DIR:", FIGURES_DIR)
# --------------------------------------------------
# Fixed representative configuration
# --------------------------------------------------

SEED = 1

CONFIG_HASHES = {
    "ADWIN": "89199180a0a3",
    "KSWIN": "6cce14754b61",
    "Page-Hinkley": "09187f69fb81",
}

SPLIT_IDS = {
    "sudden": "synth_n20000_cp5000",
    "gradual": "synth_n20000_cp5000-6000",
    "recurring": "synth_n20000_cp6666-13333",
}

DETECTOR_Y = {
    "ADWIN": 3,
    "KSWIN": 2,
    "Page-Hinkley": 1,
}


# --------------------------------------------------
# Load persisted alarms
# --------------------------------------------------


def load_alarms(drift_type: str) -> dict[str, list[int]]:
    alarms = {}

    for detector, config_hash in CONFIG_HASHES.items():
        path = (
            RUNS_DIR / config_hash / f"detections_synthetic_{drift_type}_-_{SEED}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(f"Missing persisted detection file: {path}")

        alarms[detector] = pd.read_csv(path)["observation"].astype(int).tolist()

    return alarms


# --------------------------------------------------
# Matching helpers
# --------------------------------------------------


def classify_sudden(
    alarms: list[int],
    true_cp: int,
) -> tuple[int | None, list[int]]:
    after_cp = [a for a in alarms if a >= true_cp]

    matched = after_cp[0] if after_cp else None

    unmatched = [a for a in alarms if a != matched]

    return matched, unmatched


def classify_gradual(
    alarms: list[int],
    start: int,
) -> tuple[int | None, list[int]]:
    after_start = [a for a in alarms if a >= start]

    matched = after_start[0] if after_start else None

    unmatched = [a for a in alarms if a != matched]

    return matched, unmatched


def classify_recurring(
    alarms: list[int],
    true_cps: list[int],
    tolerance: int = 336,
) -> tuple[list[int | None], list[int]]:
    matched = []

    for cp in true_cps:
        candidates = [a for a in alarms if cp <= a <= cp + tolerance]

        matched.append(candidates[0] if candidates else None)

    matched_set = {m for m in matched if m is not None}

    unmatched = [a for a in alarms if a not in matched_set]

    return matched, unmatched


# --------------------------------------------------
# Shared styling
# --------------------------------------------------


def style_detector_axis(ax):
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["Page-Hinkley", "KSWIN", "ADWIN"])

    ax.set_ylim(0.5, 3.7)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Detector")

    ax.text(
        0.01,
        0.03,
        "Vertical ticks = unmatched detections; circles = matched detections",
        transform=ax.transAxes,
        fontsize=9,
    )


# --------------------------------------------------
# Sudden
# --------------------------------------------------


def plot_sudden():
    y, _ = make_series(
        kind="sudden",
        n=20000,
        seed=SEED,
    )

    true_cp = 5000
    alarms = load_alarms("sudden")

    classified = {}

    for detector, detector_alarms in alarms.items():
        matched, unmatched = classify_sudden(
            detector_alarms,
            true_cp,
        )

        classified[detector] = {
            "matched": matched,
            "unmatched": unmatched,
        }

    x = np.arange(len(y))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    ax1.plot(
        x,
        y,
        linewidth=0.7,
        label="Synthetic series",
    )

    ax1.axvline(
        true_cp,
        linestyle="--",
        linewidth=2,
        label="True changepoint = 5000",
    )

    ax1.set_title("Synthetic Sudden Drift Detection — T1 Visual Companion")
    ax1.set_ylabel("Synthetic target value")
    ax1.legend(loc="upper right")

    for detector, info in classified.items():
        y_pos = DETECTOR_Y[detector]

        unmatched = info["unmatched"]

        ax2.scatter(
            unmatched,
            [y_pos] * len(unmatched),
            marker="|",
            s=20,
            alpha=0.35,
        )

        matched = info["matched"]

        if matched is not None:
            ax2.scatter(
                matched,
                y_pos,
                marker="o",
                s=70,
                zorder=5,
            )

            ax2.annotate(
                f"{detector}: matched\n{matched} (+{matched - true_cp})",
                xy=(matched, y_pos),
                xytext=(matched + 500, y_pos + 0.12),
                arrowprops={"arrowstyle": "->"},
                fontsize=9,
            )

    ax2.axvline(
        true_cp,
        linestyle="--",
        linewidth=2,
    )

    style_detector_axis(ax2)

    plt.tight_layout()

    output_path = FIGURES_DIR / "synthetic_sudden_detection.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Gradual
# --------------------------------------------------


def plot_gradual():
    y, _ = make_series(
        kind="gradual",
        n=20000,
        seed=SEED,
    )

    start = 5000
    end = 6000

    alarms = load_alarms("gradual")

    classified = {}

    for detector, detector_alarms in alarms.items():
        matched, unmatched = classify_gradual(
            detector_alarms,
            start,
        )

        classified[detector] = {
            "matched": matched,
            "unmatched": unmatched,
        }

    x = np.arange(len(y))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    ax1.plot(
        x,
        y,
        linewidth=0.7,
        label="Synthetic series",
    )

    ax1.axvspan(
        start,
        end,
        alpha=0.18,
        label="True gradual drift interval (5000–6000)",
    )

    ax1.set_title("Synthetic Gradual Drift Detection — T1 Visual Companion")
    ax1.set_ylabel("Synthetic target value")
    ax1.legend(loc="upper right")

    for detector, info in classified.items():
        y_pos = DETECTOR_Y[detector]

        unmatched = info["unmatched"]

        ax2.scatter(
            unmatched,
            [y_pos] * len(unmatched),
            marker="|",
            s=20,
            alpha=0.35,
        )

        matched = info["matched"]

        if matched is not None:
            ax2.scatter(
                matched,
                y_pos,
                marker="o",
                s=70,
                zorder=5,
            )

            ax2.annotate(
                f"{detector}: matched\n{matched} (+{matched - start})",
                xy=(matched, y_pos),
                xytext=(matched + 500, y_pos + 0.12),
                arrowprops={"arrowstyle": "->"},
                fontsize=9,
            )

    ax2.axvspan(
        start,
        end,
        alpha=0.18,
    )

    style_detector_axis(ax2)

    plt.tight_layout()

    output_path = FIGURES_DIR / "synthetic_gradual_detection.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Recurring
# --------------------------------------------------


def plot_recurring():
    y, _ = make_series(
        kind="recurring",
        n=20000,
        seed=SEED,
    )

    true_cps = [6666, 13333]

    alarms = load_alarms("recurring")

    classified = {}

    for detector, detector_alarms in alarms.items():
        matched, unmatched = classify_recurring(
            detector_alarms,
            true_cps,
        )

        classified[detector] = {
            "matched": matched,
            "unmatched": unmatched,
        }

    x = np.arange(len(y))

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    ax1.plot(
        x,
        y,
        linewidth=0.7,
        label="Synthetic series",
    )

    for i, cp in enumerate(true_cps):
        ax1.axvline(
            cp,
            linestyle="--",
            linewidth=2,
            label="True changepoint" if i == 0 else None,
        )

    ax1.set_title("Synthetic Recurring Drift Detection — T1 Visual Companion")
    ax1.set_ylabel("Synthetic target value")
    ax1.legend(loc="upper right")

    for detector, info in classified.items():
        y_pos = DETECTOR_Y[detector]

        unmatched = info["unmatched"]

        ax2.scatter(
            unmatched,
            [y_pos] * len(unmatched),
            marker="|",
            s=20,
            alpha=0.35,
        )

        for matched, cp in zip(
            info["matched"],
            true_cps,
        ):
            if matched is None:
                continue

            delay = matched - cp

            ax2.scatter(
                matched,
                y_pos,
                marker="o",
                s=70,
                zorder=5,
            )

            ax2.annotate(
                f"{matched} (+{delay})",
                xy=(matched, y_pos),
                xytext=(matched + 350, y_pos + 0.12),
                arrowprops={"arrowstyle": "->"},
                fontsize=9,
            )

    for cp in true_cps:
        ax2.axvline(
            cp,
            linestyle="--",
            linewidth=2,
        )

    style_detector_axis(ax2)

    plt.tight_layout()

    output_path = FIGURES_DIR / "synthetic_recurring_detection.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------


def main():
    print("Synthetic T1 visual companion")
    print(f"Seed: {SEED}")
    print("Config hashes:")

    for detector, config_hash in CONFIG_HASHES.items():
        print(f"  {detector}: {config_hash}")

    print("Split IDs:")

    for drift_type, split_id in SPLIT_IDS.items():
        print(f"  {drift_type}: {split_id}")

    plot_sudden()
    plot_gradual()
    plot_recurring()


if __name__ == "__main__":
    main()
