"""Evaluate the frozen drift detectors on canonical synthetic streams.

The detector hyperparameters used here were selected using synthetic-only
sensitivity experiments. This experiment does not tune using AEMO data.

Outputs:
- standard detection metrics through record_run()
- detected and true changepoints under results/changepoints/
- false alarms per simulated year
- representative synthetic line plots under results/figures/

One simulated year is defined as:
    48 half-hour observations/day * 365 days = 17,520 observations.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from drift_lab.config import SEEDS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series
from experiments.run_harness import config_of, record_run

KINDS = ("none", "sudden", "gradual", "recurring")

N = 20_000
NOISE = 1.0
SPLIT_ID = "synth_n20000"

SAMPLES_PER_DAY = 48
DAYS_PER_YEAR = 365
SAMPLES_PER_YEAR = SAMPLES_PER_DAY * DAYS_PER_YEAR

FALSE_ALARM_BUDGET_PER_YEAR = 2.0

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CHANGEPOINT_DIR = RESULTS_DIR / "changepoints"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

CHANGEPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def make_frozen_detectors():
    """Return final detector settings selected using synthetic data only."""

    return (
        ADWINDetector(delta=0.00075),
        KSWINDetector(
            alpha=0.005,
            window_size=300,
            stat_size=48,
            seed=42,
        ),
        PageHinkleyDetector(
            min_instances=30,
            delta=0.005,
            threshold=400.0,
        ),
    )

def false_alarms_per_year(
    n_false_alarms: int,
    n_observations: int,
) -> float:
    """Annualise false alarms assuming half-hourly synthetic observations."""

    simulated_years = n_observations / SAMPLES_PER_YEAR

    return float(n_false_alarms / simulated_years)


def save_changepoints(
    *,
    detector_name: str,
    kind: str,
    seed: int,
    series,
    true_changepoints: list[int],
    detected_changepoints: list[int],
    evaluation: dict,
) -> Path:
    """Save true, detected, matched, and false-alarm changepoints."""

    matched_detected = {
        int(pair["detected"])
        for pair in evaluation["matched_pairs"]
    }

    false_alarm_indices = {
        int(index)
        for index in evaluation["false_alarm_indices"]
    }

    rows = []

    for index in true_changepoints:
        rows.append(
            {
                "detector": detector_name,
                "drift_type": kind,
                "seed": seed,
                "point_type": "true",
                "changepoint_index": int(index),
                "value": float(series[index]),
                "matched": None,
            }
        )

    for index in detected_changepoints:
        if index in matched_detected:
            point_type = "detected_matched"
            matched = True
        elif index in false_alarm_indices:
            point_type = "detected_false_alarm"
            matched = False
        else:
            point_type = "detected"
            matched = None

        rows.append(
            {
                "detector": detector_name,
                "drift_type": kind,
                "seed": seed,
                "point_type": point_type,
                "changepoint_index": int(index),
                "value": float(series[index]),
                "matched": matched,
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "detector",
            "drift_type",
            "seed",
            "point_type",
            "changepoint_index",
            "value",
            "matched",
        ],
    )

    output = CHANGEPOINT_DIR / (
        f"synthetic_{kind}_{detector_name}_seed{seed}_changepoints.csv"
    )

    frame.to_csv(output, index=False)

    return output


def plot_synthetic_run(
    *,
    detector_name: str,
    kind: str,
    seed: int,
    series,
    changepoint_path: Path,
    detection_delay: float,
) -> Path:
    """Create a line plot using changepoints read from stored results."""

    frame = pd.read_csv(changepoint_path)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(
        range(len(series)),
        series,
        linewidth=0.8,
        label="Synthetic stream",
    )

    true_points = frame[
        frame["point_type"] == "true"
    ]

    matched_points = frame[
        frame["point_type"] == "detected_matched"
    ]

    false_points = frame[
        frame["point_type"] == "detected_false_alarm"
    ]

    for index in true_points["changepoint_index"]:
        ax.axvline(
            int(index),
            linestyle="--",
            linewidth=1.5,
            label="True changepoint",
        )

    if not matched_points.empty:
        ax.scatter(
            matched_points["changepoint_index"],
            matched_points["value"],
            marker="o",
            s=60,
            label="Matched detection",
            zorder=3,
        )

    if not false_points.empty:
        ax.scatter(
            false_points["changepoint_index"],
            false_points["value"],
            marker="x",
            s=40,
            label="False alarm",
            zorder=3,
        )

    if pd.isna(detection_delay):
        delay_text = "N/A"
    else:
        delay_text = f"{detection_delay:.1f} observations"

    ax.set_title(
        f"{detector_name} | {kind} | seed {seed} | "
        f"detection delay: {delay_text}"
    )
    ax.set_xlabel("Observation")
    ax.set_ylabel("Synthetic value")

    handles, labels = ax.get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    ax.legend(
        unique.values(),
        unique.keys(),
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    output = FIGURE_DIR / (
        f"synthetic_{kind}_{detector_name}_seed{seed}.png"
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


def main() -> None:
    """Run frozen detectors over all canonical synthetic streams."""

    summary_rows = []

    for detector in make_frozen_detectors():

        print(
            f"\n=== {detector.name} | "
            f"{config_of(detector)} ==="
        )

        for kind in KINDS:
            for seed in SEEDS:

                series, true_changepoints = make_series(
                    kind=kind,
                    n=N,
                    noise=NOISE,
                    seed=seed,
                )

                t0 = time.perf_counter()

                detected = detector.detect(series)

                wall_clock_s = (
                    time.perf_counter() - t0
                )

                evaluation = evaluate_detections(
                    detected_changepoints=detected,
                    true_changepoints=true_changepoints,
                    n_observations=len(series),
                    drift_type=kind,
                )

                n_false_alarms = len(
                    evaluation["false_alarm_indices"]
                )

                annual_false_alarms = false_alarms_per_year(
                    n_false_alarms=n_false_alarms,
                    n_observations=len(series),
                )

                budget_met = (
                    annual_false_alarms
                    <= FALSE_ALARM_BUDGET_PER_YEAR
                )

                config = {
                    **config_of(detector),
                    "parameter_selection": "synthetic_only",
                    "evaluation": "frozen_synthetic",
                    "samples_per_year": SAMPLES_PER_YEAR,
                    "false_alarm_budget_per_year": (
                        FALSE_ALARM_BUDGET_PER_YEAR
                    ),
                }

                record_run(
                    method=detector.name,
                    dataset=f"synthetic_{kind}",
                    region=None,
                    seed=seed,
                    config=config,
                    wall_clock_s=wall_clock_s,
                    split_id=SPLIT_ID,
                    detection=(
                        detected,
                        true_changepoints,
                        len(series),
                    ),
                )

                changepoint_path = save_changepoints(
                    detector_name=detector.name,
                    kind=kind,
                    seed=seed,
                    series=series,
                    true_changepoints=true_changepoints,
                    detected_changepoints=detected,
                    evaluation=evaluation,
                )

                summary_rows.append(
                    {
                        "detector": detector.name,
                        "drift_type": kind,
                        "seed": seed,
                        "n_detections": len(detected),
                        "n_false_alarms": n_false_alarms,
                        "false_alarms_per_year": annual_false_alarms,
                        "false_alarm_budget_per_year": (
                            FALSE_ALARM_BUDGET_PER_YEAR
                        ),
                        "false_alarm_budget_met": budget_met,
                        "detection_delay": (
                            evaluation["detection_delay"]
                        ),
                        "missed_detections": (
                            evaluation["missed_detections"]
                        ),
                    }
                )

                print(
                    f"  {kind:9s} "
                    f"seed={seed} "
                    f"detections={len(detected):3d} "
                    f"false/year={annual_false_alarms:.2f} "
                    f"budget={'PASS' if budget_met else 'FAIL'} "
                    f"delay={evaluation['detection_delay']}"
                )

                # One representative plot per detector/drift type.
                # All five seeds remain recorded in the metrics/results.
                if seed == SEEDS[0]:
                    plot_synthetic_run(
                        detector_name=detector.name,
                        kind=kind,
                        seed=seed,
                        series=series,
                        changepoint_path=changepoint_path,
                        detection_delay=(
                            evaluation["detection_delay"]
                        ),
                    )

    summary = pd.DataFrame(summary_rows)

    summary_output = (
        TABLE_DIR / "frozen_synthetic_detector_results.csv"
    )

    summary.to_csv(
        summary_output,
        index=False,
    )

    print(
        f"\nSaved synthetic summary: {summary_output}"
    )

    print(
        f"False-alarm budget: <= "
        f"{FALSE_ALARM_BUDGET_PER_YEAR:.1f} "
        "false alarms per simulated year."
    )

    print(
        f"One simulated year = "
        f"{SAMPLES_PER_YEAR:,} half-hour observations."
    )


if __name__ == "__main__":
    main()