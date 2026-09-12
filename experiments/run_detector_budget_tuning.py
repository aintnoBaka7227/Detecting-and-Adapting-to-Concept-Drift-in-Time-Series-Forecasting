"""Final synthetic-only detector tuning against the false-alarm budget.

Acceptance criteria:
1. <= 2 false alarms per simulated year.
2. No missed known drift events.

Only one detector parameter is varied within each sweep.
AEMO data and AEMO events are not used for tuning.
"""

from __future__ import annotations

import time
from pathlib import Path

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

SAMPLES_PER_YEAR = 48 * 365
FALSE_ALARM_BUDGET_PER_YEAR = 2.0

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    REPO_ROOT
    / "results"
    / "tables"
    / "detector_budget_tuning.csv"
)


def false_alarms_per_year(
    n_false_alarms: int,
    n_observations: int,
) -> float:
    simulated_years = n_observations / SAMPLES_PER_YEAR
    return float(n_false_alarms / simulated_years)


def candidate_sweeps():
    """Yield targeted final sensitivity candidates."""

    # ADWIN: move slightly more conservative than delta=0.001.
    for delta in (0.00025, 0.0005, 0.00075, 0.001):
        yield (
            "adwin_budget_delta",
            ADWINDetector(delta=delta),
        )

        # KSWIN final refinement:
    # vary only stat_size while keeping alpha, window and seed fixed.
    for stat_size in (35, 40, 42, 44, 45, 46, 47, 48, 50):
        yield (
            "kswin_budget_stat_size",
            KSWINDetector(
                alpha=0.005,
                window_size=300,
                stat_size=stat_size,
                seed=42,
            ),
        )

    # Page-Hinkley: vary only threshold.
    for threshold in (200.0, 300.0, 400.0, 500.0, 750.0, 1000.0):
        yield (
            "page_hinkley_budget_threshold",
            PageHinkleyDetector(
                min_instances=30,
                delta=0.005,
                threshold=threshold,
            ),
        )


def main() -> None:
    rows = []

    for sweep_name, detector in candidate_sweeps():

        detector_config = config_of(detector)

        print(
            f"\n=== {detector.name} | "
            f"{sweep_name} | {detector_config} ==="
        )

        for kind in KINDS:
            for seed in SEEDS:

                series, truth = make_series(
                    kind=kind,
                    n=N,
                    noise=NOISE,
                    seed=seed,
                )

                t0 = time.perf_counter()
                detected = detector.detect(series)
                wall_clock_s = time.perf_counter() - t0

                evaluation = evaluate_detections(
                    detected_changepoints=detected,
                    true_changepoints=truth,
                    n_observations=len(series),
                    drift_type=kind,
                )

                n_false = len(
                    evaluation["false_alarm_indices"]
                )

                fa_year = false_alarms_per_year(
                    n_false,
                    len(series),
                )

                budget_met = (
                    fa_year
                    <= FALSE_ALARM_BUDGET_PER_YEAR
                )

                no_misses = (
                    evaluation["missed_detections"] == 0
                )

                accepted = budget_met and no_misses

                config = {
                    **detector_config,
                    "tuning_sweep": sweep_name,
                    "parameter_selection": "synthetic_only",
                    "false_alarm_budget_per_year": (
                        FALSE_ALARM_BUDGET_PER_YEAR
                    ),
                    "samples_per_year": SAMPLES_PER_YEAR,
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
                        truth,
                        len(series),
                    ),
                )

                rows.append(
                    {
                        "method": detector.name,
                        "sweep": sweep_name,
                        "config": str(detector_config),
                        "drift_type": kind,
                        "seed": seed,
                        "n_detections": len(detected),
                        "n_false_alarms": n_false,
                        "false_alarms_per_year": fa_year,
                        "budget_met": budget_met,
                        "detection_delay": (
                            evaluation["detection_delay"]
                        ),
                        "missed_detections": (
                            evaluation["missed_detections"]
                        ),
                        "accepted": accepted,
                    }
                )

                print(
                    f"  {kind:9s} "
                    f"seed={seed} "
                    f"det={len(detected):3d} "
                    f"FA/year={fa_year:6.2f} "
                    f"miss={evaluation['missed_detections']} "
                    f"{'PASS' if accepted else 'FAIL'}"
                )

    frame = pd.DataFrame(rows)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        OUTPUT,
        index=False,
    )

    print(f"\nSaved: {OUTPUT}")

    print("\n=== CONFIGURATION ACCEPTANCE SUMMARY ===")

    summary = (
        frame.groupby(
            ["method", "sweep", "config"],
            dropna=False,
        )
        .agg(
            worst_false_alarms_per_year=(
                "false_alarms_per_year",
                "max",
            ),
            total_missed_detections=(
                "missed_detections",
                "sum",
            ),
            all_runs_accepted=(
                "accepted",
                "all",
            ),
            mean_detection_delay=(
                "detection_delay",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False,
        )
    )


if __name__ == "__main__":
    main()