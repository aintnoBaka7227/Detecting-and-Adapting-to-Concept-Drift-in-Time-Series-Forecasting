"""Evaluate the frozen drift detectors on canonical synthetic streams.

The detector hyperparameters used here were selected using synthetic-only
sensitivity experiments (see run_fine_tune_on_synthetic.py -- one frozen
config per detector, see post_tune_detector_configs.py). This experiment
does not tune using AEMO data.

Outputs:
- standard detection metrics through record_run() -- appended to
  results/runs.csv under split_id="synthetic_full_series_20000_observations",
  the same way run_pre_tune_on_synthetic.py logs its (pre-tuning) rows. No
  table is written directly by this script.
- detected changepoint indices under results/runs/<config_hash>/, via
  results_io.dump_synthetic_detections() -- the same config_hash-keyed
  convention run_pre_tune_on_synthetic.py and the AEMO detection scripts
  use, not a separate top-level directory. True changepoints and the
  matched/false-alarm split aren't persisted; they're deterministic from
  (kind, seed) and cheap to recompute, so a plotting script regenerates
  them with make_series()/evaluate_detections() instead of reading them
  back from a file.

This script does not plot anything itself; produce_figure_f2_synthetic_post_tune.py
reads its persisted detections afterwards to build the F2 companion figures
(one per drift type, one panel per seed, all three detectors together) --
an earlier per-(detector, drift_type) single-seed plot generated inline
here was retired once that composite figure covered the same ground more
completely.
"""

from __future__ import annotations

import time

from drift_lab.config import SEEDS
from drift_lab.evaluation.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series
from experiments import results_io
from experiments.run.detection.post_tune_detector_configs import make_post_tune_detectors
from experiments.run_harness import config_of, record_run

KINDS = ("none", "sudden", "gradual", "recurring")

N = 20_000
NOISE = 1.0

SAMPLES_PER_YEAR = 48 * 365

FALSE_ALARM_BUDGET_PER_YEAR = 2.0

SPLIT_ID = "synthetic_full_series_20000_observations"


def false_alarms_per_year(n_false_alarms: int, n_observations: int) -> float:
    simulated_years = n_observations / SAMPLES_PER_YEAR
    return float(n_false_alarms / simulated_years)


def main() -> None:
    """Run frozen detectors over all canonical synthetic streams."""
    for detector in make_post_tune_detectors():

        print(f"\n=== {detector.name} | {config_of(detector)} ===")

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

                wall_clock_s = time.perf_counter() - t0

                evaluation = evaluate_detections(
                    detected_changepoints=detected,
                    true_changepoints=true_changepoints,
                    n_observations=len(series),
                    drift_type=kind,
                )

                n_false_alarms = len(evaluation["false_alarm_indices"])

                annual_false_alarms = false_alarms_per_year(
                    n_false_alarms=n_false_alarms,
                    n_observations=len(series),
                )

                budget_met = annual_false_alarms <= FALSE_ALARM_BUDGET_PER_YEAR

                config = {
                    **config_of(detector),
                    "parameter_selection": "synthetic_only",
                    "evaluation": "frozen_synthetic",
                    "samples_per_year": SAMPLES_PER_YEAR,
                    "false_alarm_budget_per_year": FALSE_ALARM_BUDGET_PER_YEAR,
                }

                record_run(
                    method=detector.name,
                    dataset=f"synthetic_{kind}",
                    region=None,
                    seed=seed,
                    config=config,
                    wall_clock_s=wall_clock_s,
                    split_id=SPLIT_ID,
                    detection=(detected, true_changepoints, len(series)),
                    samples_per_year=SAMPLES_PER_YEAR,
                )

                results_io.dump_synthetic_detections(
                    config_hash_=results_io.config_hash(config),
                    dataset=f"synthetic_{kind}",
                    seed=seed,
                    detected_indices=detected,
                )

                print(
                    f"  {kind:9s} "
                    f"seed={seed} "
                    f"detections={len(detected):3d} "
                    f"false/year={annual_false_alarms:.2f} "
                    f"budget={'PASS' if budget_met else 'FAIL'} "
                    f"delay={evaluation['detection_delay']}"
                )

    print(
        f"\nfalse-alarm budget <= {FALSE_ALARM_BUDGET_PER_YEAR:.1f} per "
        f"simulated year; one simulated year = {SAMPLES_PER_YEAR:,} observations."
    )


if __name__ == "__main__":
    main()
