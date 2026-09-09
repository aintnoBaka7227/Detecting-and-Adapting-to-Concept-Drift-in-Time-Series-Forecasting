"""Synthetic-only detector hyperparameter sensitivity experiments.

Each sweep varies exactly one detector parameter while all other
parameters remain at the canonical baseline. Every candidate is
evaluated on the four canonical synthetic drift types and five shared
experiment seeds.

AEMO data are deliberately not used for hyperparameter selection.
"""

from __future__ import annotations

import time

from drift_lab.config import SEEDS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.synthetic.generator import make_series
from experiments.run_harness import config_of, record_run

KINDS = ("none", "sudden", "gradual", "recurring")
N = 20_000
NOISE = 1.0
SPLIT_ID = "synth_n20000"


def detector_sweeps():
    """Yield the original detector sensitivity configurations."""

    # ADWIN baseline delta=0.002.
    for delta in (0.0005, 0.001, 0.002, 0.005, 0.01):
        yield (
            "adwin_delta",
            ADWINDetector(delta=delta),
        )

    # KSWIN baseline:
    # alpha=0.005, window_size=100, stat_size=30, internal seed=42.
    for alpha in (0.001, 0.0025, 0.005, 0.01):
        yield (
            "kswin_alpha",
            KSWINDetector(
                alpha=alpha,
                window_size=100,
                stat_size=30,
                seed=42,
            ),
        )

    for window_size in (100, 200, 300):
        yield (
            "kswin_window_size",
            KSWINDetector(
                alpha=0.005,
                window_size=window_size,
                stat_size=30,
                seed=42,
            ),
        )

    for stat_size in (20, 30, 50):
        yield (
            "kswin_stat_size",
            KSWINDetector(
                alpha=0.005,
                window_size=100,
                stat_size=stat_size,
                seed=42,
            ),
        )

    # Page-Hinkley baseline:
    # min_instances=30, delta=0.005, threshold=50.
    for threshold in (25.0, 50.0, 100.0, 200.0):
        yield (
            "page_hinkley_threshold",
            PageHinkleyDetector(
                min_instances=30,
                delta=0.005,
                threshold=threshold,
            ),
        )

    for delta in (0.001, 0.005, 0.01, 0.05):
        yield (
            "page_hinkley_delta",
            PageHinkleyDetector(
                min_instances=30,
                delta=delta,
                threshold=50.0,
            ),
        )

    for min_instances in (30, 60, 100):
        yield (
            "page_hinkley_min_instances",
            PageHinkleyDetector(
                min_instances=min_instances,
                delta=0.005,
                threshold=50.0,
            ),
        )


def kswin_refinement_sweep():
    """Test KSWIN stat_size values between the previous 30 and 50 settings."""

    for stat_size in (35, 40, 45):
        yield (
            "kswin_stat_size_refinement",
            KSWINDetector(
                alpha=0.005,
                window_size=100,
                stat_size=stat_size,
                seed=42,
            ),
        )


def run_detector(detector, sweep_name: str) -> None:
    """Run one configuration across all canonical synthetic datasets."""

    config = {
        **config_of(detector),
        "tuning_sweep": sweep_name,
    }

    print(f"\n=== {detector.name} | {sweep_name} | {config_of(detector)} ===")

    for kind in KINDS:
        for seed in SEEDS:
            y, changepoints = make_series(
                kind=kind,
                n=N,
                noise=NOISE,
                seed=seed,
            )

            t0 = time.perf_counter()
            detected = detector.detect(y)
            wall_clock_s = time.perf_counter() - t0

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
                    changepoints,
                    len(y),
                ),
            )

            print(
                f"  {kind:9s} seed={seed} "
                f"detections={len(detected):4d} "
                f"runtime={wall_clock_s:.3f}s"
            )


def main() -> None:
    """Run the original full sensitivity experiment."""

    for sweep_name, detector in detector_sweeps():
        run_detector(detector, sweep_name)


def run_kswin_refinement() -> None:
    """Run only the additional KSWIN stat_size refinement."""

    for sweep_name, detector in kswin_refinement_sweep():
        run_detector(detector, sweep_name)


if __name__ == "__main__":
    main()
