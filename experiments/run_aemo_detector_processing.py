"""Compare raw and daily-aggregated AEMO detector inputs.

This experiment investigates whether reducing high-frequency daily
variation decreases excessive drift-detector alarms on real AEMO demand.

No AEMO ground-truth changepoints are assumed here. Therefore each run
records only n_detections through the shared record_run() harness.

Detector hyperparameters are baseline/frozen settings at this stage.
Parameter tuning is performed separately on synthetic data.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.data.deseasonalise import aggregate_daily_demand
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_frozen_v1"


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

def run_frozen_daily(region: str, stream: pd.Series) -> None:
    """Run synthetic-selected detector configurations on daily AEMO demand."""

    input_name = "daily_mean_complete_days_frozen"

    print(
        f"\n{region} | {input_name} | n={len(stream):,} | "
        f"{stream.index.min()} -> {stream.index.max()}"
    )

    for detector in make_frozen_detectors():
        t0 = time.perf_counter()
        detected = detector.detect(stream)
        wall_clock_s = time.perf_counter() - t0

        config = {
            **config_of(detector),
            "input_processing": "daily_mean_complete_days",
            "parameter_selection": "synthetic_only",
        }

        record_run(
            method=detector.name,
            dataset=f"aemo_{input_name}",
            region=region,
            seed=None,
            config=config,
            wall_clock_s=wall_clock_s,
            split_id=SPLIT_ID,
            train_samples=0,
            detection=(detected, None, len(stream)),
        )

        print(
            f"  {detector.name:12s} "
            f"detections={len(detected):4d} "
            f"runtime={wall_clock_s:.3f}s"
        )


def demand_series(frame: pd.DataFrame) -> pd.Series:
    """Return TOTALDEMAND indexed by settlement timestamp."""
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )




def make_detectors():
    """Return fresh detector instances using the current baseline settings."""
    return (
        ADWINDetector(delta=0.002),
        KSWINDetector(
            alpha=0.005,
            window_size=100,
            stat_size=30,
            seed=42,
        ),
        PageHinkleyDetector(
            min_instances=30,
            delta=0.005,
            threshold=50.0,
        ),
    )


def run_input(
    *,
    region: str,
    input_name: str,
    stream: pd.Series,
) -> None:
    """Run all detectors on one AEMO detector-input representation."""
    print(
        f"\n{region} | {input_name} | "
        f"n={len(stream):,} | "
        f"{stream.index.min()} -> {stream.index.max()}"
    )

    for detector in make_detectors():
        t0 = time.perf_counter()
        detected = detector.detect(stream)
        wall_clock_s = time.perf_counter() - t0

        # Input processing is part of the experimental configuration.
        config = {
            **config_of(detector),
            "input_processing": input_name,
        }

        record_run(
            method=detector.name,
            dataset=f"aemo_{input_name}",
            region=region,
            seed=None,
            config=config,
            wall_clock_s=wall_clock_s,
            split_id=SPLIT_ID,
            train_samples=0,
            detection=(
                detected,
                None,
                len(stream),
            ),
        )

        print(
            f"  {detector.name:12s} "
            f"detections={len(detected):4d} "
            f"runtime={wall_clock_s:.3f}s"
        )


def main() -> None:
    for region in REGIONS:
        _, _, test = loader.load(region)

        raw = demand_series(test)
        daily = aggregate_daily_demand(raw)

        print(
            f"\n{region} | complete daily processing | "
            f"raw_n={len(raw):,} | daily_n={len(daily):,}"
        )

        # Raw and default daily baselines have already been recorded.
        # Run only the synthetic-selected frozen configurations here.
        run_frozen_daily(region, daily)


if __name__ == "__main__":
    main()
