"""Compare AEMO detector input-processing techniques.

The experiment compares three representations of already-cleaned AEMO demand:

1. Raw half-hourly test demand.
2. Complete-day daily mean demand.
3. Half-hourly demand after removing the historical weekday/half-hour profile.

The seasonal profile is estimated from historical pre-test data. Only the
processed test stream is supplied to detector.detect().

Detector parameters are frozen from synthetic-only tuning. AEMO events are
not used for hyperparameter selection.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.data.deseasonalise import (
    aggregate_daily_demand,
    remove_daily_weekly_profile,
)
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from experiments.run_harness import config_of, record_run


SPLIT_ID = "aemo_frozen_v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
CHANGEPOINT_DIR = RESULTS_DIR / "changepoints"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

CHANGEPOINT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def make_frozen_detectors():
    """Return final synthetic-selected detector configurations."""

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


def demand_series(frame: pd.DataFrame) -> pd.Series:
    """Return TOTALDEMAND indexed by settlement timestamp."""

    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def save_changepoints(
    *,
    region: str,
    detector_name: str,
    input_name: str,
    stream: pd.Series,
    detections: list[int],
) -> Path:
    """Save every detected changepoint index and timestamp."""

    rows = []

    for detection_number, index in enumerate(detections, start=1):
        if index < 0 or index >= len(stream):
            raise ValueError(
                f"Invalid changepoint index {index} "
                f"for stream of length {len(stream)}."
            )

        rows.append(
            {
                "region": region,
                "detector": detector_name,
                "input_processing": input_name,
                "detection_number": detection_number,
                "changepoint_index": index,
                "changepoint_timestamp": stream.index[index],
                "demand_value": float(stream.iloc[index]),
            }
        )

    frame = pd.DataFrame(
        rows,
        columns=[
            "region",
            "detector",
            "input_processing",
            "detection_number",
            "changepoint_index",
            "changepoint_timestamp",
            "demand_value",
        ],
    )

    output = CHANGEPOINT_DIR / (
        f"{region}_{detector_name}_{input_name}_changepoints.csv"
    )

    frame.to_csv(output, index=False)

    return output


def detection_clustering(detections: list[int]) -> dict[str, float]:
    """Summarise how closely detector alarms occur."""

    if len(detections) < 2:
        return {
            "median_gap": float("nan"),
            "min_gap": float("nan"),
        }

    gaps = pd.Series(detections).diff().dropna()

    return {
        "median_gap": float(gaps.median()),
        "min_gap": float(gaps.min()),
    }


def run_input(
    *,
    region: str,
    input_name: str,
    stream: pd.Series,
) -> tuple[dict[str, Path], list[dict]]:
    """Run every frozen detector on one AEMO test representation."""

    print(
        f"\n{region} | {input_name} | "
        f"n={len(stream):,} | "
        f"{stream.index.min()} -> {stream.index.max()}"
    )

    changepoint_files: dict[str, Path] = {}
    rows: list[dict] = []

    for detector in make_frozen_detectors():
        t0 = time.perf_counter()

        detections = detector.detect(stream)

        wall_clock_s = time.perf_counter() - t0

        config = {
            **config_of(detector),
            "input_processing": input_name,
            "parameter_selection": "synthetic_only_budget",
            "false_alarm_budget_per_simulated_year": 2.0,
            "experiment_runner": "run_detector_input_experiments",
        }

        record_run(
            method=detector.name,
            dataset=f"aemo_{input_name}_final",
            region=region,
            seed=None,
            config=config,
            wall_clock_s=wall_clock_s,
            split_id=SPLIT_ID,
            train_samples=0,
            detection=(
                detections,
                None,
                len(stream),
            ),
        )

        path = save_changepoints(
            region=region,
            detector_name=detector.name,
            input_name=input_name,
            stream=stream,
            detections=detections,
        )

        changepoint_files[detector.name] = path

        cluster = detection_clustering(detections)

        rows.append(
            {
                "region": region,
                "input_processing": input_name,
                "detector": detector.name,
                "n_observations": len(stream),
                "n_detections": len(detections),
                "detections_per_1000_observations": (
                    len(detections) / len(stream) * 1000
                ),
                "median_detection_gap": cluster["median_gap"],
                "minimum_detection_gap": cluster["min_gap"],
                "wall_clock_s": wall_clock_s,
            }
        )

        print(
            f"  {detector.name:12s} "
            f"detections={len(detections):4d} "
            f"runtime={wall_clock_s:.3f}s"
        )

    return changepoint_files, rows


def read_changepoints(path: Path) -> pd.DataFrame:
    """Read stored changepoints for reproducible plotting."""

    return pd.read_csv(
        path,
        parse_dates=["changepoint_timestamp"],
    )


def plot_input(
    *,
    region: str,
    input_name: str,
    stream: pd.Series,
    changepoint_files: dict[str, Path],
) -> Path:
    """Plot an AEMO test stream with all frozen detector changepoints."""

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.plot(
        stream.index,
        stream.values,
        linewidth=0.8,
        label="AEMO demand input",
    )

    markers = {
        "adwin": "o",
        "kswin": "s",
        "page_hinkley": "^",
    }

    for detector_name, path in changepoint_files.items():
        frame = read_changepoints(path)

        if frame.empty:
            continue

        ax.scatter(
            frame["changepoint_timestamp"],
            frame["demand_value"],
            marker=markers.get(detector_name, "o"),
            s=35,
            label=f"{detector_name} ({len(frame)})",
            zorder=3,
        )

    ax.set_title(
        f"{region} AEMO test demand — {input_name}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("TOTALDEMAND / processed demand")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    output = FIGURE_DIR / (
        f"{region}_{input_name}_final_changepoints.png"
    )

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output


def main() -> None:
    """Run final AEMO detector-input comparison."""

    summary_rows: list[dict] = []

    for region in REGIONS:
        train, calibration, test = loader.load(region)

        train_series = demand_series(train)
        calibration_series = demand_series(calibration)
        test_series = demand_series(test)

        # Restrict the detector target explicitly to the required test period.
        test_series = test_series.loc[
            "2020-03-01":"2023-12-31 23:59:59"
        ]

        # Historical reference only. It is never passed to detector.detect().
        seasonal_reference = pd.concat(
            [train_series, calibration_series]
        ).sort_index()

        raw_test = test_series.copy()

        daily_test = aggregate_daily_demand(
            test_series
        )

        seasonal_test = remove_daily_weekly_profile(
            test_series,
            reference=seasonal_reference,
        )

        inputs = {
            "raw_30min": raw_test,
            "daily_mean_complete_days": daily_test,
            "daily_weekly_adjusted": seasonal_test,
        }

        for input_name, stream in inputs.items():
            changepoint_files, rows = run_input(
                region=region,
                input_name=input_name,
                stream=stream,
            )

            summary_rows.extend(rows)

            figure = plot_input(
                region=region,
                input_name=input_name,
                stream=stream,
                changepoint_files=changepoint_files,
            )

            print(f"  figure saved: {figure}")

    summary = pd.DataFrame(summary_rows)

    output = TABLE_DIR / "aemo_detector_input_comparison.csv"

    summary.to_csv(
        output,
        index=False,
    )

    print("\n=== FINAL AEMO INPUT COMPARISON ===")

    print(
        summary[
            [
                "region",
                "input_processing",
                "detector",
                "n_observations",
                "n_detections",
                "detections_per_1000_observations",
                "median_detection_gap",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved summary: {output}")


if __name__ == "__main__":
    main()