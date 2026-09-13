"""Compare AEMO detector input-processing techniques.

The experiment compares three representations of already-cleaned AEMO demand:

1. Raw half-hourly test demand.
2. Complete-day daily mean demand.
3. Half-hourly demand after removing the historical weekday/half-hour profile.

The seasonal profile is estimated from historical pre-test data. Only the
processed test stream is supplied to detector.detect().

Detector parameters are frozen from synthetic-only tuning. AEMO events are
not used for hyperparameter selection.

All AEMO documented-event evaluation and runs.csv logging is performed through
the team's shared record_run() harness. The harness calls
evaluation.match_detections_to_events(), persists detection timestamps through
results_io, and appends the resulting metric rows to results/runs.csv.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS
from drift_lab.data.deseasonalise import (
    aggregate_daily_demand,
    remove_daily_weekly_profile,
)
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import (
    DOCUMENTED_PERIOD_GRACE,
    DOCUMENTED_POINT_TOLERANCE,
)
from experiments import results_io
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_detector_input_v1"

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
FIGURE_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

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


def tier1_events(region: str) -> pd.DataFrame:
    """Return frozen Tier 1 documented events relevant to a region."""
    events = pd.read_csv(
        DOCUMENTED_EVENTS_CSV,
        parse_dates=["start_date", "end_date"],
    )

    return events[
        (events["tier"] == 1)
        & events["region"].isin([region, "NEM"])
    ].reset_index(drop=True)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    """Return TOTALDEMAND indexed by settlement timestamp."""
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    ).sort_index()


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


def metric_from_logged(
    logged: pd.DataFrame,
    metric_name: str,
) -> float:
    """Extract one scalar metric returned by the shared run harness."""
    rows = logged.loc[logged["metric_name"] == metric_name, "metric_value"]

    if rows.empty:
        return float("nan")

    return float(rows.iloc[0])


def run_input(
    region: str,
    input_name: str,
    stream: pd.Series,
    events: pd.DataFrame,
) -> tuple[dict[str, Path], list[dict]]:
    """Run every frozen detector on one processed AEMO test stream."""
    print(
        f"\n{region} | {input_name} | "
        f"n={len(stream):,} | "
        f"{stream.index.min()} -> {stream.index.max()}"
    )

    detection_files: dict[str, Path] = {}
    summary_rows: list[dict] = []

    for detector in make_frozen_detectors():
        t0 = time.perf_counter()

        flagged_indices = detector.detect(stream.to_numpy())

        wall_clock_s = time.perf_counter() - t0

        detected_timestamps = list(stream.index[flagged_indices])

        config = {
            **config_of(detector),
            "input_processing": input_name,
            "parameter_selection": "synthetic_only_budget",
            "false_alarm_budget_per_simulated_year": 2.0,
            "experiment_runner": "run_detector_input_experiments",
            "match_point_tolerance_days": (
                DOCUMENTED_POINT_TOLERANCE.days
            ),
            "match_period_grace_days": (
                DOCUMENTED_PERIOD_GRACE.days
            ),
        }

        # Shared team pipeline:
        #
        # record_run()
        #   -> evaluation.match_detections_to_events()
        #   -> results_io.dump_detections()
        #   -> results_io.append_runs()
        #
        # Therefore this experiment does not calculate or append the
        # documented-event metrics independently.
        logged = record_run(
            method=detector.name,
            dataset="aemo",
            region=region,
            seed=None,
            config=config,
            wall_clock_s=wall_clock_s,
            split_id=SPLIT_ID,
            train_samples=0,
            detection=(
                detected_timestamps,
                events,
                None,
            ),
        )

        config_hash = str(logged["config_hash"].iloc[0])

        # record_run() has already persisted these detections through the
        # shared results_io layer. We only derive the shared output path here
        # so the plotting function can read the same stored artifact.
        detection_path = results_io.detections_path(
            config_hash,
            "aemo",
            region,
            None,
        )

        detection_files[detector.name] = detection_path

        cluster = detection_clustering(flagged_indices)

        summary_rows.append(
            {
                "region": region,
                "input_processing": input_name,
                "detector": detector.name,
                "config_hash": config_hash,
                "n_observations": len(stream),
                "n_detections": metric_from_logged(
                    logged,
                    "n_detections",
                ),
                "n_matched": metric_from_logged(
                    logged,
                    "n_matched",
                ),
                "n_unmatched_events": metric_from_logged(
                    logged,
                    "n_unmatched_events",
                ),
                "n_unmatched_detections": metric_from_logged(
                    logged,
                    "n_unmatched_detections",
                ),
                "precision": metric_from_logged(
                    logged,
                    "precision",
                ),
                "mean_delay_days": metric_from_logged(
                    logged,
                    "mean_delay_days",
                ),
                "detections_per_1000_observations": (
                    len(flagged_indices) / len(stream) * 1000
                    if len(stream)
                    else float("nan")
                ),
                "median_detection_gap": cluster["median_gap"],
                "minimum_detection_gap": cluster["min_gap"],
                "wall_clock_s": wall_clock_s,
            }
        )

        print(
            f"  {detector.name:12s} "
            f"detections={len(flagged_indices):4d} "
            f"matched={metric_from_logged(logged, 'n_matched'):.0f} "
            f"precision={metric_from_logged(logged, 'precision'):.3f} "
            f"mean_delay_days="
            f"{metric_from_logged(logged, 'mean_delay_days'):.3f} "
            f"runtime={wall_clock_s:.3f}s "
            f"config={config_hash}"
        )

    return detection_files, summary_rows


def read_detections(path: Path) -> pd.DataFrame:
    """Read detector timestamps saved by the shared results_io layer."""
    return pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )


def plot_input(
    *,
    region: str,
    input_name: str,
    stream: pd.Series,
    detection_files: dict[str, Path],
) -> Path:
    """Plot AEMO input using detections stored by the shared results system."""
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

    for detector_name, path in detection_files.items():
        frame = read_detections(path)

        if frame.empty:
            continue

        timestamps = pd.DatetimeIndex(frame["timestamp"])

        values = stream.reindex(timestamps)

        valid = values.notna()

        ax.scatter(
            timestamps[valid],
            values[valid],
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

        # Frozen AEMO test interval.
        test_series = test_series.loc[
            "2020-03-01":"2023-12-31 23:59:59"
        ]

        # Historical data is used only to estimate the seasonal profile.
        # It is never supplied to detector.detect().
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

        # Use the same frozen documented-event catalogue and regional
        # filtering rule as the team's canonical AEMO detector runner.
        events = tier1_events(region)

        print(
            f"\n{region}: evaluating against "
            f"{len(events)} Tier 1 documented events"
        )

        for input_name, stream in inputs.items():
            detection_files, rows = run_input(
                region=region,
                input_name=input_name,
                stream=stream,
                events=events,
            )

            summary_rows.extend(rows)

            figure = plot_input(
                region=region,
                input_name=input_name,
                stream=stream,
                detection_files=detection_files,
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
                "config_hash",
                "n_observations",
                "n_detections",
                "n_matched",
                "n_unmatched_events",
                "n_unmatched_detections",
                "precision",
                "mean_delay_days",
                "detections_per_1000_observations",
                "median_detection_gap",
            ]
        ].to_string(index=False)
    )

    print(f"\nSaved summary: {output}")


if __name__ == "__main__":
    main()
