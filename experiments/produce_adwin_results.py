"""Produce ADWIN summaries from already-recorded experiment results.

This module is read-only with respect to detector execution. It does not run
ADWIN, generate synthetic data, or recompute shared detection metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RUNS_CSV = Path("results/runs.csv")
OUTPUT_DIR = Path("results/tables")
OUTPUT_CSV = OUTPUT_DIR / "adwin_results.csv"

METRICS = (
    "detection_delay",
    "false_alarms_per_10000",
    "missed_detections",
)


def load_adwin_runs() -> pd.DataFrame:
    """Load recorded ADWIN synthetic detection rows."""
    runs = pd.read_csv(RUNS_CSV)

    adwin = runs[
        (runs["group"] == "detection")
        & (runs["method"] == "adwin")
        & (runs["dataset"].str.startswith("synthetic_"))
    ].copy()

    if adwin.empty:
        raise RuntimeError(
            "No recorded ADWIN synthetic runs were found in results/runs.csv."
        )

    adwin["timestamp"] = pd.to_datetime(adwin["timestamp"], utc=True)
    adwin["metric_value"] = pd.to_numeric(adwin["metric_value"], errors="coerce")

    return adwin


def keep_latest_per_config(adwin: pd.DataFrame) -> pd.DataFrame:
    """Keep the latest recorded run for each config/dataset/seed combination."""
    run_keys = [
        "config_hash",
        "dataset",
        "seed",
        "split_id",
    ]

    latest_timestamps = (
        adwin.groupby(run_keys, dropna=False)["timestamp"]
        .max()
        .reset_index(name="latest_timestamp")
    )

    latest = adwin.merge(
        latest_timestamps,
        on=run_keys,
        how="inner",
    )

    latest = latest[latest["timestamp"] == latest["latest_timestamp"]].copy()

    return latest.drop(columns="latest_timestamp")


def produce_summary(latest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the recorded ADWIN metrics across experiment seeds."""
    metrics = latest[latest["metric_name"].isin(METRICS)].copy()

    metric_summary = (
        metrics.groupby(
            ["config_hash", "dataset", "metric_name"],
            dropna=False,
        )["metric_value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    metric_summary["result"] = metric_summary.apply(
        lambda row: (
            f"{row['mean']:.3f}"
            if pd.isna(row["std"])
            else f"{row['mean']:.3f} ± {row['std']:.3f}"
        ),
        axis=1,
    )

    metric_table = metric_summary.pivot(
        index=["config_hash", "dataset"],
        columns="metric_name",
        values="result",
    ).reset_index()

    runtime_summary = (
        latest.groupby(
            ["config_hash", "dataset"],
            dropna=False,
        )["wall_clock_s"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(
            columns={
                "mean": "runtime_mean_s",
                "std": "runtime_std_s",
            }
        )
    )

    summary = metric_table.merge(
        runtime_summary,
        on=["config_hash", "dataset"],
        how="left",
    )

    dataset_order = {
        "synthetic_none": 0,
        "synthetic_sudden": 1,
        "synthetic_gradual": 2,
        "synthetic_recurring": 3,
    }

    summary["_order"] = summary["dataset"].map(dataset_order)
    summary = summary.sort_values(["config_hash", "_order"]).drop(columns="_order")

    return summary


def main() -> None:
    latest = keep_latest_per_config(load_adwin_runs())
    summary = produce_summary(latest)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print("\nADWIN recorded-results summary")
    print(summary.to_string(index=False))
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
