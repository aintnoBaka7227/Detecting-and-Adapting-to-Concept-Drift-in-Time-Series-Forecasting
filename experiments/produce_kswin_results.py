"""Produce KSWIN summaries from already-recorded experiment results.

This module reads recorded results only. It does not run KSWIN, generate
synthetic data, or recompute shared detection metrics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RUNS_CSV = Path("results/runs.csv")
OUTPUT_DIR = Path("results/tables")
OUTPUT_CSV = OUTPUT_DIR / "kswin_results.csv"

METRICS = (
    "detection_delay",
    "false_alarms_per_10000",
    "missed_detections",
)


def load_kswin_runs() -> pd.DataFrame:
    """Load recorded KSWIN synthetic detection rows."""
    runs = pd.read_csv(RUNS_CSV)

    kswin = runs[
        (runs["group"] == "detection")
        & (runs["method"] == "kswin")
        & (runs["dataset"].str.startswith("synthetic_"))
    ].copy()

    if kswin.empty:
        raise RuntimeError(
            "No recorded KSWIN synthetic runs were found in results/runs.csv."
        )

    kswin["timestamp"] = pd.to_datetime(kswin["timestamp"], utc=True)
    kswin["metric_value"] = pd.to_numeric(kswin["metric_value"], errors="coerce")

    return kswin


def keep_latest_per_config(kswin: pd.DataFrame) -> pd.DataFrame:
    """Keep latest recorded run for each config/dataset/seed combination."""
    run_keys = [
        "config_hash",
        "dataset",
        "seed",
        "split_id",
    ]

    latest_timestamps = (
        kswin.groupby(run_keys, dropna=False)["timestamp"]
        .max()
        .reset_index(name="latest_timestamp")
    )

    latest = kswin.merge(
        latest_timestamps,
        on=run_keys,
        how="inner",
    )

    latest = latest[latest["timestamp"] == latest["latest_timestamp"]].copy()

    return latest.drop(columns="latest_timestamp")


def produce_summary(latest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate recorded KSWIN metrics across experiment seeds."""
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

    return summary.sort_values(["config_hash", "_order"]).drop(columns="_order")


def main() -> None:
    latest = keep_latest_per_config(load_kswin_runs())
    summary = produce_summary(latest)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_CSV, index=False)

    print("\nKSWIN recorded-results summary")
    print(summary.to_string(index=False))
    print(f"\nWrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
