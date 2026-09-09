"""Summarise recorded synthetic detector tuning experiments.

This script is read-only with respect to experiment results. It reads
runs.csv and the config.json files produced by record_run(), then
aggregates the shared detection metrics across seeds.

It does not rerun detectors or recompute detection metrics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path("results")
RUNS_CSV = RESULTS_DIR / "runs.csv"
RUNS_DIR = RESULTS_DIR / "runs"
SPLIT_ID = "synth_n20000"

METRICS = (
    "detection_delay",
    "false_alarms_per_10000",
    "missed_detections",
)


def load_config(config_hash: str) -> dict:
    """Load the raw configuration associated with a config hash."""
    path = RUNS_DIR / config_hash / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    return json.loads(path.read_text())


def config_label(config: dict) -> str:
    """Create a compact human-readable detector configuration label."""
    excluded = {"tuning_sweep"}
    parts = [
        f"{key}={value}" for key, value in sorted(config.items()) if key not in excluded
    ]
    return ", ".join(parts)


def main() -> None:
    runs = pd.read_csv(RUNS_CSV)

    tuning = runs[
        (runs["group"] == "detection")
        & (runs["split_id"] == SPLIT_ID)
        & (runs["metric_name"].isin(METRICS))
        & (runs["dataset"].str.startswith("synthetic_"))
    ].copy()

    if tuning.empty:
        raise RuntimeError("No synthetic tuning results found.")

    hashes = tuning["config_hash"].dropna().unique()

    configs = {}
    for config_hash in hashes:
        config = load_config(str(config_hash))
        if "tuning_sweep" in config:
            configs[str(config_hash)] = config

    tuning = tuning[tuning["config_hash"].astype(str).isin(configs)].copy()

    tuning["sweep"] = (
        tuning["config_hash"].astype(str).map(lambda h: configs[h]["tuning_sweep"])
    )
    tuning["config"] = (
        tuning["config_hash"].astype(str).map(lambda h: config_label(configs[h]))
    )

    # A baseline configuration can appear in more than one one-at-a-time
    # sweep. Keep sweep identity, but remove exact duplicate recorded metric
    # rows within the same sweep/config/dataset/seed/metric.
    tuning = tuning.drop_duplicates(
        subset=[
            "method",
            "sweep",
            "config_hash",
            "dataset",
            "seed",
            "metric_name",
        ],
        keep="last",
    )

    summary = (
        tuning.groupby(
            [
                "method",
                "sweep",
                "config_hash",
                "config",
                "dataset",
                "metric_name",
            ],
            dropna=False,
        )["metric_value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary["mean"] = summary["mean"].round(3)
    summary["std"] = summary["std"].round(3)

    table = summary.pivot(
        index=[
            "method",
            "sweep",
            "config_hash",
            "config",
        ],
        columns=[
            "dataset",
            "metric_name",
        ],
        values="mean",
    )

    table.columns = [
        f"{dataset.replace('synthetic_', '')}_{metric}"
        for dataset, metric in table.columns
    ]
    table = table.reset_index()

    desired_columns = [
        "method",
        "sweep",
        "config_hash",
        "config",
        "none_false_alarms_per_10000",
        "sudden_detection_delay",
        "sudden_false_alarms_per_10000",
        "sudden_missed_detections",
        "gradual_detection_delay",
        "gradual_false_alarms_per_10000",
        "gradual_missed_detections",
        "recurring_detection_delay",
        "recurring_false_alarms_per_10000",
        "recurring_missed_detections",
    ]

    existing_columns = [column for column in desired_columns if column in table.columns]
    table = table[existing_columns]

    table = table.sort_values(
        [
            "method",
            "sweep",
            "none_false_alarms_per_10000",
        ],
        na_position="last",
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_colwidth", 80)

    print("\n=== SYNTHETIC DETECTOR TUNING SUMMARY ===\n")
    print(table.to_string(index=False))

    output = RESULTS_DIR / "detector_tuning_summary.csv"
    table.to_csv(output, index=False)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
