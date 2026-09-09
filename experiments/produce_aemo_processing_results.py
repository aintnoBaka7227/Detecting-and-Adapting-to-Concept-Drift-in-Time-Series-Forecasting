"""Produce the AEMO detector input-processing comparison from recorded runs.

This script is read-only. It does not rerun detectors or recompute
detection metrics. It selects the latest recorded result for each
relevant region, detector, and processing/configuration stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.results_io import RUNS_CSV, RUNS_DIR

OUTPUT = Path("results/aemo_processing_summary.csv")

METHODS = ("adwin", "kswin", "page_hinkley")
REGIONS = ("SA1", "NSW1")

STAGES = {
    "aemo_raw_30min": "Raw 30-min",
    "aemo_daily_mean": "Daily baseline",
    "aemo_daily_mean_complete_days_frozen": "Daily + frozen",
}


def load_config(config_hash: str) -> dict:
    """Load the recorded configuration for one experiment run."""

    path = RUNS_DIR / config_hash / "config.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing configuration: {path}")

    with path.open(encoding="utf-8") as file:
        return json.load(file)


def latest_detection_counts() -> pd.DataFrame:
    """Return the latest AEMO detection count for each experiment."""

    df = pd.read_csv(RUNS_CSV)

    required = {
        "group",
        "method",
        "dataset",
        "region",
        "config_hash",
        "split_id",
        "metric_name",
        "metric_value",
        "timestamp",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"runs.csv missing columns: {sorted(missing)}")

    df = df[
        (df["group"] == "detection")
        & (df["method"].isin(METHODS))
        & (df["region"].isin(REGIONS))
        & (df["dataset"].isin(STAGES))
        & (df["split_id"] == "aemo_frozen_v1")
        & (df["metric_name"] == "n_detections")
    ].copy()

    if df.empty:
        raise ValueError("No matching AEMO detection-count runs found.")

    # runs.csv is append-only. Sorting by timestamp and keeping the
    # final occurrence selects the latest recorded result without
    # deleting or modifying earlier experimental evidence.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    df = (
        df.sort_values("timestamp")
        .drop_duplicates(
            subset=["region", "method", "dataset"],
            keep="last",
        )
        .copy()
    )

    df["stage"] = df["dataset"].map(STAGES)

    return df


def validate_frozen_configs(df: pd.DataFrame) -> None:
    """Confirm frozen AEMO runs use synthetic-only parameter selection."""

    frozen = df[df["dataset"] == "aemo_daily_mean_complete_days_frozen"]

    expected_runs = len(REGIONS) * len(METHODS)

    if len(frozen) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} frozen AEMO results, found {len(frozen)}."
        )

    for row in frozen.itertuples():
        config = load_config(row.config_hash)

        if config.get("parameter_selection") != "synthetic_only":
            raise ValueError(
                f"Frozen run {row.config_hash} is not labelled synthetic_only."
            )

        if config.get("input_processing") != "daily_mean_complete_days":
            raise ValueError(
                f"Frozen run {row.config_hash} has unexpected input processing."
            )


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build the raw-vs-processed detector-count comparison."""

    table = df.pivot(
        index=["region", "method"],
        columns="stage",
        values="metric_value",
    ).reset_index()

    expected = {
        "Raw 30-min",
        "Daily baseline",
        "Daily + frozen",
    }

    missing = expected - set(table.columns)

    if missing:
        raise ValueError(f"Missing comparison stages: {sorted(missing)}")

    for column in expected:
        table[column] = table[column].astype(int)

    table["Raw to final reduction (%)"] = (
        100 * (table["Raw 30-min"] - table["Daily + frozen"]) / table["Raw 30-min"]
    ).round(1)

    order = {method: index for index, method in enumerate(METHODS)}

    table["_method_order"] = table["method"].map(order)

    table = (
        table.sort_values(["region", "_method_order"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    return table


def main() -> None:
    """Produce and save the AEMO processing comparison."""

    df = latest_detection_counts()

    validate_frozen_configs(df)

    summary = build_summary(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT, index=False)

    print("\n=== AEMO Raw vs Processed Detector Comparison ===")
    print(summary.to_string(index=False))

    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
