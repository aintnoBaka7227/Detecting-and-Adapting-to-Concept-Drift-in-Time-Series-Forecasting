
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.detectors import run_detector
from src.preprocessing import preprocess_synthetic
from src.evaluation import evaluate_detections


# File paths
test_folder = Path("data/synthetic/test")
parameters_file = Path("outputs/expanded_best_parameters.csv")
output_folder = Path("outputs")

output_folder.mkdir(exist_ok=True)

# Load the selected settings
best_parameters = pd.read_csv(parameters_file)

parameters = {}

for _, row in best_parameters.iterrows():
    parameters[row["detector"]] = json.loads(row["parameters"])

print("Selected parameters:")

for name, params in parameters.items():
    print(name, params)


# Drift points used in the generated data
def get_true_drifts(drift_type, n):

    if drift_type == "none":
        return []

    if drift_type == "sudden":
        return [n // 4]

    if drift_type == "gradual":
        start = n // 4
        end = start + min(1000, n // 4)
        return [start, end]

    if drift_type == "recurring":
        return [n // 3, (2 * n) // 3]

    raise ValueError("Unknown drift type")


# Get the data column
def get_values(df):

    for column in ["value", "y", "demand", "TOTALDEMAND"]:
        if column in df.columns:
            return pd.to_numeric(
                df[column], errors="raise"
            ).to_numpy(dtype=float)

    raise ValueError(
        "Cannot find the data column. Columns: "
        + str(df.columns.tolist())
    )


# Get the test files
files = sorted(test_folder.glob("*.csv"))

print("\nTest files found:", len(files))

if len(files) != 20:
    raise ValueError(
        "Expected 20 test files, found " + str(len(files))
    )

results = []

drift_types = ["none", "sudden", "gradual", "recurring"]

for file_path in files:

    filename = file_path.stem.lower()

    matching_types = [
        kind for kind in drift_types
        if kind in filename
    ]

    if len(matching_types) != 1:
        raise ValueError(
            "Cannot identify drift type from: "
            + file_path.name
        )

    drift_type = matching_types[0]

    df = pd.read_csv(file_path)
    values = get_values(df)

    if len(values) != 20000:
        raise ValueError(
            "Unexpected number of observations in "
            + file_path.name
        )

    if not np.isfinite(values).all():
        raise ValueError(
            "Invalid values in " + file_path.name
        )

    # Use the same preprocessing as calibration
    processed = preprocess_synthetic(
        values,
        train_size=4000,
        period=48
    )

    true_drifts = get_true_drifts(drift_type, len(values))

    print("\nDataset:", file_path.name)
    print("Drift type:", drift_type)
    print("True drift points:", true_drifts)

    for name, params in parameters.items():

        alarms = run_detector(
            processed,
            name,
            params
        )

        metrics = evaluate_detections(
            alarms=alarms,
            true_drifts=true_drifts,
            n=len(values),
            matching_window=336,
            drift_type=drift_type
        )

        results.append({
            "dataset": file_path.name,
            "drift_type": drift_type,
            "detector": name,
            "parameters": json.dumps(params),
            "true_drifts": json.dumps(true_drifts),
            "alarms": json.dumps(alarms),
            "mean_delay": metrics["mean_delay"],
            "false_alarms": metrics["false_alarms"],
            "false_alarms_per_10k": metrics["false_alarms_per_10k"],
            "missed_detections": metrics["missed_detections"],
            "matched_detections": metrics["matched_detections"]
        })

        print(
            name,
            "| Delay:", metrics["mean_delay"],
            "| False alarms:", metrics["false_alarms_per_10k"],
            "| Missed:", metrics["missed_detections"]
        )


# Save results for each dataset
results_df = pd.DataFrame(results)

results_df.to_csv(
    output_folder / "synthetic_test_results.csv",
    index=False
)


# Calculate the summary
summary = (
    results_df
    .groupby(["detector", "drift_type"])
    .agg(
        mean_delay=("mean_delay", "mean"),
        mean_delay_sd=("mean_delay", "std"),
        false_alarms_per_10k=("false_alarms_per_10k", "mean"),
        total_missed=("missed_detections", "sum"),
        total_matched=("matched_detections", "sum"),
        datasets=("dataset", "count")
    )
    .reset_index()
)

summary.to_csv(
    output_folder / "synthetic_test_summary.csv",
    index=False
)

print("\nHELD-OUT SYNTHETIC TEST COMPLETED")
print("--------------------------------")
print(summary.to_string(index=False))

print("\nResults saved:")
print("outputs/synthetic_test_results.csv")
print("outputs/synthetic_test_summary.csv")
