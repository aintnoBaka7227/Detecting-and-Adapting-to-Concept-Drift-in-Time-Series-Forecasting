
import os
import sys

import pandas as pd

# Import from the project folder
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.detectors import run_detector
from src.evaluation import evaluate_detections
from src.preprocessing import preprocess_synthetic


# Settings
drift_types = ["sudden", "gradual", "recurring", "none"]
detectors = ["ADWIN", "KSWIN", "Page-Hinkley"]
seeds = [101, 102, 103, 104, 105]

train_size = 4000
period = 48
matching_window = 1000

results = []

print("PREPROCESSED SYNTHETIC BASELINE")
print("--------------------------------")


# Test the calibration data
for kind in drift_types:

    for seed in seeds:

        file_path = (
            f"data/synthetic/calibration/"
            f"series_{kind}_n20000_seed{seed}.csv"
        )

        # Read the dataset
        df = pd.read_csv(file_path)

        # Get the demand values
        raw_values = df["y"].values

        # Remove seasonality and scale the data
        values = preprocess_synthetic(
            raw_values,
            train_size=train_size,
            period=period
        )

        # Get the known drift points
        true_drifts = df.loc[
            df["is_changepoint"] == 1, "t"
        ].tolist()

        # Run the detectors
        for name in detectors:

            alarms = run_detector(values, name)

            # Ignore training alarms
            test_alarms = [
                alarm for alarm in alarms
                if alarm >= train_size
            ]

            # Calculate the results
            metrics = evaluate_detections(
                alarms=test_alarms,
                true_drifts=true_drifts,
                n=len(df) - train_size,
                matching_window=matching_window,
                drift_type=kind
            )

            # Save the results
            results.append({
                "detector": name,
                "drift_type": kind,
                "seed": seed,
                "total_alarms": len(test_alarms),
                "mean_delay": metrics["mean_delay"],
                "false_alarms": metrics["false_alarms"],
                "false_alarms_per_10k":
                    metrics["false_alarms_per_10k"],
                "missed": metrics["missed_detections"],
                "matched": metrics["matched_detections"]
            })

            # Print the results
            print("\nDrift type:", kind)
            print("Seed:", seed)
            print("Detector:", name)
            print("Total alarms:", len(test_alarms))
            print("Detection delay:", metrics["mean_delay"])
            print(
                "False alarms per 10k:",
                round(metrics["false_alarms_per_10k"], 2)
            )
            print("Missed detections:", metrics["missed_detections"])
            print("--------------------------------")


# Save results to CSV
os.makedirs("outputs", exist_ok=True)

results_df = pd.DataFrame(results)

output_file = "outputs/calibration_preprocessed_baseline.csv"

results_df.to_csv(output_file, index=False)

print("\nPreprocessed baseline evaluation completed.")
print("Total experiments:", len(results_df))
print("Results saved to:", output_file)
