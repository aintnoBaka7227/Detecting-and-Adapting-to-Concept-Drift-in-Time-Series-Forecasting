
import os
import sys
import json
import itertools

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.detectors import run_detector
from src.evaluation import evaluate_detections
from src.preprocessing import preprocess_synthetic


# Calibration settings
drift_types = ["sudden", "gradual", "recurring", "none"]
seeds = [101, 102, 103, 104, 105]

train_size = 4000
period = 48
matching_window = 1000


# Settings to test
parameter_grid = {
    "ADWIN": {
        "delta": [0.002, 0.001, 0.0001]
    },
    "KSWIN": {
        "alpha": [0.005, 0.001, 0.0001],
        "window_size": [200, 400],
        "stat_size": [50]
    },
    "Page-Hinkley": {
        "threshold": [20, 50, 100],
        "delta": [0.005, 0.01],
        "min_instances": [30]
    }
}


def get_combinations(grid):
    keys = list(grid.keys())
    values = list(grid.values())

    combinations = []

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        combinations.append(params)

    return combinations


# Load the calibration data
datasets = {}

for kind in drift_types:
    for seed in seeds:

        path = (
            f"data/synthetic/calibration/"
            f"series_{kind}_n20000_seed{seed}.csv"
        )

        df = pd.read_csv(path)

        values = preprocess_synthetic(
            df["y"].values,
            train_size=train_size,
            period=period
        )

        true_drifts = df.loc[
            df["is_changepoint"] == 1, "t"
        ].tolist()

        datasets[(kind, seed)] = (values, true_drifts)


results = []

# Test each setting
for name, grid in parameter_grid.items():

    combinations = get_combinations(grid)

    print("\nTuning:", name)
    print("Configurations:", len(combinations))

    for config_id, params in enumerate(combinations, start=1):

        all_delays = []
        total_missed = 0
        false_alarm_rates = []

        for kind in drift_types:
            for seed in seeds:

                values, true_drifts = datasets[(kind, seed)]

                # Set the KSWIN seed
                run_params = params.copy()

                if name == "KSWIN":
                    run_params["seed"] = seed

                alarms = run_detector(
                    values,
                    name,
                    run_params
                )

                # Ignore training alarms
                alarms = [
                    alarm for alarm in alarms
                    if alarm >= train_size
                ]

                metrics = evaluate_detections(
                    alarms=alarms,
                    true_drifts=true_drifts,
                    n=len(values) - train_size,
                    matching_window=matching_window,
                    drift_type=kind
                )

                all_delays.extend(metrics["delays"])

                total_missed += metrics["missed_detections"]

                false_alarm_rates.append(
                    metrics["false_alarms_per_10k"]
                )

        mean_delay = (
            float(np.mean(all_delays))
            if all_delays else float("nan")
        )

        mean_false_alarms = float(
            np.mean(false_alarm_rates)
        )

        # Score:
        # Missed detections and false alarms count more
        # Also include detection delay
        delay_penalty = (
            mean_delay / 100
            if not np.isnan(mean_delay)
            else 20
        )

        score = (
            total_missed * 100
            + mean_false_alarms
            + delay_penalty
        )

        results.append({
            "detector": name,
            "config_id": config_id,
            "parameters": json.dumps(params),
            "mean_delay": mean_delay,
            "false_alarms_per_10k": mean_false_alarms,
            "total_missed": total_missed,
            "score": score
        })

        print(
            "Config:", config_id,
            "Params:", params,
            "Delay:", round(mean_delay, 2),
            "False alarms:", round(mean_false_alarms, 2),
            "Missed:", total_missed,
            "Score:", round(score, 2)
        )


# Save the tuning results
os.makedirs("outputs", exist_ok=True)

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/hyperparameter_tuning_results.csv",
    index=False
)


# Choose the best settings
best_results = (
    results_df
    .sort_values(["score", "config_id"])
    .groupby("detector", as_index=False)
    .first()
)

best_results.to_csv(
    "outputs/best_detector_parameters.csv",
    index=False
)

print("\nHYPERPARAMETER TUNING COMPLETED")
print("--------------------------------")
print(best_results.to_string(index=False))

print("\nResults saved to:")
print("outputs/hyperparameter_tuning_results.csv")
print("outputs/best_detector_parameters.csv")
