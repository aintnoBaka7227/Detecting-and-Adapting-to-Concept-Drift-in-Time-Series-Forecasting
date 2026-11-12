
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


# Settings
drift_types = ["sudden", "gradual", "recurring", "none"]
seeds = [101, 102, 103, 104, 105]

train_size = 4000
period = 48
matching_window = 1000

output_file = "outputs/expanded_tuning_results.csv"
best_file = "outputs/expanded_best_parameters.csv"

os.makedirs("outputs", exist_ok=True)


# More settings to test
parameter_grid = {
    "ADWIN": {
        "delta": [0.01, 0.002, 0.0001, 0.000001],
        "clock": [16, 32, 48],
        "min_window_length": [5, 20, 50]
    },

    "KSWIN": {
        "alpha": [0.005, 0.001, 0.0001, 0.00005],
        "window_size": [200, 400, 800],
        "stat_size": [50, 96, 150]
    },

    "Page-Hinkley": {
        "threshold": [100, 300, 600],
        "delta": [0.0001, 0.005, 0.01],
        "alpha": [0.999, 0.9995],
        "min_instances": [20, 30, 50, 100]
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


# Load calibration data
datasets = {}

print("Loading calibration datasets...")

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

print("Datasets loaded:", len(datasets))


# Load earlier tuning results
if os.path.exists(output_file):
    results_df = pd.read_csv(output_file)
    results = results_df.to_dict("records")
    print("Previous results found:", len(results))
else:
    results = []


# Test more settings
for name, grid in parameter_grid.items():

    combinations = get_combinations(grid)

    print("\nTuning:", name)
    print("Configurations:", len(combinations))

    for config_id, params in enumerate(combinations, start=1):
                # Skip KSWIN settings that do not work
        if name == "KSWIN":
            if params["window_size"] < 2 * params["stat_size"]:
                print(
                    "Skipping invalid KSWIN configuration:",
                    config_id,
                    params
                )
                continue

        # Skip settings already tested
        completed = any(
            row["detector"] == name
            and int(row["config_id"]) == config_id
            for row in results
        )

        if completed:
            print("Skipping completed config:", config_id)
            continue

        all_delays = []
        false_alarm_rates = []
        total_missed = 0

        for kind in drift_types:
            for seed in seeds:

                values, true_drifts = datasets[(kind, seed)]

                run_params = params.copy()

                if name == "KSWIN":
                    run_params["seed"] = seed

                alarms = run_detector(
                    values,
                    name,
                    run_params
                )

                test_alarms = [
                    alarm for alarm in alarms
                    if alarm >= train_size
                ]

                metrics = evaluate_detections(
                    alarms=test_alarms,
                    true_drifts=true_drifts,
                    n=len(values) - train_size,
                    matching_window=matching_window,
                    drift_type=kind
                )

                all_delays.extend(metrics["delays"])

                false_alarm_rates.append(
                    metrics["false_alarms_per_10k"]
                )

                total_missed += metrics["missed_detections"]

        mean_delay = (
            float(np.mean(all_delays))
            if all_delays else float("nan")
        )

        mean_false_alarms = float(
            np.mean(false_alarm_rates)
        )

        # Use the same score as before
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

        # Save after each test
        pd.DataFrame(results).to_csv(
            output_file,
            index=False
        )

        print(
            "Config:", config_id,
            "Params:", params,
            "Delay:", round(mean_delay, 2),
            "False alarms:", round(mean_false_alarms, 2),
            "Missed:", total_missed,
            "Score:", round(score, 2),
            flush=True
        )


# Choose the best settings
results_df = pd.DataFrame(results)

best_results = (
    results_df
    .sort_values(["score", "config_id"])
    .groupby("detector", as_index=False)
    .first()
)

best_results.to_csv(
    best_file,
    index=False
)

print("\nEXPANDED TUNING COMPLETED")
print("-------------------------")
print(best_results.to_string(index=False))

print("\nResults saved to:")
print(output_file)
print(best_file)
