"""
This code runs the Fixed Split Conformal evaluation for Table 5.
I use calibration errors to create 90% prediction intervals for each adaptation arm.
I then calculate the evaluation metrics for each region, seed and drift regime.
"""

from pathlib import Path
import numpy as np
import pandas as pd

from conformal import calculate_radius, create_intervals, load_calibration
from metrics import calculate_regime_metrics


# Set the folders
base_folder = Path(__file__).resolve().parents[1]
forecast_folder = base_folder / "data" / "prepared_forecasts"
calibration_folder = base_folder / "data" / "calibration"
output_folder = base_folder / "outputs"

output_folder.mkdir(parents=True, exist_ok=True)


# Mean demand from the training split only
train_means = {
    "SA1": 1300.873054,
    "NSW1": 7982.251665
}

regions = ["SA1", "NSW1"]
seeds = [1, 2, 3, 4, 5]
arms = ["A", "B", "C", "D"]
regimes = ["pre_drift", "drift", "post_drift"]

results = []


for region in regions:
    for seed in seeds:

        # Calculate one conformal radius for this region and seed
        calibration_file = (
            calibration_folder
            / f"calibration_forecast_aemo_{region}_{seed}.csv"
        )

        residuals = load_calibration(calibration_file)
        radius = calculate_radius(residuals, alpha=0.10)

        print(f"\nRegion: {region}, Seed: {seed}")
        print(f"Calibration observations: {len(residuals)}")
        print(f"Conformal radius: {radius:.4f}")

        for arm in arms:

            # Load the prepared forecasts
            forecast_file = (
                forecast_folder
                / f"forecast_{region}_{arm}_{seed}.csv"
            )

            data = pd.read_csv(forecast_file)

            required_columns = [
                "timestamp", "actual", "forecast", "regime"
            ]

            missing_columns = [
                column for column in required_columns
                if column not in data.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"Missing columns in {forecast_file.name}: "
                    f"{missing_columns}"
                )

            actual = data["actual"].to_numpy(dtype=float)
            forecast = data["forecast"].to_numpy(dtype=float)

            if not np.all(np.isfinite(forecast)):
                raise ValueError(
                    f"Invalid forecasts in {forecast_file.name}"
                )

            # Apply the fixed conformal radius
            lower, upper = create_intervals(forecast, radius)

            for regime in regimes:

                # Calculate the metrics for this drift regime
                values = calculate_regime_metrics(
                    actual=actual,
                    lower=lower,
                    upper=upper,
                    regimes=data["regime"].to_numpy(),
                    timestamps=data["timestamp"],
                    selected_regime=regime,
                    train_mean=train_means[region]
                )

                if values is None:
                    raise ValueError(
                        f"Regime {regime} is missing from "
                        f"{forecast_file.name}"
                    )

                results.append({
                    "region": region,
                    "seed": seed,
                    "arm": arm,
                    "regime": regime,
                    "conformal_radius": radius,
                    **values
                })

            print(f"  Arm {arm}: completed")


# Save all seed-level results
results_table = pd.DataFrame(results)

expected_rows = len(regions) * len(seeds) * len(arms) * len(regimes)

if len(results_table) != expected_rows:
    raise ValueError(
        f"Expected {expected_rows} rows, got {len(results_table)}"
    )

output_file = output_folder / "table5_seed_results.csv"
results_table.to_csv(output_file, index=False)

print("\n=== Table 5 Evaluation Summary ===")
print("Results saved:", output_file)
print("Total rows:", len(results_table))
print("Regions:", results_table["region"].unique().tolist())
print("Arms:", results_table["arm"].unique().tolist())
print("Regimes:", results_table["regime"].unique().tolist())
