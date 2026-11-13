"""
This code prepares the forecast data for Table 5.
I check the timestamps and actual demand values for each adaptation arm.
I also assign Page-Hinkley regimes so the forecasts can be evaluated.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# Set the folder paths
base = Path(__file__).resolve().parents[1]
forecast_folder = base / "data" / "adaptation_forecasts"
output_folder = base / "data" / "prepared_forecasts"

output_folder.mkdir(parents=True, exist_ok=True)


# Adaptation arms
arms = {
    "A": "A_never_retrain",
    "B": "B_every_30_days",
    "C": "C_drift_full_history",
    "D": "D_drift_recent_180d"
}

regions = ["SA1", "NSW1"]
seeds = range(1, 6)

prepared_count = 0
missing_count = 0


for region in regions:
    for seed in seeds:

        filename = f"curve_aemo_{region}_{seed}.csv"

        # Use Arm C as the reference for Page-Hinkley regimes
        reference_path = forecast_folder / "arm_C" / filename

        if not reference_path.exists():
            print(f"Waiting for Arm C: {region}, seed {seed}")
            missing_count += 4
            continue

        reference = pd.read_csv(reference_path)

        required = ["timestamp", "actual", "regime"]

        if not all(col in reference.columns for col in required):
            raise ValueError(f"Missing columns in {reference_path}")

        reference["timestamp"] = pd.to_datetime(reference["timestamp"])

        if reference["timestamp"].duplicated().any():
            raise ValueError(f"Duplicate timestamps in {reference_path}")

        reference = reference.sort_values("timestamp").reset_index(drop=True)

        # Use the same regime names in every file
        reference["regime"] = reference["regime"].str.replace("-", "_")

        valid_regimes = {"pre_drift", "drift", "post_drift"}

        if not reference["regime"].isin(valid_regimes).all():
            raise ValueError(f"Invalid regime in {reference_path}")

        for arm_letter, arm_name in arms.items():

            input_path = forecast_folder / f"arm_{arm_letter}" / filename

            if not input_path.exists():
                print(f"Missing: Arm {arm_letter}, {region}, seed {seed}")
                missing_count += 1
                continue

            df = pd.read_csv(input_path)

            needed = ["timestamp", "actual", "forecast"]

            if not all(col in df.columns for col in needed):
                raise ValueError(f"Missing columns in {input_path}")

            df["timestamp"] = pd.to_datetime(df["timestamp"])

            if df["timestamp"].duplicated().any():
                raise ValueError(f"Duplicate timestamps in {input_path}")

            df = df.sort_values("timestamp").reset_index(drop=True)

            # Check timestamps match the reference
            if not df["timestamp"].equals(reference["timestamp"]):
                raise ValueError(
                    f"Timestamp mismatch: {region}, Arm {arm_letter}, seed {seed}"
                )

            # Check actual demand values match
            if not np.allclose(
                df["actual"].to_numpy(),
                reference["actual"].to_numpy(),
                rtol=0,
                atol=0.000001
            ):
                raise ValueError(
                    f"Actual demand mismatch: {region}, Arm {arm_letter}, seed {seed}"
                )

            # Keep the columns needed for evaluation
            prepared = df[["timestamp", "actual", "forecast"]].copy()

            prepared["regime"] = reference["regime"]
            prepared["arm"] = arm_name
            prepared["seed"] = seed
            prepared["region"] = region

            if not np.isfinite(
                prepared[["actual", "forecast"]].to_numpy()
            ).all():
                raise ValueError(f"Invalid values in {input_path}")

            output_name = f"forecast_{region}_{arm_letter}_{seed}.csv"
            output_path = output_folder / output_name

            prepared.to_csv(output_path, index=False)

            prepared_count += 1
            print(f"Prepared: {output_name} ({len(prepared)} rows)")


print()
print("=== Preparation Summary ===")
print("Prepared files:", prepared_count)
print("Missing files:", missing_count)
print("Output folder:", output_folder)
