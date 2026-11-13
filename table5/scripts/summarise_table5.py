"""
This code summarises my Table 5 results across five seeds.
I calculate the mean and standard deviation for each evaluation metric.
The results are saved in the same reporting format used for Table 5.
"""

from pathlib import Path
import pandas as pd


# Set the input and output files
base_folder = Path(__file__).resolve().parents[1]
input_file = base_folder / "outputs" / "table5_seed_results.csv"
output_file = base_folder / "outputs" / "table5_final.csv"

data = pd.read_csv(input_file)


# Set the regions, arms and regimes
regions = ["SA1", "NSW1"]
arms = ["A", "B", "C", "D"]
regimes = ["pre_drift", "drift", "post_drift"]
seeds = [1, 2, 3, 4, 5]

metrics = [
    "coverage",
    "mean_interval_width",
    "normalised_interval_width",
    "worst_24h_coverage"
]


# Check that all five seeds are available
expected_rows = len(regions) * len(arms) * len(regimes) * len(seeds)

if len(data) != expected_rows:
    raise ValueError("Incorrect number of seed results")

if data.duplicated(["region", "arm", "regime", "seed"]).any():
    raise ValueError("Duplicate seed results found")


results = []

for region in regions:
    for arm in arms:
        for regime in regimes:

            group = data[
                (data["region"] == region)
                & (data["arm"] == arm)
                & (data["regime"] == regime)
            ]

            if sorted(group["seed"].tolist()) != seeds:
                raise ValueError(
                    f"Missing seeds for {region}, {arm}, {regime}"
                )

            row = {
                "model": "xgboost",
                "arm": arm,
                "region": region,
                "UQ method": "fixed split conformal",
                "regime": regime
            }

            # Calculate mean and sample standard deviation
            for metric in metrics:
                mean = group[metric].mean()
                std = group[metric].std(ddof=1)

                row[metric + "_mean"] = mean
                row[metric + "_std"] = std

                if metric == "mean_interval_width":
                    row[metric] = f"{mean:.0f} ± {std:.0f}"
                else:
                    row[metric] = f"{mean:.3f} ± {std:.3f}"

            # Window counts and seed counts
            window_counts = group["n_complete_windows"]

            if window_counts.nunique() != 1:
                raise ValueError(
                    f"Window counts differ for {region}, {arm}, {regime}"
                )

            row["complete_24h_windows"] = int(window_counts.iloc[0])
            row["seeds"] = len(group)

            results.append(row)


# Create the final table
summary = pd.DataFrame(results)

# Arrange the columns for presentation
display_columns = [
    "model",
    "arm",
    "region",
    "UQ method",
    "regime",
    "coverage",
    "mean_interval_width",
    "normalised_interval_width",
    "worst_24h_coverage",
    "complete_24h_windows",
    "seeds"
]

# Save the full results, including numerical means and SDs
summary.to_csv(output_file, index=False)

# Display the formatted table
print("\n=== Final Table 5 Summary ===")
print(summary[display_columns].to_string(index=False))

print("\nResults saved:", output_file)
print("Total rows:", len(summary))
print("Seeds per group:", summary["seeds"].unique().tolist())
