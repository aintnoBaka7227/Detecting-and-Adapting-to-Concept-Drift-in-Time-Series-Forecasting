"""
Check residual values at unmatched AEMO alarm timestamps.
Compare alarm residuals with the full test-period residuals.
Save the comparison results as CSV files.
"""

import sys
from pathlib import Path

import pandas as pd

# Set the folder paths
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"
analysis_dir = output_dir / "alarm_analysis"
src_dir = project_dir / "src"

sys.path.insert(0, str(src_dir))

from aemo_preprocessing import load_aemo_data, split_aemo_data
from deseasonalise import fit_seasonal_profile, transform_data

regions = ["NSW1", "SA1"]
detectors = ["ADWIN", "KSWIN", "Page-Hinkley"]

analysis_dir.mkdir(parents=True, exist_ok=True)


def load_test_residuals(region):

    # Load the demand data
    filename = "NSW.csv" if region == "NSW1" else "SA.csv"

    data = load_aemo_data(filename, region)

    train, calibration, test = split_aemo_data(data)

    # Fit seasonal patterns using training data only
    profile = fit_seasonal_profile(train)

    # Calculate test-period residuals
    test = transform_data(test, profile)

    test = test.copy()

    test["timestamp"] = pd.to_datetime(
        test["SETTLEMENTDATE"]
    )

    test["abs_residual"] = (
        test["standardised_residual"].abs()
    )

    return test[
        ["timestamp", "standardised_residual", "abs_residual"]
    ]


def load_unmatched_alarms(region, detector):

    # Load the evaluated detector alarms
    detector_name = detector.lower().replace("-", "_")

    filename = (
        f"{region.lower()}_{detector_name}_matches.csv"
    )

    alarms = pd.read_csv(
        output_dir / filename
    )

    alarms["timestamp"] = pd.to_datetime(
        alarms["timestamp"]
    )

    # Keep unmatched accepted alarms
    unmatched = alarms[
        alarms["label"] == "Unmatch"
    ].copy()

    return unmatched


def main():

    summary_rows = []
    alarm_rows = []

    for region in regions:

        print("\nRegion:", region)

        # Load residuals for the full test period
        test = load_test_residuals(region)

        typical_median = test["abs_residual"].median()
        typical_p90 = test["abs_residual"].quantile(0.90)
        typical_p95 = test["abs_residual"].quantile(0.95)

        for detector in detectors:

            unmatched = load_unmatched_alarms(
                region, detector
            )

            # Find the residual at each alarm timestamp
            alarm_values = unmatched.merge(
                test,
                on="timestamp",
                how="left",
                validate="many_to_one"
            )

            if alarm_values["abs_residual"].isna().any():
                raise ValueError(
                    f"Missing residual values for {region} {detector}"
                )

            # Compare alarms with the test-period distribution
            alarm_median = alarm_values[
                "abs_residual"
            ].median()

            alarm_mean = alarm_values[
                "abs_residual"
            ].mean()

            above_p90 = (
                alarm_values["abs_residual"] >= typical_p90
            ).sum()

            above_p95 = (
                alarm_values["abs_residual"] >= typical_p95
            ).sum()

            total_alarms = len(alarm_values)

            summary_rows.append({
                "region": region,
                "detector": detector,
                "unmatched_alarms": total_alarms,
                "test_median_abs_residual": round(
                    typical_median, 3
                ),
                "alarm_median_abs_residual": round(
                    alarm_median, 3
                ),
                "alarm_mean_abs_residual": round(
                    alarm_mean, 3
                ),
                "above_test_p90": int(above_p90),
                "above_test_p95": int(above_p95),
                "percent_above_p90": round(
                    100 * above_p90 / total_alarms, 1
                ),
                "percent_above_p95": round(
                    100 * above_p95 / total_alarms, 1
                )
            })

            # Save individual alarm information
            alarm_values["region"] = region
            alarm_values["detector"] = detector

            alarm_rows.append(
                alarm_values[
                    [
                        "region",
                        "detector",
                        "timestamp",
                        "standardised_residual",
                        "abs_residual"
                    ]
                ]
            )

    # Save the summary
    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        analysis_dir / "alarm_residual_summary.csv",
        index=False
    )

    # Save residual values for individual alarms
    alarm_details = pd.concat(
        alarm_rows,
        ignore_index=True
    )

    alarm_details.to_csv(
        analysis_dir / "alarm_residual_details.csv",
        index=False
    )

    print("\nResidual Analysis Results")
    print(summary.to_string(index=False))

    print("\nResidual analysis completed.")


if __name__ == "__main__":
    main()
