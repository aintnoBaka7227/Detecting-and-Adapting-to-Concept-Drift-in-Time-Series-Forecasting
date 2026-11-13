"""
Analyse unmatched drift alarms in the AEMO results.
Check monthly patterns, seasonal patterns and detector agreement.
Save the results as CSV files for further investigation.
"""

from pathlib import Path
import pandas as pd

# Set the folder paths
project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs"
analysis_dir = output_dir / "alarm_analysis"

analysis_dir.mkdir(parents=True, exist_ok=True)

regions = ["NSW1", "SA1"]
detectors = ["ADWIN", "KSWIN", "Page-Hinkley"]

# Compare alarms within three days
agreement_window = pd.Timedelta(days=3)


def load_unmatched_alarms(region, detector):

    # Load the evaluated alarms
    detector_name = detector.lower().replace("-", "_")

    filename = (
        f"{region.lower()}_{detector_name}_matches.csv"
    )

    alarms = pd.read_csv(output_dir / filename)

    alarms["timestamp"] = pd.to_datetime(
        alarms["timestamp"]
    )

    # Keep unmatched accepted alarms only
    unmatched = alarms[
        alarms["label"] == "Unmatch"
    ].copy()

    unmatched["region"] = region
    unmatched["detector"] = detector

    return unmatched


def main():

    all_alarms = []

    # Load unmatched alarms from all detectors
    for region in regions:

        for detector in detectors:

            alarms = load_unmatched_alarms(
                region, detector
            )

            all_alarms.append(alarms)

    unmatched = pd.concat(
        all_alarms,
        ignore_index=True
    )

    unmatched = unmatched.sort_values(
        ["region", "timestamp"]
    )

    # Save all unmatched alarms
    unmatched.to_csv(
        analysis_dir / "unmatched_alarms.csv",
        index=False
    )

    # Count unmatched alarms by month
    unmatched["month"] = (
        unmatched["timestamp"]
        .dt.to_period("M")
        .astype(str)
    )

    monthly = (
        unmatched.groupby(
            ["region", "detector", "month"]
        )
        .size()
        .reset_index(name="unmatched_alarms")
    )

    monthly.to_csv(
        analysis_dir / "monthly_unmatched.csv",
        index=False
    )

    # Assign each alarm to a season
    month_numbers = unmatched["timestamp"].dt.month

    season_names = {
        12: "Summer",
        1: "Summer",
        2: "Summer",
        3: "Autumn",
        4: "Autumn",
        5: "Autumn",
        6: "Winter",
        7: "Winter",
        8: "Winter",
        9: "Spring",
        10: "Spring",
        11: "Spring"
    }

    unmatched["season"] = month_numbers.map(
        season_names
    )

    # Count alarms by season
    seasonal = (
        unmatched.groupby(
            ["region", "detector", "season"]
        )
        .size()
        .reset_index(name="unmatched_alarms")
    )

    seasonal.to_csv(
        analysis_dir / "seasonal_unmatched.csv",
        index=False
    )

    # Check whether other detectors alarmed nearby
    agreement_rows = []

    for region in regions:

        region_alarms = unmatched[
            unmatched["region"] == region
        ]

        for detector in detectors:

            detector_alarms = region_alarms[
                region_alarms["detector"] == detector
            ]

            other_alarms = region_alarms[
                region_alarms["detector"] != detector
            ]

            for _, alarm in detector_alarms.iterrows():

                timestamp = alarm["timestamp"]

                nearby = other_alarms[
                    (
                        other_alarms["timestamp"]
                        - timestamp
                    ).abs() <= agreement_window
                ]

                other_detectors = sorted(
                    nearby["detector"].unique()
                )

                agreement_rows.append({
                    "region": region,
                    "detector": detector,
                    "timestamp": timestamp,
                    "other_detectors": ", ".join(
                        other_detectors
                    ),
                    "detectors_agreeing": (
                        1 + len(other_detectors)
                    )
                })

    agreement = pd.DataFrame(agreement_rows)

    agreement.to_csv(
        analysis_dir / "detector_agreement.csv",
        index=False
    )

    # Summarise the results
    print("\nUnmatched Alarms by Detector")

    summary = (
        unmatched.groupby(
            ["region", "detector"]
        )
        .size()
        .reset_index(name="unmatched_alarms")
    )

    print(summary.to_string(index=False))

    print("\nSeasonal Unmatched Alarms")
    print(seasonal.to_string(index=False))

    print("\nDetector Agreement")

    agreement_summary = (
        agreement.groupby(
            ["region", "detector", "detectors_agreeing"]
        )
        .size()
        .reset_index(name="alarms")
    )

    print(agreement_summary.to_string(index=False))

    # Show months with the highest alarm counts
    print("\nTop 10 Months with Most Unmatched Alarms")

    top_months = monthly.sort_values(
        "unmatched_alarms",
        ascending=False
    ).head(10)

    print(top_months.to_string(index=False))

    print("\nUnmatched alarm analysis completed.")


if __name__ == "__main__":
    main()
