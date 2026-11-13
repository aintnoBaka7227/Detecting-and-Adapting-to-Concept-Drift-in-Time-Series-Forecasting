"""
Create the final Table 2 using AEMO detection results.
Include event delays, alarm counts and the chance baseline.
Save the main table and detailed results as CSV files.
"""

from pathlib import Path
import pandas as pd

# Set the folder paths
project_dir = Path(__file__).resolve().parent.parent
output_dir = project_dir / "outputs"

# Tier 1 events
event_names = {
    "EVT-2020-02": "COVID (delay, d)",
    "EVT-2020-07": "solar ramp",
    "EVT-2022-03": "2022 suspension",
    "EVT-2023-14": "security directions"
}


def main():

    # Load the existing results
    results = pd.read_csv(
        output_dir / "table2_evaluation_summary.csv"
    )

    tier1_matches = pd.read_csv(
        output_dir / "tier1_matched_events.csv"
    )

    alarm_checks = pd.read_csv(
        output_dir / "aemo_alarm_checks.csv"
    )

    chance_results = pd.read_csv(
        output_dir / "table2_chance_baseline.csv"
    )

    table_rows = []

    # Create one row for each detector and region
    for _, result in results.iterrows():

        region = result["region"]
        detector = result["detector"]

        matches = tier1_matches[
            (tier1_matches["demand_region"] == region)
            & (tier1_matches["detector"] == detector)
        ]

        row = {
            "detector": detector,
            "region": region
        }

        # Add the detection delay for each event
        for event_id, event_name in event_names.items():

            event_matches = matches[
                matches["event_id"] == event_id
            ]

            if event_matches.empty:
                row[event_name] = "not detected"
            else:
                row[event_name] = round(
                    float(event_matches.iloc[0]["delay_days"]), 2
                )

        # Add evaluation metrics
        row["raw alarms"] = int(result["n_total_detections"])
        row["accepted alarms"] = int(result["n_effective_detections"])
        row["Tier 1 matches"] = int(result["n_matched_t1"])
        row["Tier 2 matches"] = int(result["n_matched_t2"])
        row["unmatched"] = int(result["n_unmatched_detections"])
        row["precision"] = round(float(result["precision_t1"]), 3)

        # Add the chance baseline
        chance_row = chance_results[
            (chance_results["region"] == region)
            & (chance_results["detector"] == detector)
        ]

        if chance_row.empty:
            raise ValueError(
                f"Missing chance baseline for {region} {detector}"
            )

        row["chance_tier1"] = float(
            chance_row.iloc[0]["chance_tier1"]
        )

        table_rows.append(row)

    # Save the main results
    table2 = pd.DataFrame(table_rows)

    table2.to_csv(
        output_dir / "table2_results.csv",
        index=False
    )

    # Combine all evaluation information
    detailed_results = results.merge(
        alarm_checks,
        on=["region", "detector"],
        how="left"
    )

    detailed_results = detailed_results.merge(
        chance_results[
            ["region", "detector", "chance_tier1"]
        ],
        on=["region", "detector"],
        how="left"
    )

    detailed_results.to_csv(
        output_dir / "table2_detailed_results.csv",
        index=False
    )

    print("\nFinal Table 2 Results")
    print(table2.to_string(index=False))

    print("\nTable 2 files saved successfully.")


if __name__ == "__main__":
    main()