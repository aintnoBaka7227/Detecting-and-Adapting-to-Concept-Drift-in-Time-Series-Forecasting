"""
Calculate the random-chance baseline for AEMO events.
Use the Tier 1 event windows and accepted alarm counts.
Save the results for Table 2.
"""

import sys
from pathlib import Path

import pandas as pd

# Set the folder paths
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = project_dir / "outputs"
src_dir = project_dir / "src"

sys.path.insert(0, str(src_dir))

from evaluation import calculate_match_window

# AEMO test period
test_start = pd.Timestamp("2020-03-01")
test_end = pd.Timestamp("2024-01-01")

test_days = (test_end - test_start) / pd.Timedelta(days=1)


def main():

    # Load the event catalogue and evaluation results
    events = pd.read_csv(data_dir / "events.csv")

    results = pd.read_csv(
        output_dir / "table2_evaluation_summary.csv"
    )

    # Select positive Tier 1 events
    tier1_events = events[
        (pd.to_numeric(events["tier"]) == 1)
        & (events["event_id"] != "EVT-2021-10")
    ].copy()

    table_rows = []

    for _, result in results.iterrows():

        region = result["region"]
        detector = result["detector"]

        # Use accepted alarms after the refractory period
        n_alarms = int(result["n_effective_detections"])

        # Include regional events and NEM events
        region_events = tier1_events[
            tier1_events["region"].isin(["NEM", region])
        ]

        chance_matches = 0.0

        for _, event in region_events.iterrows():

            # Get the same window used in event matching
            start, end = calculate_match_window(
                event["start_date"],
                event["end_date"],
                event["date_precision"]
            )

            # Keep only the part inside the test period
            start = max(start, test_start)
            end = min(end, test_end)

            window_days = max(
                0.0,
                (end - start) / pd.Timedelta(days=1)
            )

            # Chance of at least one random alarm in this window
            chance = 1 - (
                1 - window_days / test_days
            ) ** n_alarms

            chance_matches += chance

        table_rows.append({
            "region": region,
            "detector": detector,
            "accepted_alarms": n_alarms,
            "tier1_events": len(region_events),
            "tier1_matched": int(result["n_matched_t1"]),
            "chance_tier1": round(chance_matches, 2)
        })

    # Create the results table
    chance_results = pd.DataFrame(table_rows)

    # Save the results
    chance_results.to_csv(
        output_dir / "table2_chance_baseline.csv",
        index=False
    )

    print("\nAEMO Random-Chance Baseline")
    print(chance_results.to_string(index=False))

    print("\nRandom-chance baseline completed.")


if __name__ == "__main__":
    main()