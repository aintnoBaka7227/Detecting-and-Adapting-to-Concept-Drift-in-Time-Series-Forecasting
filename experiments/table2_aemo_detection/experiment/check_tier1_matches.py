"""
Check the Tier 1 events matched by the AEMO drift detectors.
Read the results from the shared evaluation stage.
Save the matched event details for verification.
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def main():
    # Load the documented event catalogue
    events = pd.read_csv(DATA_DIR / "events.csv")

    # Keep Tier 1 events except the negative control
    tier1_events = events[
        (events["tier"] == 1)
        & (events["event_id"] != "EVT-2021-10")
    ].copy()

    tier1_events = tier1_events[
        ["event_id", "start_date", "end_date", "region"]
    ]

    results = []

    for region in ["NSW1", "SA1"]:
        for detector in ["ADWIN", "KSWIN", "Page-Hinkley"]:

            detector_name = detector.lower().replace("-", "_")

            filename = (
                f"{region.lower()}_{detector_name}_matches.csv"
            )

            matches = pd.read_csv(OUTPUT_DIR / filename)

            # Select Tier 1 matched detections
            tier1_matches = matches[
                (matches["label"] == "Match")
                & (matches["tier"] == 1)
            ].copy()

            # Add the documented event dates
            tier1_matches = tier1_matches.merge(
                tier1_events,
                on="event_id",
                how="left"
            )

            tier1_matches["detector"] = detector
            tier1_matches["demand_region"] = region

            results.append(tier1_matches)

            print(
                region,
                detector,
                "Tier 1 matches:",
                len(tier1_matches)
            )

    # Combine all Tier 1 matches
    summary = pd.concat(results, ignore_index=True)

    columns = [
        "demand_region",
        "detector",
        "event_id",
        "start_date",
        "end_date",
        "timestamp",
        "delay_days"
    ]

    summary = summary[columns]

    summary.to_csv(
        OUTPUT_DIR / "tier1_matched_events.csv",
        index=False
    )

    print("\nTier 1 Matched Events")
    print(summary.to_string(index=False))

    print("\nTier 1 verification completed.")


if __name__ == "__main__":
    main()