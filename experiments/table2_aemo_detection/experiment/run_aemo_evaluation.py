"""
Evaluate AEMO drift detections using the team's shared evaluation code.
Apply the 14-day refractory period and match alarms with documented events.
Save the evaluation results for Table 2.
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sys.path.insert(0, str(SRC_DIR))

from evaluation import match_unmatch, calculate_event_metrics


def main():
    # Load the documented AEMO events
    events = pd.read_csv(DATA_DIR / "events.csv")

    # Keep the five-minute settlement event as a separate negative control
    negative_control = events[
        events["event_id"] == "EVT-2021-10"
    ]

    positive_events = events[
        events["event_id"] != "EVT-2021-10"
    ].copy()

    print("Positive event catalogue:", len(positive_events))
    print("Negative control events:", len(negative_control))

    results = []

    for region in ["NSW1", "SA1"]:
        for detector in ["ADWIN", "KSWIN", "Page-Hinkley"]:

            # Read the raw alarms from the detection stage
            filename = (
                f"{region.lower()}_"
                f"{detector.lower().replace('-', '_')}_raw_alarms.csv"
            )

            alarms = pd.read_csv(OUTPUT_DIR / filename)

            # Apply the shared matching and refractory rules
            matches = match_unmatch(
                detected_timestamps=alarms["timestamp"],
                events=positive_events,
                region=region
            )

            # Calculate the shared evaluation metrics
            metrics = calculate_event_metrics(
                match_results=matches,
                events=positive_events,
                region=region
            )

            # Save the labels for each detection
            match_filename = (
                f"{region.lower()}_"
                f"{detector.lower().replace('-', '_')}_matches.csv"
            )

            matches.to_csv(OUTPUT_DIR / match_filename, index=False)

            metrics["region"] = region
            metrics["detector"] = detector

            results.append(metrics)

            print(
                region,
                detector,
                "Raw:", metrics["n_total_detections"],
                "Accepted:", metrics["n_effective_detections"],
                "Matched:", metrics["n_matched_detections"]
            )

    # Save the summary of all six detector runs
    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "table2_evaluation_summary.csv",
        index=False
    )

    print("\nAEMO evaluation completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()