"""
Check the AEMO accepted alarm rate and negative control.
Use the results from the shared evaluation stage.
Save the checks for Table 2.
"""

from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Test period used for AEMO detection
TEST_START = pd.Timestamp("2020-03-01")
TEST_END = pd.Timestamp("2024-01-01")

# Five-Minute Settlement negative control
CONTROL_START = pd.Timestamp("2021-10-01")
CONTROL_END = pd.Timestamp("2021-10-09")

ALARM_BUDGET = 2.0


def main():
    results = []

    test_years = (
        (TEST_END - TEST_START) / pd.Timedelta(days=365.25)
    )

    for region in ["NSW1", "SA1"]:
        for detector in ["ADWIN", "KSWIN", "Page-Hinkley"]:

            detector_name = detector.lower().replace("-", "_")

            filename = (
                f"{region.lower()}_{detector_name}_matches.csv"
            )

            matches = pd.read_csv(OUTPUT_DIR / filename)
            matches["timestamp"] = pd.to_datetime(
                matches["timestamp"]
            )

            # Ignored alarms do not count towards the budget
            accepted = matches[
                matches["label"] != "Ignored"
            ].copy()

            n_accepted = len(accepted)
            alarms_per_year = n_accepted / test_years

            # Check accepted alarms in the negative-control window
            control_alarms = accepted[
                (accepted["timestamp"] >= CONTROL_START)
                & (accepted["timestamp"] < CONTROL_END)
            ]

            results.append({
                "region": region,
                "detector": detector,
                "accepted_alarms": n_accepted,
                "alarms_per_year": round(alarms_per_year, 2),
                "within_budget": alarms_per_year <= ALARM_BUDGET,
                "negative_control_alarms": len(control_alarms)
            })

    summary = pd.DataFrame(results)

    summary.to_csv(
        OUTPUT_DIR / "aemo_alarm_checks.csv",
        index=False
    )

    print("\nAEMO Alarm Budget and Negative Control")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()