"""
This script runs ADWIN, KSWIN and Page-Hinkley on AEMO demand data.
It uses the detector settings selected from the synthetic experiments.
Detectors are warmed up before testing, and test alarms are saved to CSV files.
"""

import sys
from pathlib import Path

import pandas as pd


# Set the project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"
OUTPUT_DIR = PROJECT_DIR / "outputs"

sys.path.insert(0, str(SRC_DIR))

from aemo_preprocessing import load_aemo_data, split_aemo_data
from deseasonalise import fit_seasonal_profile, transform_data
from aemo_detectors import create_detector


# Use the parameters selected in Table 1
DETECTOR_PARAMS = {
    "ADWIN": {
        "delta": 0.000001,
        "clock": 16,
        "min_window_length": 5
    },
    "KSWIN": {
        "alpha": 0.00005,
        "window_size": 400,
        "stat_size": 50,
        "seed": 42
    },
    "Page-Hinkley": {
        "threshold": 300,
        "delta": 0.0001,
        "alpha": 0.9995,
        "min_instances": 20
    }
}


def run_aemo_detector(train, calibration, test, name, params):
    # Create the detector using Table 1 parameters
    detector = create_detector(name, params)

    # Warm up the detector using training data
    for value in train["standardised_residual"]:
        detector.update(float(value))

    # Continue processing the calibration period
    for value in calibration["standardised_residual"]:
        detector.update(float(value))

    # Record alarms during the testing period
    alarms = []

    for index, row in test.iterrows():
        detector.update(float(row["standardised_residual"]))

        if detector.drift_detected:
            alarms.append({
                "timestamp": row["SETTLEMENTDATE"],
                "test_index": index,
                "demand": row["TOTALDEMAND"],
                "standardised_residual": row["standardised_residual"]
            })

    return pd.DataFrame(
        alarms,
        columns=[
            "timestamp",
            "test_index",
            "demand",
            "standardised_residual"
        ]
    )


def main():
    # Create the output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {
        "NSW1": "NSW.csv",
        "SA1": "SA.csv"
    }

    for region, filename in datasets.items():
        # Load and prepare the AEMO data
        data = load_aemo_data(filename, region)
        train, calibration, test = split_aemo_data(data)

        profile = fit_seasonal_profile(train)

        train = transform_data(train, profile)
        calibration = transform_data(calibration, profile)
        test = transform_data(test, profile)

        print(f"\nRegion: {region}")

        # Run each detector on the same prepared data
        for name, params in DETECTOR_PARAMS.items():
            alarms = run_aemo_detector(
                train, calibration, test, name, params
            )

            detector_name = name.lower().replace("-", "_")

            output_file = OUTPUT_DIR / (
                f"{region.lower()}_{detector_name}_raw_alarms.csv"
            )

            alarms.to_csv(output_file, index=False)

            print(f"{name}: {len(alarms)} raw alarms")

    print("\nAEMO drift detection completed.")


if __name__ == "__main__":
    main()
