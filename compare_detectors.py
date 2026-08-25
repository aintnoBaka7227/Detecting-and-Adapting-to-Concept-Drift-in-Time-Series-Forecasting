from pathlib import Path

import numpy as np
import pandas as pd

from drift_forecasting.detection.page_hinkley import PageHinkleyDetector
from drift_forecasting.detection.adwin import ADWINDetector
from drift_forecasting.detection.kswin import KSWINDetector


DRIFT_TYPE = "gradual"

CSV_DIR = Path(
    "src/drift_forecasting/data/csv_output"
)


DETECTORS = {
    "Page-Hinkley": {
        "factory": lambda: PageHinkleyDetector(),
        "threshold": "threshold = 50.0",
    },
    "ADWIN": {
        "factory": lambda: ADWINDetector(),
        "threshold": "δ = 0.002",
    },
    "KSWIN": {
        "factory": lambda: KSWINDetector(),
        "threshold": "α = 0.005",
    },
}


csv_files = sorted(
    CSV_DIR.glob("*gradual*.csv")
)


if len(csv_files) != 5:
    raise ValueError(
        f"Expected 5 gradual-drift CSV files, found {len(csv_files)}"
    )


results = []


for detector_name, config in DETECTORS.items():

    delays = []
    false_alarm_rates = []
    missed = 0

    for csv_file in csv_files:

        df = pd.read_csv(csv_file)

        stream = df["y"].to_numpy(dtype=float)

        true_changepoints = df.index[
            df["is_changepoint"] == 1
        ].tolist()

        if not true_changepoints:
            raise ValueError(
                f"No changepoint found in {csv_file.name}"
            )

        true_changepoint = true_changepoints[0]

        detector = config["factory"]()

        detected = detector.detect(stream)

        false_alarms = [
            cp
            for cp in detected
            if cp < true_changepoint
        ]

        valid_detections = [
            cp
            for cp in detected
            if cp >= true_changepoint
        ]

        false_alarm_rate = (
            len(false_alarms)
            / true_changepoint
        ) * 10000

        false_alarm_rates.append(false_alarm_rate)

        if valid_detections:

            first_detection = valid_detections[0]

            delay = (
                first_detection
                - true_changepoint
            )

            delays.append(delay)

        else:
            missed += 1


    if delays:

        delay_mean = np.mean(delays)

        # Sample standard deviation across seeds
        delay_sd = (
            np.std(delays, ddof=1)
            if len(delays) > 1
            else 0.0
        )

        delay_text = (
            f"{delay_mean:.1f} ± {delay_sd:.1f}"
        )

    else:
        delay_text = "N/A"


    false_alarm_mean = np.mean(
        false_alarm_rates
    )


    results.append(
        {
            "detector": detector_name,
            "drift_type": DRIFT_TYPE,
            "delay": delay_text,
            "false_alarms_10k": false_alarm_mean,
            "missed": missed,
            "threshold": config["threshold"],
            "seeds": len(csv_files),
        }
    )


print(
    "\nT1 — DETECTION ON THE SYNTHETIC BENCHMARK"
)

print("=" * 120)

print(
    f"{'detector':<18}"
    f"{'drift type':<15}"
    f"{'delay (mean ± sd)':<22}"
    f"{'false alarms / 10k':<22}"
    f"{'missed':<10}"
    f"{'threshold':<22}"
    f"{'seeds':<8}"
)

print("-" * 120)


for result in results:

    print(
        f"{result['detector']:<18}"
        f"{result['drift_type']:<15}"
        f"{result['delay']:<22}"
        f"{result['false_alarms_10k']:<22.2f}"
        f"{result['missed']:<10}"
        f"{result['threshold']:<22}"
        f"{result['seeds']:<8}"
    )