import numpy as np

from drift_forecasting.synthetic.generator import make_series
from drift_forecasting.detection.page_hinkley import PageHinkleyDetector
from drift_forecasting.detection.adwin import ADWINDetector
from drift_forecasting.detection.kswin import KSWINDetector


DRIFT_TYPE = "gradual"
N = 20000
NOISE = 1.0
SEEDS = [1, 2, 3, 4, 5]


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


results = []


for detector_name, config in DETECTORS.items():

    delays = []
    false_alarm_rates = []
    missed = 0

    for seed in SEEDS:

        stream, changepoints = make_series(
            kind=DRIFT_TYPE,
            n=N,
            noise=NOISE,
            seed=seed,
        )

        true_changepoint = changepoints[0]

        detector = config["factory"]()
        detected = detector.detect(stream)

        false_alarms = [
            index
            for index in detected
            if index < true_changepoint
        ]

        valid_detections = [
            index
            for index in detected
            if index >= true_changepoint
        ]

        false_alarm_rate = (
            len(false_alarms) / true_changepoint
        ) * 10000

        false_alarm_rates.append(false_alarm_rate)

        if valid_detections:
            first_detection = valid_detections[0]
            delay = first_detection - true_changepoint
            delays.append(delay)
        else:
            missed += 1

    if delays:
        delay_mean = np.mean(delays)
        delay_sd = np.std(delays)
        delay_text = f"{delay_mean:.1f} ± {delay_sd:.1f}"
    else:
        delay_text = "N/A"

    mean_false_alarm_rate = np.mean(false_alarm_rates)

    results.append(
        {
            "detector": detector_name,
            "drift_type": DRIFT_TYPE,
            "delay": delay_text,
            "false_alarms_10k": mean_false_alarm_rate,
            "missed": missed,
            "threshold": config["threshold"],
            "seeds": len(SEEDS),
        }
    )


print("\nT1 — DETECTION ON THE SYNTHETIC BENCHMARK")
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