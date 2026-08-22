import numpy as np

from drift_forecasting.detection.page_hinkley import PageHinkleyDetector
from drift_forecasting.detection.adwin import ADWINDetector
from drift_forecasting.detection.kswin import KSWINDetector


# Reproducible synthetic gradual drift
rng = np.random.default_rng(42)

# Stable behaviour for the first 1,000 observations
stable = rng.normal(0, 0.5, 1000)

# Mean gradually changes from 0 to 5
gradual_mean = np.linspace(0, 5, 1000)
gradual = rng.normal(gradual_mean, 0.5)

stream = np.concatenate([stable, gradual])

# For this temporary benchmark, gradual drift begins here
true_changepoint = 1000


detectors = {
    "Page-Hinkley": PageHinkleyDetector(),
    "ADWIN": ADWINDetector(),
    "KSWIN": KSWINDetector(),
}


print("\nGRADUAL DRIFT DETECTOR COMPARISON")
print("=" * 78)

print(
    f"{'Detector':<18}"
    f"{'First Detection':<18}"
    f"{'Delay':<12}"
    f"{'False Alarms':<15}"
    f"{'Missed':<10}"
)

print("-" * 78)


for name, detector in detectors.items():
    detected = detector.detect(stream)

    # Detections before the true drift starts
    false_alarms = [index for index in detected if index < true_changepoint]

    # Detections occurring after drift starts
    valid_detections = [
        index for index in detected
        if index >= true_changepoint
    ]

    if valid_detections:
        first_detection = valid_detections[0]
        delay = first_detection - true_changepoint
        missed = 0
    else:
        first_detection = None
        delay = None
        missed = 1

    print(
        f"{name:<18}"
        f"{str(first_detection):<18}"
        f"{str(delay):<12}"
        f"{len(false_alarms):<15}"
        f"{missed:<10}"
    )