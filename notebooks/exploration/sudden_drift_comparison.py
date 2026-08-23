from pathlib import Path
import pandas as pd
from drift_forecasting.detection.adwin import ADWINDetector
from drift_forecasting.detection.kswin import KSWINDetector
from drift_forecasting.detection.page_hinkley import PageHinkleyDetector

# Locate the project root from this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "synthetic"
    / "ac_demand_abrupt_drift.csv"
)

# The synthetic CSV has no header.
# Column 0 = sample index
# Column 1 = demand value
df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=["index", "demand"],
)

print("Dataset loaded successfully.")
print()
print("Shape:", df.shape)
print()
print("First 5 rows:")
print(df.head())
print()
print("Last 5 rows:")
print(df.tail())


#show 
import matplotlib.pyplot as plt


TRUE_CHANGEPOINT = 2000

plt.figure(figsize=(14, 5))

plt.plot(
    df["index"],
    df["demand"],
    linewidth=0.8,
    label="Synthetic demand"
)

plt.axvline(
    TRUE_CHANGEPOINT,
    linestyle="--",
    linewidth=2,
    label="True changepoint = 2000"
)

plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.title("Synthetic Sudden Drift Dataset")
plt.legend()
plt.tight_layout()

plt.show()



#Local map to see the data more clearly (can delete)
plt.figure(figsize=(14, 5))

zoom_df = df[
    (df["index"] >= 1500) &
    (df["index"] <= 2500)
]

plt.plot(
    zoom_df["index"],
    zoom_df["demand"],
    linewidth=0.8,
    label="Synthetic demand"
)

plt.axvline(
    TRUE_CHANGEPOINT,
    linestyle="--",
    linewidth=2,
    label="True changepoint = 2000"
)

plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.title("Sudden Drift Around the True Changepoint")
plt.legend()
plt.tight_layout()

plt.show()

# ============================================================
# ADWIN sudden drift detection
# ============================================================

adwin_detector = ADWINDetector()

adwin_changepoints = adwin_detector.detect(
    df["demand"].to_numpy()
)

print()
print("========== ADWIN RESULT ==========")
print("True changepoint:", TRUE_CHANGEPOINT)
print("Detected changepoints:", adwin_changepoints)

adwin_false_alarms = [
    cp for cp in adwin_changepoints
    if cp < TRUE_CHANGEPOINT
]

adwin_after_drift = [
    cp for cp in adwin_changepoints
    if cp >= TRUE_CHANGEPOINT
]

if len(adwin_after_drift) > 0:
    adwin_detection = adwin_after_drift[0]
    adwin_delay = adwin_detection - TRUE_CHANGEPOINT
else:
    adwin_detection = None
    adwin_delay = None

print("False alarms before drift:", adwin_false_alarms)
print("Number of false alarms:", len(adwin_false_alarms))
print("Matched detection:", adwin_detection)
print("Detection delay:", adwin_delay)


# ============================================================
# Page-Hinkley sudden drift detection
# ============================================================

page_hinkley_detector = PageHinkleyDetector()

page_hinkley_changepoints = page_hinkley_detector.detect(
    df["demand"].to_numpy()
)

page_hinkley_false_alarms = [
    cp for cp in page_hinkley_changepoints
    if cp < TRUE_CHANGEPOINT
]

page_hinkley_after_drift = [
    cp for cp in page_hinkley_changepoints
    if cp >= TRUE_CHANGEPOINT
]

if len(page_hinkley_after_drift) > 0:
    page_hinkley_detection = page_hinkley_after_drift[0]
    page_hinkley_delay = (
        page_hinkley_detection - TRUE_CHANGEPOINT
    )
else:
    page_hinkley_detection = None
    page_hinkley_delay = None

print()
print("========== PAGE-HINKLEY RESULT ==========")
print("True changepoint:", TRUE_CHANGEPOINT)
print("Detected changepoints:", page_hinkley_changepoints)
print(
    "False alarms before drift:",
    page_hinkley_false_alarms
)
print(
    "Number of false alarms:",
    len(page_hinkley_false_alarms)
)
print(
    "Matched detection:",
    page_hinkley_detection
)
print(
    "Detection delay:",
    page_hinkley_delay
)


# ============================================================
# KSWIN sudden drift detection
# ============================================================

kswin_detector = KSWINDetector()

kswin_changepoints = kswin_detector.detect(
    df["demand"].to_numpy()
)

print()
print("========== KSWIN RESULT ==========")
print("True changepoint:", TRUE_CHANGEPOINT)
print("Detected changepoints:", kswin_changepoints)

# Detections before the real drift are false alarms
kswin_false_alarms = [
    cp for cp in kswin_changepoints
    if cp < TRUE_CHANGEPOINT
]

# Detections occurring at/after the real drift
kswin_after_drift = [
    cp for cp in kswin_changepoints
    if cp >= TRUE_CHANGEPOINT
]

print("False alarms before drift:", kswin_false_alarms)
print("Number of false alarms:", len(kswin_false_alarms))

if len(kswin_after_drift) > 0:
    kswin_detection = kswin_after_drift[0]
    kswin_delay = kswin_detection - TRUE_CHANGEPOINT

    print("Matched detection:", kswin_detection)
    print("Detection delay:", kswin_delay)
else:
    kswin_detection = None
    kswin_delay = None

    print("KSWIN missed the sudden drift.")


# ============================================================
# Comparison visualisation
# ============================================================


# Focus on the area around the true sudden drift
plot_df = df[
    (df["index"] >= 1500) &
    (df["index"] <= 2500)
]

plt.figure(figsize=(14, 6))

plt.plot(
    plot_df["index"],
    plot_df["demand"],
    linewidth=0.8,
    label="Synthetic demand"
)

# Ground truth
plt.axvline(
    TRUE_CHANGEPOINT,
    linestyle="--",
    linewidth=2,
    label="True changepoint (2000)"
)

# ADWIN detection
plt.axvline(
    adwin_detection,
    linestyle=":",
    linewidth=2,
    label="ADWIN detection (2047)"
)

# KSWIN detection
plt.axvline(
    kswin_detection,
    linestyle="-.",
    linewidth=2,
    label="KSWIN detection (2031)"
)
# page_hinkley_detection
plt.axvline(
    page_hinkley_detection,
    linestyle=(0, (3, 1, 1, 1)),
    linewidth=2,
    label=f"Page-Hinkley detection ({page_hinkley_detection})"
)

plt.xlabel("Time Index")
plt.ylabel("Demand")
plt.title(
    "Sudden Drift Detection: ADWIN vs KSWIN vs Page-Hinkley"
)

plt.legend()
plt.tight_layout()
plt.savefig(
    "reports/figures/sudden_drift_detection_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nDetection figure saved to:")
print("reports/figures/sudden_drift_detection_comparison.png")
plt.show()


# ============================================================
# Performance comparison table
# ============================================================

# False alarms per 10,000 observations before the true drift
adwin_false_alarm_rate = (
    len(adwin_false_alarms) / TRUE_CHANGEPOINT
) * 10000

kswin_false_alarm_rate = (
    len(kswin_false_alarms) / TRUE_CHANGEPOINT
) * 10000

page_hinkley_false_alarm_rate = (
    len(page_hinkley_false_alarms) / TRUE_CHANGEPOINT
) * 10000

comparison = pd.DataFrame({
    "Detector": [
        "ADWIN",
        "KSWIN",
        "Page-Hinkley"
    ],

    "True Changepoint": [
        TRUE_CHANGEPOINT,
        TRUE_CHANGEPOINT,
        TRUE_CHANGEPOINT
    ],

    "Detected Changepoint": [
        adwin_detection,
        kswin_detection,
        page_hinkley_detection
    ],

    "Detection Delay": [
        adwin_delay,
        kswin_delay,
        page_hinkley_delay
    ],

    "Pre-drift False Alarms": [
        len(adwin_false_alarms),
        len(kswin_false_alarms),
        len(page_hinkley_false_alarms)
    ],

    "False Alarms per 10k": [
        adwin_false_alarm_rate,
        kswin_false_alarm_rate,
        page_hinkley_false_alarm_rate
    ],

    "Missed": [
        "No" if adwin_detection is not None else "Yes",
        "No" if kswin_detection is not None else "Yes",
        "No" if page_hinkley_detection is not None else "Yes"
    ]
})

print()
print("========== SUDDEN DRIFT COMPARISON ==========")
print(comparison.to_string(index=False))

# Save comparison results
comparison.to_csv(
    "reports/results/sudden_drift_comparison.csv",
    index=False
)

print("\nComparison table saved to:")
print("reports/results/sudden_drift_comparison.csv")