import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river.datasets import synth

from drift_forecasting.detection.adwin import ADWINDetector
from drift_forecasting.detection.kswin import KSWINDetector
from drift_forecasting.detection.page_hinkley import PageHinkleyDetector


# ============================================================
# Benchmark configuration
# ============================================================

NUM_SAMPLES = 20000
TRUE_CHANGEPOINT = 2000

# Five independent synthetic runs
SEEDS = [1, 2, 3, 4, 5]

parameter_text = {
    "ADWIN": "delta=0.002",
    "KSWIN": "alpha=0.005; window=100; stat=30",
    "Page-Hinkley": "delta=0.005; threshold=50; alpha=0.9999; mode=both",
}

RESULTS_DIR = Path("reports/results")
FIGURES_DIR = Path("reports/figures")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Synthetic sudden-drift generator
# Matches the team's current generator:
# Friedman stream -> 50% target increase at index 2000
# ============================================================

class SummerFriedman(synth.Friedman):
    def __iter__(self):
        for x, y in super().__iter__():
            yield x, y * 1.5


def generate_sudden_stream(seed: int) -> np.ndarray:
    """
    Generate one synthetic sudden-drift series.

    Before index 2000:
        Normal Friedman target.

    From index 2000:
        Target increases by 50%.
    """
    normal_stream = iter(synth.Friedman(seed=seed))
    summer_stream = iter(SummerFriedman(seed=seed))

    values = []

    for i in range(NUM_SAMPLES):
        if i < TRUE_CHANGEPOINT:
            _, y = next(normal_stream)
        else:
            _, y = next(summer_stream)

        values.append(float(y))

    return np.asarray(values)

def generate_no_drift_stream(seed: int) -> np.ndarray:
    """
    Generate a clean synthetic stream with no concept drift.
    """
    normal_stream = iter(synth.Friedman(seed=seed))

    values = []

    for _ in range(NUM_SAMPLES):
        _, y = next(normal_stream)
        values.append(float(y))

    return np.asarray(values)
# ============================================================
# Evaluation
# ============================================================

def evaluate_detector(
    detector_name: str,
    detector,
    stream: np.ndarray,
    seed: int,
) -> dict:

    changepoints = detector.detect(stream)

    # Anything before the known drift is a false alarm
    false_alarms = [
        cp for cp in changepoints
        if cp < TRUE_CHANGEPOINT
    ]

    # First detection after the true drift is the matched detection
    post_drift = [
        cp for cp in changepoints
        if cp >= TRUE_CHANGEPOINT
    ]

    if post_drift:
        matched_detection = post_drift[0]
        detection_delay = matched_detection - TRUE_CHANGEPOINT
        missed = 0
    else:
        matched_detection = np.nan
        detection_delay = np.nan
        missed = 1

    # Normalise false alarms to 10,000 pre-drift observations
    false_alarms_per_10k = (
        len(false_alarms) / TRUE_CHANGEPOINT
    ) * 10000

    return {
        "Detector": detector_name,
        "Drift Type": "sudden",
        "Seed": seed,
        "True Changepoint": TRUE_CHANGEPOINT,
        "Detected Changepoint": matched_detection,
        "Detection Delay": detection_delay,
        "Pre-drift False Alarms": len(false_alarms),
        "False Alarms per 10k": false_alarms_per_10k,
        "Missed": missed,
    }


# ============================================================
# Run five-seed benchmark
# ============================================================

all_results = []

for seed in SEEDS:

    print()
    print("=" * 60)
    print(f"Running synthetic sudden drift benchmark - seed {seed}")
    print("=" * 60)

    # Reproducibility for any Python-random operations
    random.seed(seed)

    stream = generate_sudden_stream(seed)

    detectors = [
        (
            "ADWIN",
            ADWINDetector(delta=0.002),
        ),
        (
            "KSWIN",
            KSWINDetector(
                alpha=0.005,
                window_size=100,
                stat_size=30,
                seed=seed,
            ),
        ),
        (
            "Page-Hinkley",
            PageHinkleyDetector(
                min_instances=30,
                delta=0.005,
                threshold=50.0,
                alpha=0.9999,
                mode="both",
            ),
        ),
    ]

    for detector_name, detector in detectors:

        result = evaluate_detector(
            detector_name,
            detector,
            stream,
            seed,
        )

        all_results.append(result)

        print(
            f"{detector_name:15s} "
            f"detected={result['Detected Changepoint']} "
            f"delay={result['Detection Delay']} "
            f"false alarms={result['Pre-drift False Alarms']} "
            f"missed={result['Missed']}"
        )

# ============================================================
# No-drift control benchmark
# ============================================================

control_results = []

for seed in SEEDS:

    print()
    print("=" * 60)
    print(f"Running NO-DRIFT control benchmark - seed {seed}")
    print("=" * 60)

    stream = generate_no_drift_stream(seed)

    detectors = [
        (
            "ADWIN",
            ADWINDetector(delta=0.002),
        ),
        (
            "KSWIN",
            KSWINDetector(
                alpha=0.005,
                window_size=100,
                stat_size=30,
                seed=seed,
            ),
        ),
        (
            "Page-Hinkley",
            PageHinkleyDetector(
                min_instances=30,
                delta=0.005,
                threshold=50.0,
                alpha=0.9999,
                mode="both",
            ),
        ),
    ]

    for detector_name, detector in detectors:

        changepoints = detector.detect(stream)

        false_alarm_count = len(changepoints)

        false_alarms_per_10k = (
            false_alarm_count / NUM_SAMPLES
        ) * 10000

        control_results.append(
            {
                "Detector": detector_name,
                "Drift Type": "none",
                "Seed": seed,
                "False Alarms": false_alarm_count,
                "False Alarms per 10k": false_alarms_per_10k,
            }
        )

        print(
            f"{detector_name:15s} "
            f"false alarms={false_alarm_count} "
            f"false alarms/10k={false_alarms_per_10k:.1f}"
        )
# ============================================================
# Per-seed results
# ============================================================

raw_results = pd.DataFrame(all_results)

raw_output_path = (
    RESULTS_DIR / "sudden_drift_benchmark_by_seed.csv"
)

raw_results.to_csv(
    raw_output_path,
    index=False,
)

control_df = pd.DataFrame(control_results)

control_output_path = (
    RESULTS_DIR /
    "no_drift_control_by_seed.csv"
)

control_df.to_csv(
    control_output_path,
    index=False,
)

print()
print("No-drift control results:")
print(control_df.to_string(index=False))

print()
print("Per-seed results:")
print(raw_results.to_string(index=False))


# ============================================================
# Aggregate results for supervisor T1 table
# ============================================================

summary_rows = []
control_summary_rows = []

for detector_name in [
    "ADWIN",
    "KSWIN",
    "Page-Hinkley",
]:

    subset = control_df[
        control_df["Detector"] == detector_name
    ]

    mean_false_alarm_rate = subset[
        "False Alarms per 10k"
    ].mean()

    total_false_alarms = int(
        subset["False Alarms"].sum()
    )

    control_summary_rows.append(
        {
            "Detector": detector_name,
            "Drift Type": "none",
            "Delay (mean ± sd)": "N/A",
            "False Alarms / 10k":
                mean_false_alarm_rate,
            "Missed": "N/A",
            "Threshold / Parameters":
                parameter_text[detector_name],
            "Seeds": len(SEEDS),
            "Total False Alarms":
                total_false_alarms,
        }
    )


control_summary = pd.DataFrame(
    control_summary_rows
)

parameter_text = {
    "ADWIN": "delta=0.002",
    "KSWIN": "alpha=0.005; window=100; stat=30",
    "Page-Hinkley":
        "delta=0.005; threshold=50; alpha=0.9999; mode=both",
}

for detector_name in [
    "ADWIN",
    "KSWIN",
    "Page-Hinkley",
]:

    subset = raw_results[
        raw_results["Detector"] == detector_name
    ]

    valid_delays = subset[
        "Detection Delay"
    ].dropna()

    mean_delay = valid_delays.mean()

    # Sample standard deviation across seeds
    sd_delay = valid_delays.std(ddof=1)

    mean_false_alarm_rate = subset[
        "False Alarms per 10k"
    ].mean()

    total_missed = int(
        subset["Missed"].sum()
    )

    summary_rows.append(
        {
            "Detector": detector_name,
            "Drift Type": "sudden",
            "Delay (mean ± sd)": (
                f"{mean_delay:.1f} ± {sd_delay:.1f}"
            ),
            "Mean Delay": mean_delay,
            "Delay SD": sd_delay,
            "False Alarms / 10k":
                mean_false_alarm_rate,
            "Missed": total_missed,
            "Threshold / Parameters":
                parameter_text[detector_name],
            "Seeds": len(SEEDS),
        }
    )


summary = pd.DataFrame(summary_rows)

summary_output_path = (
    RESULTS_DIR /
    "sudden_drift_benchmark_summary.csv"
)

summary.to_csv(
    summary_output_path,
    index=False,
)

print()
print("=" * 60)

final_t1 = pd.concat(
    [
        summary[[
            "Detector",
            "Drift Type",
            "Delay (mean ± sd)",
            "False Alarms / 10k",
            "Missed",
            "Threshold / Parameters",
            "Seeds",
        ]],
        control_summary[[
            "Detector",
            "Drift Type",
            "Delay (mean ± sd)",
            "False Alarms / 10k",
            "Missed",
            "Threshold / Parameters",
            "Seeds",
        ]],
    ],
    ignore_index=True,
)

final_t1_output_path = (
    RESULTS_DIR /
    "synthetic_benchmark_T1.csv"
)

final_t1.to_csv(
    final_t1_output_path,
    index=False,
)

print("FINAL SYNTHETIC BENCHMARK TABLE")
print("=" * 60)

display_columns = [
    "Detector",
    "Drift Type",
    "Delay (mean ± sd)",
    "False Alarms / 10k",
    "Missed",
    "Threshold / Parameters",
    "Seeds",
]

print(
    final_t1.to_string(
        index=False
    )
)


# ============================================================
# Figure: mean sudden-drift detection delay
# ============================================================

plot_data = summary.copy()

x = np.arange(len(plot_data))

plt.figure(figsize=(9, 6))

bars = plt.bar(
    x,
    plot_data["Mean Delay"],
    yerr=plot_data["Delay SD"],
    capsize=6,
)

plt.xticks(
    x,
    plot_data["Detector"],
)

plt.ylabel(
    "Detection Delay (samples)"
)

plt.xlabel(
    "Drift Detector"
)

plt.title(
    "Synthetic Sudden Drift Detection Delay"
)

plt.grid(
    axis="y",
    alpha=0.25,
)

for bar, mean, sd in zip(
    bars,
    plot_data["Mean Delay"],
    plot_data["Delay SD"],
):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + sd + 2,
        f"{mean:.1f} ± {sd:.1f}",
        ha="center",
        va="bottom",
    )

plt.tight_layout()

figure_output_path = (
    FIGURES_DIR /
    "sudden_drift_mean_delay_comparison.png"
)

plt.savefig(
    figure_output_path,
    dpi=300,
    bbox_inches="tight",
)

plt.show()


print()
print("Saved:")
print(raw_output_path)
print(summary_output_path)
print(figure_output_path)
print(control_output_path)
print(final_t1_output_path)