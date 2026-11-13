"""
This code calculates the evaluation metrics for Table 5.
I measure prediction interval coverage, interval width and worst 24-hour coverage.
The metrics are calculated separately for each drift regime.
"""

import numpy as np
import pandas as pd


# Check the actual values and prediction intervals
def check_inputs(actual, lower, upper):

    actual = np.asarray(actual, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if not (len(actual) == len(lower) == len(upper)):
        raise ValueError("Input arrays must have the same length")

    if len(actual) == 0:
        raise ValueError("Input arrays are empty")

    if not np.all(np.isfinite(actual)):
        raise ValueError("Actual values contain invalid numbers")

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Prediction intervals contain invalid numbers")

    if np.any(lower > upper):
        raise ValueError("Lower bound cannot exceed upper bound")

    return actual, lower, upper


# Calculate the fraction of actual values inside the intervals
def calculate_coverage(actual, lower, upper):

    actual, lower, upper = check_inputs(actual, lower, upper)

    covered = (actual >= lower) & (actual <= upper)

    return float(np.mean(covered))


# Calculate the average prediction interval width
def calculate_mean_width(lower, upper):

    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if len(lower) == 0 or len(lower) != len(upper):
        raise ValueError("Invalid interval arrays")

    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Prediction intervals contain invalid numbers")

    if np.any(lower > upper):
        raise ValueError("Lower bound cannot exceed upper bound")

    return float(np.mean(upper - lower))


# Normalise interval width using mean training demand
def calculate_normalised_width(lower, upper, train_mean):

    if not np.isfinite(train_mean) or train_mean <= 0:
        raise ValueError("Training mean must be positive")

    mean_width = calculate_mean_width(lower, upper)

    return float(mean_width / train_mean)


# Find the lowest coverage in complete 24-hour windows
def calculate_worst_24h(actual, lower, upper, regimes,
                        timestamps, selected_regime, window=48):

    actual, lower, upper = check_inputs(actual, lower, upper)

    regimes = np.asarray(regimes).astype(str)
    timestamps = pd.to_datetime(timestamps)

    if len(regimes) != len(actual) or len(timestamps) != len(actual):
        raise ValueError("Regimes and timestamps must match actual values")

    if window < 1:
        raise ValueError("Window size must be positive")

    if len(actual) < window:
        return float("nan"), 0

    # Mark observations covered by the prediction interval
    covered = ((actual >= lower) & (actual <= upper)).astype(int)

    # Mark observations belonging to the selected regime
    in_regime = (regimes == selected_regime).astype(int)

    # Count covered observations in each rolling window
    coverage_sum = np.cumsum(np.insert(covered, 0, 0))
    window_covered = coverage_sum[window:] - coverage_sum[:-window]

    # Count regime observations in each rolling window
    regime_sum = np.cumsum(np.insert(in_regime, 0, 0))
    window_regime = regime_sum[window:] - regime_sum[:-window]

    # Check that timestamps are 30 minutes apart
    time_differences = pd.Series(timestamps).diff()
    correct_step = (
        time_differences == pd.Timedelta(minutes=30)
    ).to_numpy().astype(int)

    # Each complete window needs window - 1 valid time differences
    step_sum = np.cumsum(np.insert(correct_step, 0, 0))
    window_steps = step_sum[window:] - step_sum[1:len(actual) - window + 2]

    # Keep only complete windows within the selected regime
    valid_windows = (
        (window_regime == window) &
        (window_steps == window - 1)
    )

    if not np.any(valid_windows):
        return float("nan"), 0

    coverages = window_covered[valid_windows] / window

    return float(np.min(coverages)), int(np.sum(valid_windows))


# Calculate metrics for one drift regime
def calculate_regime_metrics(actual, lower, upper, regimes,
                             timestamps, selected_regime, train_mean):

    actual, lower, upper = check_inputs(actual, lower, upper)
    regimes = np.asarray(regimes).astype(str)

    if len(regimes) != len(actual):
        raise ValueError("Regime labels must match actual values")

    mask = regimes == selected_regime

    if not np.any(mask):
        return None

    coverage = calculate_coverage(
        actual[mask], lower[mask], upper[mask]
    )

    mean_width = calculate_mean_width(
        lower[mask], upper[mask]
    )

    normalised_width = calculate_normalised_width(
        lower[mask], upper[mask], train_mean
    )

    worst_24h, n_windows = calculate_worst_24h(
        actual, lower, upper, regimes, timestamps, selected_regime
    )

    return {
        "coverage": coverage,
        "mean_interval_width": mean_width,
        "normalised_interval_width": normalised_width,
        "worst_24h_coverage": worst_24h,
        "n_complete_windows": n_windows,
        "n_observations": int(np.sum(mask))
    }


# Test the calculations using known values
if __name__ == "__main__":

    actual = np.array([100, 110, 120, 130, 140])
    lower = np.array([90, 100, 110, 120, 130])
    upper = np.array([110, 120, 130, 140, 150])

    print("Coverage:", calculate_coverage(actual, lower, upper))
    print("Mean interval width:", calculate_mean_width(lower, upper))
    print("Normalised width:", calculate_normalised_width(lower, upper, 120))

    # Test a complete 24-hour window with 48 observations
    test_actual = np.full(48, 100.0)
    test_lower = np.full(48, 90.0)
    test_upper = np.full(48, 110.0)

    test_regimes = np.array(["pre_drift"] * 48)
    test_times = pd.date_range(
        "2020-03-01", periods=48, freq="30min"
    )

    worst, count = calculate_worst_24h(
        test_actual, test_lower, test_upper,
        test_regimes, test_times, "pre_drift"
    )

    print("Worst 24-hour coverage:", worst)
    print("Complete 24-hour windows:", count)
