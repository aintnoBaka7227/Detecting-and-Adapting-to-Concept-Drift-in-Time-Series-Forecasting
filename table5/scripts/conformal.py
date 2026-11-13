"""
This code implements Fixed Split Conformal Prediction for Table 5.
I use calibration errors to calculate a fixed radius for 90% prediction intervals.
The radius is then used to calculate lower and upper bounds for the forecasts.
"""

import numpy as np
import pandas as pd
import math


# Calculate the conformal radius using calibration errors
def calculate_radius(residuals, alpha=0.10):

    errors = np.abs(np.asarray(residuals, dtype=float))

    if len(errors) == 0:
        raise ValueError("Calibration residuals are empty")

    if not np.all(np.isfinite(errors)):
        raise ValueError("Calibration residuals contain invalid values")

    errors = np.sort(errors)

    n = len(errors)

    # Select the conformal quantile for 90% prediction intervals
    rank = math.ceil((n + 1) * (1 - alpha))

    if rank > n:
        return float("inf")

    radius = errors[rank - 1]

    return float(radius)


# Create prediction intervals using the fixed radius
def create_intervals(forecasts, radius):

    forecasts = np.asarray(forecasts, dtype=float)

    lower = forecasts - radius
    upper = forecasts + radius

    return lower, upper


# Load calibration residuals from a CSV file
def load_calibration(file_path):

    data = pd.read_csv(file_path)

    residuals = data["residual"].to_numpy()

    return residuals


# Test the conformal calculation using one calibration file
if __name__ == "__main__":

    file_path = (
        r"C:\Users\barath\Documents\capstone_team_main"
        r"\results\runs\a00fb3049b42"
        r"\calibration_forecast_aemo_SA1_1.csv"
    )

    residuals = load_calibration(file_path)

    radius = calculate_radius(residuals)

    print("Calibration observations:", len(residuals))
    print("Conformal radius:", round(radius, 4))
    print("Target coverage: 90%")
