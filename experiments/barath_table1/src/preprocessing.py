
import numpy as np


def preprocess_synthetic(values, train_size=4000, period=48):

    values = np.asarray(values, dtype=float)

    if train_size <= 0 or train_size > len(values):
        raise ValueError("Invalid training size")

    if train_size < period:
        raise ValueError("Training size must cover one full period")

    # Use the training part of the data
    train_values = values[:train_size]

    # Average demand at each point in the daily cycle
    seasonal_pattern = np.zeros(period)

    for i in range(period):
        seasonal_pattern[i] = np.mean(
            train_values[i::period]
        )

    # Subtract the daily pattern
    positions = np.arange(len(values)) % period
    residuals = values - seasonal_pattern[positions]

    # Mean and standard deviation from training data
    train_residuals = residuals[:train_size]

    mean = np.mean(train_residuals)
    std = np.std(train_residuals)

    if std == 0:
        std = 1.0

    # Scale the residuals
    processed_values = (residuals - mean) / std

    return processed_values
