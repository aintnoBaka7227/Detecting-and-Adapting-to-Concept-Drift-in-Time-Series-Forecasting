"""
Remove yearly and weekly seasonal patterns from AEMO demand.
Fit a multiplicative seasonal profile using training data only.
Standardise the remaining residuals using training statistics.
"""

import numpy as np
import pandas as pd

from aemo_preprocessing import load_aemo_data, split_aemo_data


def add_time_features(data):
    data = data.copy()
    dates = data["SETTLEMENTDATE"]

    # Identify the day, weekday and half-hour period
    data["weekday"] = dates.dt.dayofweek
    data["time_slot"] = dates.dt.hour * 2 + dates.dt.minute // 30
    data["day_of_year"] = pd.to_datetime(
        "2020-" + dates.dt.strftime("%m-%d")
    ).dt.dayofyear

    return data


def fit_seasonal_profile(train):
    train = add_time_features(train)

    # Calculate a baseline and the typical weekly pattern
    base_demand = train["TOTALDEMAND"].mean()
    if base_demand <= 0:
        raise ValueError("Training mean demand must be positive")

    weekly_average = train.groupby(
        ["weekday", "time_slot"]
    )["TOTALDEMAND"].mean()
    weekly_profile = weekly_average / base_demand

    weekly_values = pd.MultiIndex.from_frame(
        train[["weekday", "time_slot"]]
    ).map(weekly_profile).to_numpy(dtype=float)

    # Estimate the yearly factor after accounting for the weekly pattern
    train = train.copy()
    train["yearly_ratio"] = (
        train["TOTALDEMAND"].to_numpy(dtype=float)
        / (base_demand * weekly_values)
    )
    yearly_profile = train.groupby(
        "day_of_year"
    )["yearly_ratio"].mean()

    yearly_profile = yearly_profile.reindex(range(1, 367))
    yearly_profile = yearly_profile.interpolate(limit_direction="both")

    # Smooth the annual factor over 31 days, wrapping around year-end
    yearly_values = yearly_profile.to_numpy(dtype=float)
    extended = np.concatenate([
        yearly_values[-15:], yearly_values, yearly_values[:15]
    ])
    smoothed = pd.Series(extended).rolling(
        window=31, center=True
    ).mean().iloc[15:-15].to_numpy()
    yearly_profile = pd.Series(smoothed, index=range(1, 367))

    # Calculate training residuals for standardisation
    yearly_train = train["day_of_year"].map(
        yearly_profile
    ).to_numpy(dtype=float)
    expected = base_demand * weekly_values * yearly_train
    residuals = train["TOTALDEMAND"].to_numpy(dtype=float) - expected

    train_mean = residuals.mean()
    train_std = residuals.std()
    if train_std == 0 or not np.isfinite(train_std):
        raise ValueError("Invalid training standard deviation")

    return base_demand, weekly_profile, yearly_profile, train_mean, train_std


def transform_data(data, profile):
    base_demand, weekly_profile, yearly_profile, train_mean, train_std = profile
    data = add_time_features(data)

    # Apply the same frozen seasonal factors to every data split
    weekly_values = pd.MultiIndex.from_frame(
        data[["weekday", "time_slot"]]
    ).map(weekly_profile).to_numpy(dtype=float)
    yearly_values = data["day_of_year"].map(
        yearly_profile
    ).to_numpy(dtype=float)

    data["seasonal_expected"] = (
        base_demand * weekly_values * yearly_values
    )
    data["residual"] = data["TOTALDEMAND"] - data["seasonal_expected"]
    data["standardised_residual"] = (
        data["residual"] - train_mean
    ) / train_std

    if not np.isfinite(data["standardised_residual"]).all():
        raise ValueError("Invalid standardised residual values")

    return data


def main():
    datasets = {"NSW1": "NSW.csv", "SA1": "SA.csv"}

    for region, filename in datasets.items():
        data = load_aemo_data(filename, region)
        train, calibration, test = split_aemo_data(data)

        # Fit only on 2018-2019 training observations
        profile = fit_seasonal_profile(train)
        train = transform_data(train, profile)
        calibration = transform_data(calibration, profile)
        test = transform_data(test, profile)

        print(f"\nRegion: {region}")
        print(f"Training residual mean: {train['standardised_residual'].mean():.4f}")
        print(f"Training residual std: {train['standardised_residual'].std(ddof=0):.4f}")
        print(f"Calibration observations: {len(calibration)}")
        print(f"Testing observations: {len(test)}")
        print(f"Missing test residuals: {test['standardised_residual'].isna().sum()}")


if __name__ == "__main__":
    main()
