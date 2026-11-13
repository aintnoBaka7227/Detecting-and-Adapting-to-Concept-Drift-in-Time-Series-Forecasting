"""
This script loads and prepares the NSW1 and SA1 electricity demand datasets.
It checks for missing values, duplicate dates and incorrect time intervals.
It also separates the data into training, calibration and testing periods.
"""

from pathlib import Path

import pandas as pd


# Set the project and data folder paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"


def load_aemo_data(filename, region):
    # Load the original dataset
    file_path = DATA_DIR / filename
    data = pd.read_csv(file_path)

    # Check the required columns
    required_columns = ["REGION", "SETTLEMENTDATE", "TOTALDEMAND"]

    for column in required_columns:
        if column not in data.columns:
            raise ValueError(f"Missing column: {column}")

    # Convert the dates and demand values
    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"], errors="raise"
    )

    data["TOTALDEMAND"] = pd.to_numeric(
        data["TOTALDEMAND"], errors="raise"
    )

    # Sort the data by date
    data = data.sort_values("SETTLEMENTDATE").reset_index(drop=True)

    # Check the region
    if not data["REGION"].eq(region).all():
        raise ValueError(f"Incorrect region values in {filename}")

    # Check for missing values
    if data["SETTLEMENTDATE"].isna().any():
        raise ValueError(f"Missing dates in {filename}")

    if data["TOTALDEMAND"].isna().any():
        raise ValueError(f"Missing demand values in {filename}")

    # Check for duplicate timestamps
    if data["SETTLEMENTDATE"].duplicated().any():
        raise ValueError(f"Duplicate dates in {filename}")

    # Check the half-hourly intervals
    intervals = data["SETTLEMENTDATE"].diff().dropna()

    if not intervals.eq(pd.Timedelta(minutes=30)).all():
        raise ValueError(f"Incorrect time intervals in {filename}")

    # Count negative demand values without removing them
    negative_count = (data["TOTALDEMAND"] < 0).sum()

    if negative_count > 0:
        print(f"Negative demand observations in {region}: {negative_count}")

    return data


def split_aemo_data(data):
    # Training period - 2018 and 2019
    train = data[
        (data["SETTLEMENTDATE"] >= "2018-01-01")
        & (data["SETTLEMENTDATE"] < "2020-01-01")
    ].copy()

    # Calibration period - January and February 2020
    calibration = data[
        (data["SETTLEMENTDATE"] >= "2020-01-01")
        & (data["SETTLEMENTDATE"] < "2020-03-01")
    ].copy()

    # Testing period - March 2020 to December 2023
    test = data[
        (data["SETTLEMENTDATE"] >= "2020-03-01")
        & (data["SETTLEMENTDATE"] < "2024-01-01")
    ].copy()

    return train, calibration, test


def main():
    # Dataset filenames for both regions
    datasets = {
        "NSW1": "NSW.csv",
        "SA1": "SA.csv"
    }

    # Load and check each dataset
    for region, filename in datasets.items():
        data = load_aemo_data(filename, region)

        train, calibration, test = split_aemo_data(data)

        print(f"\nRegion: {region}")
        print(f"Total observations: {len(data)}")
        print(f"Start date: {data['SETTLEMENTDATE'].min()}")
        print(f"End date: {data['SETTLEMENTDATE'].max()}")
        print(f"Training observations: {len(train)}")
        print(f"Calibration observations: {len(calibration)}")
        print(f"Testing observations: {len(test)}")
        print(f"Missing demand values: {data['TOTALDEMAND'].isna().sum()}")
        print(f"Duplicate dates: {data['SETTLEMENTDATE'].duplicated().sum()}")


if __name__ == "__main__":
    main()
