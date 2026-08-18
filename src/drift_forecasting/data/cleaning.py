"""Check and safely clean AEMO NSW1 and SA1 native-frequency datasets."""

import pandas as pd

from drift_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR


REQUIRED_COLUMNS = {
    "REGION",
    "SETTLEMENTDATE",
    "TOTALDEMAND",
    "RRP",
    "PERIODTYPE",
}

# AEMO changed from 30-minute to 5-minute settlement on 1 October 2021.
FIVE_MINUTE_START = pd.Timestamp("2021-10-01")


def clean_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """Run data-quality checks and safely clean one AEMO regional dataset."""

    data = df.copy()

    print("=== AEMO Data Quality Check ===")
    print(f"Rows before cleaning: {len(data)}")

    # 1. Check that the dataset contains all columns expected from loader.py.
    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Convert demand and price to numeric so invalid values become NaN
    # and can be identified in the missing-value check.
    data["TOTALDEMAND"] = pd.to_numeric(
        data["TOTALDEMAND"],
        errors="coerce",
    )
    data["RRP"] = pd.to_numeric(
        data["RRP"],
        errors="coerce",
    )

    print(f"Regions: {sorted(data['REGION'].dropna().unique())}")

    print()

    # 2. Check timestamp quality and chronological order.
    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"],
        errors="coerce",
    )

    invalid_timestamps = data["SETTLEMENTDATE"].isna().sum()

    print(f"Invalid timestamps: {invalid_timestamps}")

    # Invalid timestamps cannot be placed correctly in the time series.
    data = (
        data
        .dropna(subset=["SETTLEMENTDATE"])
        .reset_index(drop=True)
    )

    # Check the original ordering before sorting the dataset.
    out_of_order = (
        data["SETTLEMENTDATE"].diff() < pd.Timedelta(0)
    ).sum()

    print(f"Out-of-order timestamps: {out_of_order}")

    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    print(
        "Date range:",
        data["SETTLEMENTDATE"].min(),
        "to",
        data["SETTLEMENTDATE"].max(),
    )

    print()

    # 3. Check for exact duplicate rows and repeated timestamps.
    exact_duplicates = data.duplicated().sum()

    print(f"Exact duplicate rows: {exact_duplicates}")

    # Exact duplicates contain no additional information.
    data = (
        data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # After removing exact duplicates, repeated timestamps indicate that
    # different values exist for the same region and settlement interval.
    conflicting_duplicates = data.duplicated(
        subset=["SETTLEMENTDATE", "REGION"],
        keep=False,
    )

    print(
        "Conflicting duplicate timestamp rows:",
        conflicting_duplicates.sum(),
    )

    # Keep one observation for each region and settlement interval.
    data = (
        data
        .drop_duplicates(
            subset=["SETTLEMENTDATE", "REGION"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    print()

    # 4. Check for missing values in the columns used by the project.
    print("Missing values:")

    print(
        data[
            [
                "REGION",
                "SETTLEMENTDATE",
                "TOTALDEMAND",
                "RRP",
                "PERIODTYPE",
            ]
        ].isna().sum()
    )

    print()

    # 5. Check whether observations follow the expected AEMO frequency.
    # Before October 2021 the data should be 30-minute intervals.
    before_5ms = data[
        data["SETTLEMENTDATE"] < FIVE_MINUTE_START
    ]

    # From October 2021 onward the data should be 5-minute intervals.
    after_5ms = data[
        data["SETTLEMENTDATE"] >= FIVE_MINUTE_START
    ]

    print("30-minute period:")

    _check_frequency(
        before_5ms,
        expected_interval=pd.Timedelta(minutes=30),
    )

    print()

    print("5-minute period:")

    _check_frequency(
        after_5ms,
        expected_interval=pd.Timedelta(minutes=5),
    )

    print()

    # 6. Check which AEMO period types are present in the combined dataset.
    print("PERIODTYPE values:")
    print(data["PERIODTYPE"].value_counts(dropna=False))

    print()

    # 7. Check basic logical ranges for electricity demand and price.
    # Negative demand is suspicious, while negative or extreme RRP values
    # may be genuine electricity-market observations and are not removed.
    negative_demand = (
        data["TOTALDEMAND"] < 0
    ).sum()

    print(
        f"Negative TOTALDEMAND values: {negative_demand}"
    )

    print(
        f"TOTALDEMAND range: "
        f"{data['TOTALDEMAND'].min()} "
        f"to {data['TOTALDEMAND'].max()}"
    )

    print(
        f"RRP range: "
        f"{data['RRP'].min()} "
        f"to {data['RRP'].max()}"
    )

    print()

    # 8. Final check that the cleaned dataset is chronologically ordered.
    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    print("=== Final Quality Check ===")
    print(f"Rows after cleaning: {len(data)}")

    print(
        "Sorted by time:",
        data["SETTLEMENTDATE"].is_monotonic_increasing,
    )

    return data


def _check_frequency(
    df: pd.DataFrame,
    expected_interval: pd.Timedelta,
) -> None:
    """Check for gaps that do not match the expected sampling frequency."""

    timestamps = (
        df["SETTLEMENTDATE"]
        .drop_duplicates()
        .sort_values()
    )

    # Calculate the time difference between consecutive observations.
    differences = timestamps.diff().dropna()

    # Any difference other than the expected interval represents either
    # a missing interval or another irregularity in the time series.
    unexpected = differences[
        differences != expected_interval
    ]

    print(f"Expected interval: {expected_interval}")
    print(f"Unexpected intervals: {len(unexpected)}")

    if not unexpected.empty:
        print("Examples:")
        print(unexpected.head(10))

def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise the full dataset to 30-minute intervals."""

    data = df.copy()

    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"]
    )

    data = (
        data
        .set_index("SETTLEMENTDATE")
        .groupby("REGION")
        .resample(
            "30min",
            label="right",
            closed="right",
        )
        .agg(
            {
                "TOTALDEMAND": "mean",
                "RRP": "mean",
            }
        )
        .reset_index()
    )

    return data


if __name__ == "__main__":

    raw_files = {
        "NSW1": RAW_DATA_DIR / "NSW1_201801_202312_native.csv",
        "SA1": RAW_DATA_DIR / "SA1_201801_202312_native.csv",
    }

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for region, raw_file in raw_files.items():

        print(f"\nChecking {region}")

        df = pd.read_csv(raw_file)

        # Run quality checks and safe cleaning first.
        cleaned_df = clean_demand_series(df)

        print()

        # Standardise the entire time series to 30-minute intervals.
        resampled_df = resample_to_30min(cleaned_df)

        print(
            f"Rows after 30-minute resampling: "
            f"{len(resampled_df)}"
        )

        output_file = (
            PROCESSED_DATA_DIR
            / f"{region}_201801_202312_cleaned_30min.csv"
        )

        resampled_df.to_csv(
            output_file,
            index=False,
        )

        print(f"Saved: {output_file}")