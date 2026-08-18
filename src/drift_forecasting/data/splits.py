"""Apply the frozen train / calibration / test split from config.SPLIT."""

import pandas as pd

from drift_forecasting.config import PROCESSED_DATA_DIR, SPLIT


def train_calibration_test_split(
    df: pd.DataFrame,
    date_col: str = "SETTLEMENTDATE",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, calibration, and test sets."""

    data = df.copy()

    # Convert timestamps and keep observations in chronological order.
    data[date_col] = pd.to_datetime(data[date_col])

    data = (
        data
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    # Read the fixed date boundaries from config.py.
    train_start, train_end = SPLIT["train"]
    calibration_start, calibration_end = SPLIT["calibration"]
    test_start, test_end = SPLIT["test"]

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)

    calibration_start = pd.Timestamp(calibration_start)
    calibration_end = pd.Timestamp(calibration_end)

    test_start = pd.Timestamp(test_start)

    # Train split.
    train = data[
        (data[date_col] >= train_start)
        & (data[date_col] < train_end + pd.Timedelta(days=1))
    ].copy()

    # Calibration split.
    calibration = data[
        (data[date_col] >= calibration_start)
        & (
            data[date_col]
            < calibration_end + pd.Timedelta(days=1)
        )
    ].copy()

    # Test split.
    test = data[
        data[date_col] >= test_start
    ].copy()

    if test_end is not None:
        test_end = pd.Timestamp(test_end)

        test = test[
            test[date_col]
            < test_end + pd.Timedelta(days=1)
        ].copy()

    return (
        train.reset_index(drop=True),
        calibration.reset_index(drop=True),
        test.reset_index(drop=True),
    )


if __name__ == "__main__":

    # Cleaned native-frequency files produced by cleaning.py.
    input_files = {
        "NSW1": (
            PROCESSED_DATA_DIR
            / "NSW1_201801_202312_cleaned_30min.csv"
        ),
        "SA1": (
            PROCESSED_DATA_DIR
            / "SA1_201801_202312_cleaned_30min.csv"
        ),
    }

    for region, input_file in input_files.items():

        print(f"\nSplitting {region}")

        df = pd.read_csv(input_file)

        train_df, calibration_df, test_df = (
            train_calibration_test_split(df)
        )

        # Print split sizes and date ranges.
        for name, split_df in (
            ("Train", train_df),
            ("Calibration", calibration_df),
            ("Test", test_df),
        ):
            print(
                f"{name}: {len(split_df)} rows, "
                f"{split_df['SETTLEMENTDATE'].min()} to "
                f"{split_df['SETTLEMENTDATE'].max()}"
            )

        # Save each split separately for this region.
        train_file = (
            PROCESSED_DATA_DIR
            / f"{region}_train.csv"
        )

        calibration_file = (
            PROCESSED_DATA_DIR
            / f"{region}_calibration.csv"
        )

        test_file = (
            PROCESSED_DATA_DIR
            / f"{region}_test.csv"
        )

        train_df.to_csv(
            train_file,
            index=False,
        )

        calibration_df.to_csv(
            calibration_file,
            index=False,
        )

        test_df.to_csv(
            test_file,
            index=False,
        )

        print()
        print(f"Saved train split: {train_file}")
        print(f"Saved calibration split: {calibration_file}")
        print(f"Saved test split: {test_file}")