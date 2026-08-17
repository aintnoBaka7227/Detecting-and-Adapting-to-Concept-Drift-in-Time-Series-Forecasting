"""Enforce the frozen train / calibration / test split from `config.SPLIT`.

No other module should slice by date directly — always go through here, so
the split boundaries can only ever be changed in one place (and, per the
project brief, never after Checkpoint 1).
"""

import pandas as pd

from drift_forecasting.config import SPLIT


def train_calibration_test_split(
    df: pd.DataFrame,
    date_col: str = "SETTLEMENTDATE",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split `df` into (train, calibration, test) using `config.SPLIT`.

    The returned `test` frame must still be consumed in time order only by
    callers — this function guarantees the boundary, not the consumption
    order.
    """

    if date_col not in df.columns:
        raise ValueError(
            f"Date column {date_col!r} not found in DataFrame."
        )

    data = df.copy()

    # Parse timestamps
    data[date_col] = pd.to_datetime(
        data[date_col],
        errors="raise",
    )

    # Keep data in chronological order
    data = (
        data
        .sort_values(date_col)
        .reset_index(drop=True)
    )

    train_start, train_end = SPLIT["train"]
    calibration_start, calibration_end = SPLIT["calibration"]
    test_start, test_end = SPLIT["test"]

    train_start = pd.Timestamp(train_start)
    train_end = pd.Timestamp(train_end)

    calibration_start = pd.Timestamp(calibration_start)
    calibration_end = pd.Timestamp(calibration_end)

    test_start = pd.Timestamp(test_start)

    # -----------------------------------------------------
    # Train
    # 2018-01-01 through 2019-12-31 inclusive
    # -----------------------------------------------------

    train = data[
        (data[date_col] >= train_start)
        & (
            data[date_col]
            < train_end + pd.Timedelta(days=1)
        )
    ].copy()

    # -----------------------------------------------------
    # Calibration
    # 2020-01-01 through 2020-02-29 inclusive
    # -----------------------------------------------------

    calibration = data[
        (data[date_col] >= calibration_start)
        & (
            data[date_col]
            < calibration_end + pd.Timedelta(days=1)
        )
    ].copy()

    # -----------------------------------------------------
    # Test
    # 2020-03-01 onward
    # -----------------------------------------------------

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
    from drift_forecasting.config import PROCESSED_DATA_DIR

    input_file = PROCESSED_DATA_DIR / "NSW1_201801_202312_cleaned_30min.csv"

    df = pd.read_csv(input_file)

    train_df, calibration_df, test_df = train_calibration_test_split(df)

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

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_file = PROCESSED_DATA_DIR / "NSW1_train.csv"
    calibration_file = PROCESSED_DATA_DIR / "NSW1_calibration.csv"
    test_file = PROCESSED_DATA_DIR / "NSW1_test.csv"

    train_df.to_csv(train_file, index=False)
    calibration_df.to_csv(calibration_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"\nSaved train split: {train_file}")
    print(f"Saved calibration split: {calibration_file}")
    print(f"Saved test split: {test_file}")