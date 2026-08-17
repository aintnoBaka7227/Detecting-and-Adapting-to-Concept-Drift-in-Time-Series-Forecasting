"""Data-quality checks and fixes: DST gaps, duplicate intervals, missing intervals.

Every fix applied here must be recorded in the data-quality note that is
the Step 1 deliverable — this is where those fixes actually happen, so
keep the note in sync with what this function does.
"""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "SETTLEMENTDATE",
    "REGIONID",
    "TOTALDEMAND",
    "RRP",
}

FIVE_MINUTE_START = pd.Timestamp("2021-10-01")


def clean_demand_series(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of `df` with data-quality issues checked.

    Checks:
    - required columns and data types
    - timestamp parsing and chronological order
    - duplicate observations
    - missing values
    - missing/unexpected intervals
    - 30-minute / 5-minute frequency
    - basic demand and price outliers

    Only safe fixes are applied automatically:
    - invalid timestamps are removed
    - exact duplicate rows are removed
    - conflicting duplicate timestamps (same SETTLEMENTDATE + REGIONID,
      differing RRP and/or TOTALDEMAND) are resolved by keeping the
      first-appearing row and dropping the rest
    - rows are sorted chronologically

    Missing values, unusual gaps and outliers are reported rather than
    automatically modified.

    Must only use information available at or before each row's own
    timestamp — no full-series statistics — or the Step 7 leakage audit
    will fail on this function.
    """

    data = df.copy()

    print("=== AEMO Data Quality Check ===")
    print(f"Rows before cleaning: {len(data)}")

    # -----------------------------------------------------
    # 1. Validate structure
    # -----------------------------------------------------

    missing_columns = REQUIRED_COLUMNS - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data["TOTALDEMAND"] = pd.to_numeric(
        data["TOTALDEMAND"],
        errors="coerce",
    )

    data["RRP"] = pd.to_numeric(
        data["RRP"],
        errors="coerce",
    )

    print(f"Regions: {sorted(data['REGIONID'].dropna().unique())}")

    # -----------------------------------------------------
    # 2. Validate timestamps
    # -----------------------------------------------------

    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"],
        errors="coerce",
    )

    invalid_timestamps = data["SETTLEMENTDATE"].isna().sum()

    print(f"Invalid timestamps: {invalid_timestamps}")

    if invalid_timestamps:
        data = data.dropna(
            subset=["SETTLEMENTDATE"]
        )

    data = data.reset_index(drop=True)

    # Report any place where the raw arrival order goes backwards in time,
    # before the unconditional sort below silently fixes it.
    out_of_order = data["SETTLEMENTDATE"].diff() < pd.Timedelta(0)
    n_out_of_order = out_of_order.sum()

    print(f"Out-of-order timestamps: {n_out_of_order}")

    if n_out_of_order:
        print("Out-of-order sections (row index: previous -> current):")

        for idx in data.index[out_of_order][:10]:
            print(
                f"  row {idx}: "
                f"{data.loc[idx - 1, 'SETTLEMENTDATE']} -> "
                f"{data.loc[idx, 'SETTLEMENTDATE']}"
            )

    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    if not data.empty:
        print(
            "Date range:",
            data["SETTLEMENTDATE"].min(),
            "to",
            data["SETTLEMENTDATE"].max(),
        )

    # -----------------------------------------------------
    # 3. Check duplicates
    # -----------------------------------------------------

    exact_duplicates = data.duplicated(
        keep="first"
    )

    print(
        f"Exact duplicate rows: {exact_duplicates.sum()}"
    )

    # Exact duplicate rows contain no additional information,
    # so keeping the first occurrence is safe.
    data = (
        data
        .loc[~exact_duplicates]
        .reset_index(drop=True)
    )

    # After exact duplicates are removed, any remaining rows
    # with the same timestamp + region must contain conflicting
    # demand and/or price values.
    duplicate_intervals = data.duplicated(
        subset=["SETTLEMENTDATE", "REGIONID"],
        keep=False,
    )

    conflicting_duplicates = data.loc[
        duplicate_intervals,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "TOTALDEMAND",
            "RRP",
        ],
    ].copy()

    print(
        "Conflicting duplicate timestamp rows:",
        len(conflicting_duplicates),
    )

    if not conflicting_duplicates.empty:
        issues_dir = Path("data/processed/quality_issues")

        issues_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Tag the file with the region(s) present so cleaning a second
        # region does not overwrite the first region's report.
        region_tag = "_".join(
            sorted(data["REGIONID"].dropna().unique())
        ) or "unknown"

        output_file = (
            issues_dir
            / f"conflicting_duplicate_timestamps_{region_tag}.csv"
        )

        conflicting_duplicates.to_csv(
            output_file,
            index=False,
        )

        print(
            f"Conflicting duplicates saved to: {output_file}"
        )

        # The conflicting rows are kept in their original (chronological)
        # order, so the first occurrence per timestamp+region is the
        # earliest-published value. Later republications of the same
        # interval (e.g. a revised RRP) are dropped rather than averaged
        # or chosen arbitrarily.
        rows_before = len(data)

        data = (
            data
            .drop_duplicates(
                subset=["SETTLEMENTDATE", "REGIONID"],
                keep="first",
            )
            .reset_index(drop=True)
        )

        print(
            "Rows dropped keeping first-appearing value: "
            f"{rows_before - len(data)}"
        )

    # -----------------------------------------------------
    # 4. Check missing values
    # -----------------------------------------------------

    missing_values = data[
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "TOTALDEMAND",
            "RRP",
        ]
    ].isna().sum()

    print("\nMissing values:")
    print(missing_values)

    # Do not interpolate/backfill missing observations here.
    # They are reported so the treatment can be documented.

    # -----------------------------------------------------
    # 5. Validate frequency / missing intervals
    # -----------------------------------------------------

    before_5ms = data[
        data["SETTLEMENTDATE"] < FIVE_MINUTE_START
    ]

    after_5ms = data[
        data["SETTLEMENTDATE"] >= FIVE_MINUTE_START
    ]

    print("\n30-minute period:")
    _check_frequency(
        before_5ms,
        expected_interval=pd.Timedelta(minutes=30),
    )

    print("\n5-minute period:")
    _check_frequency(
        after_5ms,
        expected_interval=pd.Timedelta(minutes=5),
    )

    # before_5ms/after_5ms are checked independently above, so a gap
    # straddling the cutover itself would otherwise go unreported.
    if not before_5ms.empty and not after_5ms.empty:
        boundary_gap = (
            after_5ms["SETTLEMENTDATE"].min()
            - before_5ms["SETTLEMENTDATE"].max()
        )
        print(f"\nBoundary gap (30-min -> 5-min): {boundary_gap}")

    # -----------------------------------------------------
    # 6. Check basic outliers / logical validity
    # -----------------------------------------------------

    negative_demand = (
        data["TOTALDEMAND"] < 0
    ).sum()

    print(
        f"\nNegative TOTALDEMAND values: {negative_demand}"
    )

    if not data.empty:
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

    # Do not remove extreme RRP values automatically.
    # Electricity-market price spikes may be genuine observations.

    # -----------------------------------------------------
    # 7. Final validation
    # -----------------------------------------------------

    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    print("\n=== Final Quality Check ===")
    print(f"Rows after cleaning: {len(data)}")
    print(
        "Sorted by time:",
        data["SETTLEMENTDATE"].is_monotonic_increasing,
    )

    return data


def resample_to_30min(df: pd.DataFrame) -> pd.DataFrame:
    """Convert native AEMO data to 30-minute intervals.

    Expects `df` to already be cleaned (e.g. via `clean_demand_series`):
    conflicting/duplicate timestamps are not resolved here, they are
    averaged in along with everything else in the bin.

    Aggregates with a trailing mean per 30-minute window (`label="right",
    closed="right"`), matching AEMO's own interval-ending SETTLEMENTDATE
    convention, so each bin only ever averages observations at or before
    its own label - no future information crosses the boundary.

    Grouped by REGIONID first: resampling the whole frame at once would
    blend different regions' demand/price together into a single value
    per timestamp.

    A 30-minute bin with no native observations becomes a row of NaN
    rather than being silently skipped - missing intervals stay visible
    downstream instead of disappearing.
    """

    data = df.copy()

    data["SETTLEMENTDATE"] = pd.to_datetime(
        data["SETTLEMENTDATE"]
    )

    data = (
        data
        .set_index("SETTLEMENTDATE")
        .groupby("REGIONID")
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


def _check_frequency(
    df: pd.DataFrame,
    expected_interval: pd.Timedelta,
) -> None:
    """Report missing or unexpected time intervals."""

    if df.empty:
        print("No observations.")
        return

    timestamps = (
        df["SETTLEMENTDATE"]
        .drop_duplicates()
        .sort_values()
    )

    differences = timestamps.diff().dropna()

    unexpected = differences[
        differences != expected_interval
    ]

    print(f"Expected interval: {expected_interval}")
    print(f"Unexpected intervals: {len(unexpected)}")

    if len(unexpected) > 0:
        print("Examples:")
        print(unexpected.head(10))

if __name__ == "__main__":
    from drift_forecasting.config import PROCESSED_DATA_DIR, RAW_DATA_DIR

    raw_file = RAW_DATA_DIR / "NSW1_201801_202312_native.csv"

    df = pd.read_csv(raw_file)

    cleaned_df = clean_demand_series(df)

    print("\nCleaning finished.")
    print(cleaned_df.head())
    print(cleaned_df.shape)

    resampled_df = resample_to_30min(cleaned_df)

    print("\nResampling finished.")
    print(resampled_df.head())
    print(resampled_df.shape)

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_file = (
        PROCESSED_DATA_DIR / "NSW1_201801_202312_cleaned_native.csv"
    )
    resampled_file = (
        PROCESSED_DATA_DIR / "NSW1_201801_202312_cleaned_30min.csv"
    )

    cleaned_df.to_csv(cleaned_file, index=False)
    resampled_df.to_csv(resampled_file, index=False)

    print(f"\nSaved cleaned native-frequency dataset: {cleaned_file}")
    print(f"Saved cleaned 30-minute dataset: {resampled_file}")