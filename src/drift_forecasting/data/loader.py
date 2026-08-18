"""Load manually downloaded raw AEMO price and demand data.

Expected folder structure:

data/raw/
├── sa1/
│   ├── monthly CSV files...
│   └── ...
└── nsw1/
    ├── monthly CSV files...
    └── ...

Each CSV is expected to contain:

    REGION
    SETTLEMENTDATE
    TOTALDEMAND
    RRP
    PERIODTYPE

The loader concatenates the monthly files for the requested region and
sorts them by SETTLEMENTDATE.

No downloading, merging, resampling, cleaning, or train/test splitting
is performed here.
"""

from pathlib import Path

import pandas as pd

from drift_forecasting.config import RAW_DATA_DIR, REGIONS


REQUIRED_COLUMNS = [
    "REGION",
    "SETTLEMENTDATE",
    "TOTALDEMAND",
    "RRP",
    "PERIODTYPE",
]


def load_aemo_demand(
    region: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load and concatenate monthly AEMO CSV files for one region.

    Parameters
    ----------
    region : str
        AEMO region to load, for example "SA1" or "NSW1".

    raw_dir : Path
        Root directory containing the region folders.

    Returns
    -------
    pd.DataFrame
        Concatenated raw data with columns:

        REGION
        SETTLEMENTDATE
        TOTALDEMAND
        RRP
        PERIODTYPE

        Rows are sorted ascending by SETTLEMENTDATE.
    """

    if region not in REGIONS:
        raise ValueError(
            f"Unknown region {region!r}. Expected one of {REGIONS}."
        )

    # Example:
    # NSW1 -> data/raw/nsw1/
    # SA1  -> data/raw/sa1/
    region_dir = raw_dir / region.lower()

    if not region_dir.exists():
        raise FileNotFoundError(
            f"Region directory does not exist: {region_dir}"
        )

    csv_files = sorted(region_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {region_dir}"
        )

    monthly_data = []

    for file in csv_files:
        print(f"Loading {file}")

        df = pd.read_csv(file)

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{file.name} is missing columns: "
                f"{missing_columns}"
            )

        # Keep the dataset in the agreed interface format.
        df = df[REQUIRED_COLUMNS].copy()

        df["SETTLEMENTDATE"] = pd.to_datetime(
            df["SETTLEMENTDATE"]
        )

        monthly_data.append(df)

    # Concatenate all monthly files.
    data = pd.concat(
        monthly_data,
        ignore_index=True,
    )

    # Keep chronological order.
    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    return data


if __name__ == "__main__":
    # Load and concatenate all monthly files for each region.
    nsw1 = load_aemo_demand("NSW1")
    sa1 = load_aemo_demand("SA1")

    # Save the concatenated raw datasets.
    nsw1_file = RAW_DATA_DIR / "NSW1_201801_202312_native.csv"
    sa1_file = RAW_DATA_DIR / "SA1_201801_202312_native.csv"

    nsw1.to_csv(nsw1_file, index=False)
    sa1.to_csv(sa1_file, index=False)

    print()
    print(f"Saved NSW1 dataset: {nsw1_file}")
    print(f"Shape: {nsw1.shape}")

    print()
    print(f"Saved SA1 dataset: {sa1_file}")
    print(f"Shape: {sa1.shape}")