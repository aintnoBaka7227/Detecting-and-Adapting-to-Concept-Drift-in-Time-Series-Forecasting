"""Load, clean, standardise and split one AEMO region's demand series.

`load_processed(region)` runs read -> clean -> standardise and returns the
single continuous, pre-split frame. `load(region)` splits that into
`(train, calibration, test)`. Nothing here writes a file; both are memoised
per region.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.config import RAW_DATA_DIR, REGIONS, SPLIT

_REQUIRED_COLUMNS = ("REGION", "SETTLEMENTDATE", "TOTALDEMAND", "RRP", "PERIODTYPE")
_REQUIRED_COLUMNS_SET = set(_REQUIRED_COLUMNS)
_ONE_DAY = pd.Timedelta(days=1)
# AEMO switched from 30-minute to 5-minute settlement on 1 October 2021.
_FIVE_MINUTE_START = pd.Timestamp("2021-10-01")
_Splits = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

_processed_cache: dict[str, pd.DataFrame] = {}
_split_cache: dict[str, _Splits] = {}


def load_processed(region: str) -> pd.DataFrame:
    """Cleaned, 30-minute-grid series for ``region``, before splitting.

    Chain: read monthly raw CSVs -> concat -> clean -> standardise. Same
    columns as `standardise()` returns — no `split` or other column added.
    Memoised per region; treat the frame as read-only.
    """
    if region not in REGIONS:
        raise ValueError(f"Unknown region {region!r}. Expected one of {REGIONS}.")
    if region not in _processed_cache:
        _processed_cache[region] = standardise(clean(_read_monthly(region)))
    return _processed_cache[region]


def load(region: str) -> _Splits:
    """Return ``(train, calibration, test)`` for ``region`` ("SA1" or "NSW1").

    Splits `load_processed(region)`. Memoised per region; treat the frames
    as read-only.
    """
    if region not in _split_cache:
        _split_cache[region] = split(load_processed(region))
    return _split_cache[region]


def _read_monthly(region: str) -> pd.DataFrame:
    """Concatenate the monthly AEMO CSVs for one region in chronological order."""
    region_dir = RAW_DATA_DIR / region
    if not region_dir.exists():
        region_dir = RAW_DATA_DIR / region.lower()
    if not region_dir.exists():
        raise FileNotFoundError(f"Region directory does not exist: {region_dir}")

    files = sorted(region_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {region_dir}")

    months = []
    for path in files:
        df = pd.read_csv(path)
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name} is missing columns: {missing}")
        df = df[list(_REQUIRED_COLUMNS)].copy()
        df["SETTLEMENTDATE"] = pd.to_datetime(df["SETTLEMENTDATE"])
        months.append(df)
    return pd.concat(months, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Print the data-quality audit and safely clean one region's dataset.

    Cleaning is non-destructive: invalid timestamps and exact / same-interval
    duplicates are dropped, but demand and price values are never altered or
    clipped. Op sequence unchanged from the original cleaning module; the
    audit is printed as a side effect and changes nothing.
    """
    # Stage 1 — required columns must be present, else nothing downstream is valid.
    missing = _REQUIRED_COLUMNS_SET - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = df.copy()
    print("=== AEMO Data Quality Check ===")
    print(f"Rows before cleaning: {len(data)}")

    # Stage 2 — non-numeric demand/price become NaN so they show up as missing later.
    data["TOTALDEMAND"] = pd.to_numeric(data["TOTALDEMAND"], errors="coerce")
    data["RRP"] = pd.to_numeric(data["RRP"], errors="coerce")
    print(f"Regions: {sorted(data['REGION'].dropna().unique())}\n")

    # Stage 3 — unparseable timestamps can't be placed in the series, so drop them.
    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"], errors="coerce")
    print(f"Invalid timestamps: {int(data['SETTLEMENTDATE'].isna().sum())}")
    data = data.dropna(subset=["SETTLEMENTDATE"]).reset_index(drop=True)

    # Stage 4 — count how many rows arrived out of order, then enforce chronological order.
    print(f"Out-of-order timestamps: {int((data['SETTLEMENTDATE'].diff() < pd.Timedelta(0)).sum())}")
    data = data.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    print(f"Date range: {data['SETTLEMENTDATE'].min()} to {data['SETTLEMENTDATE'].max()}\n")

    # Stage 5 — fully identical rows carry no information; drop them.
    print(f"Exact duplicate rows: {int(data.duplicated().sum())}")
    data = data.drop_duplicates().reset_index(drop=True)

    # Stage 6 — a repeated (timestamp, region) with different values is a conflict; keep the first.
    conflicting = int(data.duplicated(subset=["SETTLEMENTDATE", "REGION"], keep=False).sum())
    print(f"Conflicting duplicate timestamp rows: {conflicting}\n")
    data = data.drop_duplicates(
        subset=["SETTLEMENTDATE", "REGION"], keep="first"
    ).reset_index(drop=True)

    # Stage 7 — missing values remaining in the project's columns.
    print("Missing values:")
    print(data[list(_REQUIRED_COLUMNS)].isna().sum())

    # Stage 8 — gaps that break the expected 30-min (pre-Oct-2021) / 5-min sampling cadence.
    print("\n30-minute period:")
    _check_frequency(data[data["SETTLEMENTDATE"] < _FIVE_MINUTE_START], pd.Timedelta(minutes=30))
    print("\n5-minute period:")
    _check_frequency(data[data["SETTLEMENTDATE"] >= _FIVE_MINUTE_START], pd.Timedelta(minutes=5))

    # Stage 9 — which settlement period types are present (TRADE vs DISPATCH etc.).
    print("\nPERIODTYPE values:")
    print(data["PERIODTYPE"].value_counts(dropna=False))

    # Stage 10 — implausible values: negative demand is suspicious, extreme RRP can be genuine.
    print(f"\nNegative TOTALDEMAND values: {int((data['TOTALDEMAND'] < 0).sum())}")
    print(f"TOTALDEMAND range: {data['TOTALDEMAND'].min()} to {data['TOTALDEMAND'].max()}")
    print(f"RRP range: {data['RRP'].min()} to {data['RRP'].max()}\n")

    # Stage 11 — final guarantee that the returned frame is chronologically ordered.
    data = data.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    print("=== Final Quality Check ===")
    print(f"Rows after cleaning: {len(data)}")
    print(f"Sorted by time: {data['SETTLEMENTDATE'].is_monotonic_increasing}")
    return data


def _check_frequency(df: pd.DataFrame, expected_interval: pd.Timedelta) -> None:
    """Print how many consecutive-timestamp gaps differ from the expected interval."""
    diffs = df["SETTLEMENTDATE"].drop_duplicates().sort_values().diff().dropna()
    unexpected = diffs[diffs != expected_interval]
    print(f"Expected interval: {expected_interval}")
    print(f"Unexpected intervals: {len(unexpected)}")
    if not unexpected.empty:
        print("Examples:")
        print(unexpected.head(10))


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Resample onto a regular 30-minute grid (interval-end labelled).

    Gaps become NaN rows; nothing is imputed. Values stay raw MW / $. Drops
    PERIODTYPE.
    """
    data = df.copy()
    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"])
    return (
        data.set_index("SETTLEMENTDATE")
        .groupby("REGION")
        .resample("30min", label="right", closed="right")
        .agg({"TOTALDEMAND": "mean", "RRP": "mean"})
        .reset_index()
    )


def split(df: pd.DataFrame) -> _Splits:
    """Frozen train / calibration / test split from config.SPLIT.

    End dates are inclusive of the whole day. Rows outside every window are
    dropped.
    """
    data = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    ts = data["SETTLEMENTDATE"]

    train_start, train_end = (pd.Timestamp(x) for x in SPLIT["train"])
    cal_start, cal_end = (pd.Timestamp(x) for x in SPLIT["calibration"])
    test_start, test_end = SPLIT["test"]
    test_start = pd.Timestamp(test_start)

    in_test = ts >= test_start
    if test_end is not None:
        in_test &= ts < pd.Timestamp(test_end) + _ONE_DAY

    return (
        data[(ts >= train_start) & (ts < train_end + _ONE_DAY)].reset_index(drop=True),
        data[(ts >= cal_start) & (ts < cal_end + _ONE_DAY)].reset_index(drop=True),
        data[in_test].reset_index(drop=True),
    )
