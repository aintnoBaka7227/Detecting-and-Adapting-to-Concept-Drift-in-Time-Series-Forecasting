"""Load raw AEMO price and demand data.

The loader preserves the native AEMO sampling frequency.

2018-01 to 2021-09:
    30-minute trading data.

2021-10 to 2023-12:
    5-minute dispatch data.

No resampling, cleaning or train/test splitting is performed here.
"""

from pathlib import Path

import pandas as pd
from nemosis import dynamic_data_compiler

from drift_forecasting.config import (
    AEMO_30MIN_END,
    AEMO_30MIN_START,
    AEMO_5MIN_END,
    AEMO_5MIN_START,
    NEMOSIS_CACHE,
    RAW_DATA_DIR,
    REGIONS,
)


def load_aemo_demand(
    region: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load raw AEMO price and demand data for one region.

    Uses NEMOSIS to retrieve:

    2018-01 to 2021-09:
        TRADINGREGIONSUM -> TOTALDEMAND
        TRADINGPRICE     -> RRP

    2021-10 to 2023-12:
        DISPATCHREGIONSUM -> TOTALDEMAND
        DISPATCHPRICE     -> RRP

    Three CSV files are saved:

        <region>_201801_202109_30min.csv
        <region>_202110_202312_5min.csv
        <region>_201801_202312_native.csv

    The combined file preserves the native sampling frequencies.
    No resampling, cleaning or splitting is performed.

    Returns
    -------
    pd.DataFrame
        Combined raw data sorted ascending by SETTLEMENTDATE.
    """

    if region not in REGIONS:
        raise ValueError(
            f"Unknown region {region!r}. Expected one of {REGIONS}."
        )

    # Respect a custom raw_dir if one is supplied by tests/callers.
    if raw_dir == RAW_DATA_DIR:
        cache_dir = NEMOSIS_CACHE
    else:
        cache_dir = raw_dir / "nemosis_cache"

    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # =====================================================
    # 1. Native 30-minute data
    #    2018-01-01 -> 2021-09-30
    # =====================================================

    trading_demand = dynamic_data_compiler(
        AEMO_30MIN_START,
        AEMO_30MIN_END,
        "TRADINGREGIONSUM",
        str(cache_dir),
    )

    trading_price = dynamic_data_compiler(
        AEMO_30MIN_START,
        AEMO_30MIN_END,
        "TRADINGPRICE",
        str(cache_dir),
    )

    trading_demand = trading_demand.loc[
        trading_demand["REGIONID"] == region,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "TOTALDEMAND",
        ],
    ].copy()

    trading_price = trading_price.loc[
        trading_price["REGIONID"] == region,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "RRP",
        ],
    ].copy()

    trading_demand["SETTLEMENTDATE"] = pd.to_datetime(
        trading_demand["SETTLEMENTDATE"]
    )

    trading_price["SETTLEMENTDATE"] = pd.to_datetime(
        trading_price["SETTLEMENTDATE"]
    )

    data_30min = pd.merge(
        trading_demand,
        trading_price,
        on=["SETTLEMENTDATE", "REGIONID"],
        how="inner",
    )

    data_30min = (
        data_30min
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    # =====================================================
    # 2. Native 5-minute data
    #    2021-10-01 -> 2023-12-31
    # =====================================================

    dispatch_demand = dynamic_data_compiler(
        AEMO_5MIN_START,
        AEMO_5MIN_END,
        "DISPATCHREGIONSUM",
        str(cache_dir),
    )

    dispatch_price = dynamic_data_compiler(
        AEMO_5MIN_START,
        AEMO_5MIN_END,
        "DISPATCHPRICE",
        str(cache_dir),
    )

    dispatch_demand = dispatch_demand.loc[
        dispatch_demand["REGIONID"] == region,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "TOTALDEMAND",
        ],
    ].copy()

    dispatch_price = dispatch_price.loc[
        dispatch_price["REGIONID"] == region,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "RRP",
        ],
    ].copy()

    dispatch_demand["SETTLEMENTDATE"] = pd.to_datetime(
        dispatch_demand["SETTLEMENTDATE"]
    )

    dispatch_price["SETTLEMENTDATE"] = pd.to_datetime(
        dispatch_price["SETTLEMENTDATE"]
    )

    data_5min = pd.merge(
        dispatch_demand,
        dispatch_price,
        on=["SETTLEMENTDATE", "REGIONID"],
        how="inner",
    )

    data_5min = (
        data_5min
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    # =====================================================
    # 3. Save the two native-frequency datasets
    # =====================================================

    file_30min = (
        raw_dir
        / f"{region}_201801_202109_30min.csv"
    )

    file_5min = (
        raw_dir
        / f"{region}_202110_202312_5min.csv"
    )

    data_30min.to_csv(
        file_30min,
        index=False,
    )

    data_5min.to_csv(
        file_5min,
        index=False,
    )

    print(f"Saved 30-minute dataset: {file_30min}")
    print(f"Saved 5-minute dataset: {file_5min}")

    # =====================================================
    # 4. Concatenate both periods
    # =====================================================

    data = pd.concat(
        [
            data_30min,
            data_5min,
        ],
        ignore_index=True,
    )

    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    # =====================================================
    # 5. Save the combined native-frequency dataset
    # =====================================================

    combined_file = (
        raw_dir
        / f"{region}_201801_202312_native.csv"
    )

    data.to_csv(
        combined_file,
        index=False,
    )

    print(f"Saved combined dataset: {combined_file}")

    return data


if __name__ == "__main__":
    df = load_aemo_demand("NSW1")

    print()
    print("First rows:")
    print(df.head())

    print()
    print("Last rows:")
    print(df.tail())

    print()
    print("Shape:")
    print(df.shape)