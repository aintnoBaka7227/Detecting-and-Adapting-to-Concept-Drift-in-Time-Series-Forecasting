"""Load raw AEMO price & demand data into a single, time-sorted DataFrame.

Signature is fixed so every other module can call it the same way
regardless of who wrote the body.
"""

from pathlib import Path

import pandas as pd
from nemosis import dynamic_data_compiler

from drift_forecasting.config import (
    AEMO_END,
    AEMO_START,
    NEMOSIS_CACHE,
    RAW_DATA_DIR,
    REGIONS,
)


def load_aemo_demand(
    region: str,
    raw_dir: Path = RAW_DATA_DIR,
) -> pd.DataFrame:
    """Load half-hourly AEMO price and demand data for one region.

    Uses NEMOSIS to retrieve the AEMO 30-minute trading tables:

    - TRADINGREGIONSUM -> TOTALDEMAND
    - TRADINGPRICE -> RRP

    Returns a DataFrame sorted ascending by SETTLEMENTDATE and saves
    the combined raw dataset as a CSV in data/raw/.
    """

    if region not in REGIONS:
        raise ValueError(
            f"Unknown region {region!r}. Expected one of {REGIONS}."
        )

    cache_dir = (
        NEMOSIS_CACHE
        if raw_dir == RAW_DATA_DIR
        else raw_dir / "nemosis_cache"
    )

    cache_dir.mkdir(parents=True, exist_ok=True)

    # Download/load half-hourly demand
    demand = dynamic_data_compiler(
        AEMO_START,
        AEMO_END,
        "TRADINGREGIONSUM",
        str(cache_dir),
    )

    # Download/load half-hourly price
    price = dynamic_data_compiler(
        AEMO_START,
        AEMO_END,
        "TRADINGPRICE",
        str(cache_dir),
    )

    # Keep only the requested region and required columns
    demand = demand.loc[
        demand["REGIONID"] == region,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "TOTALDEMAND",
        ],
    ].copy()

    price = price.loc[
        price["REGIONID"] == region,
        [
            "SETTLEMENTDATE",
            "REGIONID",
            "RRP",
        ],
    ].copy()

    # Convert timestamps
    demand["SETTLEMENTDATE"] = pd.to_datetime(
        demand["SETTLEMENTDATE"]
    )

    price["SETTLEMENTDATE"] = pd.to_datetime(
        price["SETTLEMENTDATE"]
    )

    # Combine demand and price
    data = pd.merge(
        demand,
        price,
        on=["SETTLEMENTDATE", "REGIONID"],
        how="inner",
    )

    # Sort chronologically
    data = (
        data
        .sort_values("SETTLEMENTDATE")
        .reset_index(drop=True)
    )

    # Save the combined raw dataset
    output_file = raw_dir / f"{region}_2018_2021.csv"
    data.to_csv(output_file, index=False)

    print(f"Saved combined dataset to: {output_file}")

    return data


if __name__ == "__main__":
    df = load_aemo_demand("NSW1")

    print(df.head())
    print(df.tail())
    print(df.shape)