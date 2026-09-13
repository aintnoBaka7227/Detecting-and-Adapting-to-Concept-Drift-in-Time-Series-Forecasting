"""Daily aggregation of a half-hourly AEMO split into a coarser stream.

Raw half-hourly demand makes a detector fire mostly on ordinary
daily/weekly seasonality, not genuine drift. Collapsing to one point per
calendar day removes that before a detector ever sees the stream, without
touching the half-hourly data used elsewhere (forecasting, retraining).
"""

from __future__ import annotations

import pandas as pd


def daily_aggregate(
    split: pd.DataFrame,
    column: str = "TOTALDEMAND",
    agg: str = "mean",
) -> pd.Series:
    """Collapse one AEMO split (any of `aemo.loader`'s outputs) to one row
    per calendar day. Returns a float Series named `column`, indexed by
    day; all-NaN/empty days are dropped."""
    if "SETTLEMENTDATE" not in split.columns:
        raise ValueError("split must have a SETTLEMENTDATE column")
    if column not in split.columns:
        raise ValueError(f"split has no column {column!r}")

    series = pd.Series(
        split[column].to_numpy(dtype=float),
        index=pd.DatetimeIndex(split["SETTLEMENTDATE"]),
        name=column,
    ).sort_index()

    return series.resample("D").agg(agg).dropna()
