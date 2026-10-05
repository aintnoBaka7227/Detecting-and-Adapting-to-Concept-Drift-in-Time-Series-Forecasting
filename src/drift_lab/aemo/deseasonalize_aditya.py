"""Convenience wrapper for deseasonalising raw AEMO demand streams.

This module makes the project's seasonal-adjustment logic easier to use on
plain raw demand series. It accepts either:

- a daily demand stream, or
- a half-hourly demand stream

and removes the usual yearly/weekly pattern using a training/reference period.

The implementation reuses the stronger seasonal logic already in
``drift_lab.aemo.deseasonalise`` rather than re-implementing the model.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

from drift_lab.aemo.deseasonalise import (
    aggregate_daily_demand,
    remove_daily_seasonal_profile,
    remove_daily_weekly_profile,
)

Frequency = Literal["daily", "half_hourly"]


def _coerce_to_series(
    data: pd.Series | pd.DataFrame,
    *,
    value_column: str = "TOTALDEMAND",
    timestamp_column: str = "SETTLEMENTDATE",
) -> pd.Series:
    """Convert a raw stream or frame into a clean datetime-indexed Series."""
    if isinstance(data, pd.Series):
        series = data.copy()
    elif isinstance(data, pd.DataFrame):
        if value_column not in data.columns:
            raise KeyError(
                f"Column {value_column!r} not found in the input data."
            )
        if timestamp_column in data.columns:
            series = data.set_index(pd.to_datetime(data[timestamp_column]))[
                value_column
            ].copy()
        else:
            series = data[value_column].copy()
    else:
        raise TypeError(
            "Input must be a pandas Series or a DataFrame with a demand column."
        )

    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Demand stream index must be datetime-like."
            ) from exc

    if series.empty:
        raise ValueError("Demand stream is empty.")

    series = series.sort_index().astype(float)
    if series.isna().any():
        series = series.dropna()

    if series.empty:
        raise ValueError("Demand stream has no valid numeric values after dropping NaNs.")

    return series


def infer_frequency(series: pd.Series) -> Frequency:
    """Infer whether a demand stream is daily or half-hourly."""
    if len(series) < 2:
        raise ValueError("At least two timestamps are required to infer frequency.")

    diffs = series.index.to_series().diff().dropna()
    if diffs.empty:
        raise ValueError("Cannot infer the sampling frequency from an empty index.")

    median_delta = diffs.median()
    if pd.Timedelta(hours=23) <= median_delta <= pd.Timedelta(hours=25):
        return "daily"
    if pd.Timedelta(minutes=29) <= median_delta <= pd.Timedelta(minutes=31):
        return "half_hourly"

    # Helpful fallback: 30-minute half-hour data is the common AEMO cadence.
    if median_delta < pd.Timedelta(hours=12):
        return "half_hourly"

    return "daily"


def deseasonalize_raw_stream(
    stream: pd.Series | pd.DataFrame,
    *,
    reference: pd.Series | pd.DataFrame | None = None,
    frequency: Frequency | str | None = None,
    annual_smoothing_days: int = 31,
    value_column: str = "TOTALDEMAND",
    timestamp_column: str = "SETTLEMENTDATE",
) -> pd.Series:
    """Remove the usual seasonal pattern from a raw demand stream.

    Parameters
    ----------
    stream:
        A daily or half-hourly demand stream. It may be a Series or a DataFrame.
        If it is a DataFrame, it must contain a demand column such as
        ``TOTALDEMAND`` and optionally a datetime column such as
        ``SETTLEMENTDATE``.

    reference:
        Training/reference period used to estimate the seasonal pattern. If not
        provided, the same stream is used as both the target and reference.

    frequency:
        ``"daily"`` or ``"half_hourly"``. If omitted, the function infers it
        from the index spacing.

    annual_smoothing_days:
        Number of days used to smooth the annual seasonal profile.

    Returns
    -------
    pandas.Series
        A seasonally adjusted residual series.
    """
    target = _coerce_to_series(
        stream,
        value_column=value_column,
        timestamp_column=timestamp_column,
    )

    if reference is None:
        reference = target
    else:
        reference = _coerce_to_series(
            reference,
            value_column=value_column,
            timestamp_column=timestamp_column,
        )

    if frequency is None:
        frequency = infer_frequency(target)
    else:
        frequency = str(frequency).lower()
        if frequency not in {"daily", "half_hourly"}:
            raise ValueError(
                "frequency must be 'daily' or 'half_hourly', "
                f"got {frequency!r}."
            )

    if frequency == "daily":
        target_daily = aggregate_daily_demand(target)
        reference_daily = aggregate_daily_demand(reference)
        adjusted = remove_daily_seasonal_profile(
            target_daily,
            reference_daily,
            annual_smoothing_days=annual_smoothing_days,
        )
        return adjusted.rename(f"{target.name or value_column}_seasonally_adjusted")

    # Half-hourly path: use the existing weekday/half-hour seasonal profile.
    adjusted = remove_daily_weekly_profile(
        target,
        reference,
        annual_smoothing_days=annual_smoothing_days,
    )
    return adjusted.rename(f"{target.name or value_column}_seasonally_adjusted")


def deseasonalize_daily_stream(
    stream: pd.Series | pd.DataFrame,
    *,
    reference: pd.Series | pd.DataFrame | None = None,
    annual_smoothing_days: int = 31,
    value_column: str = "TOTALDEMAND",
    timestamp_column: str = "SETTLEMENTDATE",
) -> pd.Series:
    """Convenience wrapper for a daily demand stream."""
    return deseasonalize_raw_stream(
        stream,
        reference=reference,
        frequency="daily",
        annual_smoothing_days=annual_smoothing_days,
        value_column=value_column,
        timestamp_column=timestamp_column,
    )


def deseasonalize_half_hourly_stream(
    stream: pd.Series | pd.DataFrame,
    *,
    reference: pd.Series | pd.DataFrame | None = None,
    annual_smoothing_days: int = 31,
    value_column: str = "TOTALDEMAND",
    timestamp_column: str = "SETTLEMENTDATE",
) -> pd.Series:
    """Convenience wrapper for a half-hourly demand stream."""
    return deseasonalize_raw_stream(
        stream,
        reference=reference,
        frequency="half_hourly",
        annual_smoothing_days=annual_smoothing_days,
        value_column=value_column,
        timestamp_column=timestamp_column,
    )


def _example_usage() -> None:
    """Small demo for interactive use in a notebook or shell."""
    # Example:
    # daily = pd.Series([...], index=pd.date_range(..., freq='D'), name='TOTALDEMAND')
    # residual = deseasonalize_daily_stream(daily)
    # print(residual.head())
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deseasonalise a raw daily or half-hourly demand stream."
    )
    parser.add_argument("csv_path", help="CSV file containing a demand stream")
    parser.add_argument(
        "--value-column",
        default="TOTALDEMAND",
        help="Demand column name to use (default: TOTALDEMAND)",
    )
    parser.add_argument(
        "--timestamp-column",
        default="SETTLEMENTDATE",
        help="Datetime column name to use (default: SETTLEMENTDATE)",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="Optional CSV file used to fit the seasonal profile",
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "half_hourly"],
        default=None,
        help="Manual frequency override",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write the deseasonalised series",
    )

    args = parser.parse_args()

    target = pd.read_csv(args.csv_path)
    reference = (
        pd.read_csv(args.reference_csv)
        if args.reference_csv is not None
        else None
    )

    adjusted = deseasonalize_raw_stream(
        target,
        reference=reference,
        frequency=args.frequency,
        value_column=args.value_column,
        timestamp_column=args.timestamp_column,
    )

    output = adjusted.reset_index().rename(columns={"index": args.timestamp_column})
    if args.output_csv is not None:
        output.to_csv(args.output_csv, index=False)
    else:
        print(output.head())
