"""Data stage summary: what loader.py's audit and cleaning found, per region.

Recomputes, silently, each check `loader.clean` prints (same order), then
the 30-minute standardisation and the frozen split, and saves them as two
tables for the report and the Streamlit Data tab:
    data_audit.csv    one row per audit / cleaning / standardise step,
                      with the action taken and the SA1 and NSW1 values;
    data_splits.csv   train / calibration / test dates and half-hour counts.
Cleaning never alters a demand or price value; price (RRP) is audited but
not used anywhere else in the project. No runs.csv rows (a data check, not
a reported run).

Usage:
    python -m experiments.run.detection.check_data_audit
"""

from __future__ import annotations

import contextlib
import io

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS, SPLIT
from experiments.produce.shared.table_image import save_table_image
from experiments.results_io import TABLES_DIR

DATA_DIR = TABLES_DIR / "data"
FIVE_MINUTE_START = pd.Timestamp("2021-10-01")


def unexpected_gaps(timestamps: pd.Series, expected: pd.Timedelta) -> int:
    diffs = timestamps.drop_duplicates().sort_values().diff().dropna()
    return int((diffs != expected).sum())


def audit_region(region: str) -> dict[str, str]:
    raw = loader._read_monthly(region)
    data = raw.copy()

    demand = pd.to_numeric(data["TOTALDEMAND"], errors="coerce")
    price = pd.to_numeric(data["RRP"], errors="coerce")
    coerced = int((demand.isna() & data["TOTALDEMAND"].notna()).sum()
                  + (price.isna() & data["RRP"].notna()).sum())
    data["TOTALDEMAND"], data["RRP"] = demand, price

    data["SETTLEMENTDATE"] = pd.to_datetime(data["SETTLEMENTDATE"], errors="coerce")
    invalid = int(data["SETTLEMENTDATE"].isna().sum())
    data = data.dropna(subset=["SETTLEMENTDATE"])
    out_of_order = int((data["SETTLEMENTDATE"].diff() < pd.Timedelta(0)).sum())
    data = data.sort_values("SETTLEMENTDATE")
    exact = int(data.duplicated().sum())
    data = data.drop_duplicates()
    conflicting = int(data.duplicated(subset=["SETTLEMENTDATE", "REGION"], keep=False).sum())
    data = data.drop_duplicates(subset=["SETTLEMENTDATE", "REGION"], keep="first")

    ts = data["SETTLEMENTDATE"]
    gaps_30 = unexpected_gaps(ts[ts < FIVE_MINUTE_START], pd.Timedelta(minutes=30))
    gaps_5 = unexpected_gaps(ts[ts >= FIVE_MINUTE_START], pd.Timedelta(minutes=5))
    period_types = ", ".join(
        f"{k} ({v:,})" for k, v in data["PERIODTYPE"].value_counts(dropna=False).items()
    )

    with contextlib.redirect_stdout(io.StringIO()):  # clean() prints its own audit
        cleaned = loader.clean(raw)
    processed = loader.load_processed(region)

    return {
        "Rows read": f"{len(raw):,}",
        "Required columns present": "yes",
        "Non-numeric demand / price": f"{coerced}",
        "Invalid timestamps": f"{invalid}",
        "Out-of-order timestamps": f"{out_of_order}",
        "Exact duplicate rows": f"{exact}",
        "Same timestamp, different values": f"{conflicting}",
        "Missing demand / price": (
            f"{int(data['TOTALDEMAND'].isna().sum())} / {int(data['RRP'].isna().sum())}"
        ),
        "Sampling gaps (30-min before Oct 2021 / 5-min after)": f"{gaps_30} / {gaps_5}",
        "Period type": period_types,
        "Negative demand intervals": f"{int((data['TOTALDEMAND'] < 0).sum())}",
        "Demand range (MW)": f"{data['TOTALDEMAND'].min():,.1f} to {data['TOTALDEMAND'].max():,.1f}",
        "Price range ($/MWh, not studied)": f"{data['RRP'].min():,.0f} to {data['RRP'].max():,.0f}",
        "Rows after cleaning": f"{len(cleaned):,}",
        "Half-hours on the 30-minute grid": f"{len(processed):,}",
        "Empty half-hours (NaN, not imputed)": f"{int(processed['TOTALDEMAND'].isna().sum())}",
    }


# (step, check, action taken) in loader.py order; the check names are the keys
# audit_region returns.
STEPS = [
    ("Read", "Rows read", "Monthly AEMO CSVs, joined in time order"),
    ("Clean", "Required columns present", "Stop if any is missing"),
    ("Clean", "Non-numeric demand / price", "Coerced to NaN so they show up as missing"),
    ("Clean", "Invalid timestamps", "Dropped (cannot be placed in time)"),
    ("Clean", "Out-of-order timestamps", "Counted, then the series is sorted by time"),
    ("Clean", "Exact duplicate rows", "Dropped"),
    ("Clean", "Same timestamp, different values", "First row kept"),
    ("Clean", "Missing demand / price", "Reported"),
    ("Clean", "Sampling gaps (30-min before Oct 2021 / 5-min after)", "Reported"),
    ("Clean", "Period type", "Reported"),
    ("Clean", "Negative demand intervals", "Reported, kept (values never altered)"),
    ("Clean", "Demand range (MW)", "Reported, kept"),
    ("Clean", "Price range ($/MWh, not studied)", "Reported, kept"),
    ("Clean", "Rows after cleaning", "Final sort check"),
    ("Standardise", "Half-hours on the 30-minute grid", "5-minute data averaged to 30 minutes, interval-end label"),
    ("Standardise", "Empty half-hours (NaN, not imputed)", "Gaps left as NaN"),
]


def splits_table() -> pd.DataFrame:
    rows = []
    roles = {
        "train": "Fit forecasters and the seasonal profile; tune",
        "calibration": "Select the forecaster; calibrate the conformal intervals",
        "test": "All evaluation; every Tier 1 and all but one Tier 2 event fall here",
    }
    frames = {region: loader.load(region) for region in REGIONS}
    for i, name in enumerate(("train", "calibration", "test")):
        start, end = SPLIT[name]
        row = {"split": name.capitalize(), "from": start, "to": end or "end of data (2023-12-31)"}
        for region in REGIONS:
            row[f"{region} half-hours"] = f"{len(frames[region][i]):,}"
        row["role"] = roles[name]
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    values = {region: audit_region(region) for region in REGIONS}
    audit = pd.DataFrame(
        [
            {"step": step, "check": check, "action": action}
            | {region: values[region][check] for region in REGIONS}
            for step, check, action in STEPS
        ]
    )
    splits = splits_table()

    audit_out = DATA_DIR / "data_audit.csv"
    splits_out = DATA_DIR / "data_splits.csv"
    audit.to_csv(audit_out, index=False)
    splits.to_csv(splits_out, index=False)
    print(audit.to_string(index=False), "\n")
    print(splits.to_string(index=False))
    save_table_image(
        audit,
        audit_out.with_suffix(".png"),
        title="AEMO data audit and cleaning (loader.py)",
        subtitle="Cleaning is non-destructive: only rows that cannot be placed in time are dropped; "
        "demand and price values are never altered. Price is audited but not studied.",
    )
    save_table_image(splits, splits_out.with_suffix(".png"), title="Train / Calibration / Test split")
    print(f"wrote {audit_out}\nwrote {splits_out}")


if __name__ == "__main__":
    main()
