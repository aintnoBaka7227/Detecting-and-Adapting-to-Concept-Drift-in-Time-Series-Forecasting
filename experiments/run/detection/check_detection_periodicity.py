"""Validation: do the AEMO detectors fire periodically?

A detector whose alarms follow the clock or calendar (every ~2 weeks, every
Monday, every summer) is reacting to timing, not to change. For every
detector x region of the standard-half-hourly post-tuning runs, and for both
alarm sets -- raw (every TEST alarm) and accepted (after the 14-day
REFRACTORY_PERIOD debounce, the ones that trigger retraining) -- this reports:

    gap CV        sd / mean of the days between alarms (~1 random, << 1
                  evenly spaced, > 1 clustered), as a percentile of N_SIM
                  random sequences with the same alarm count over the TEST
                  period (and, for accepted alarms, the same 14-day minimum
                  gap, since the debounce alone makes spacing more regular);
    edge share    accepted only: share of gaps in [14, 16) days, i.e. firing
                  as soon as the debounce allows, with its random percentile;
    phase R (p)   Rayleigh test of alarm phase in four cycles: day of week,
                  day of year, a fixed 14-day and a fixed 30-day period.

The table lists these metrics only. With 15-50 alarms per detector the tests
have limited power.

Reads the saved raw_ / accepted_detections files; writes
results/tables/detection_periodicity/detection_periodicity.{csv,png}.
Nothing is logged to runs.csv (a validation check, not a reported run).

Usage:
    python -m experiments.run.detection.check_detection_periodicity
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from drift_lab import evaluation
from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.evaluation.evaluation import REFRACTORY_PERIOD
from experiments.produce.shared.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLES_DIR, output_path
from experiments.run.detection.detection_artifacts import (
    accepted_detections_path,
    raw_detections_path,
)
from experiments.run.detection.standard_stream_common import TEST_START

SPLIT_ID = "aemo_detect_standard_half_hourly_post_tune_v1"
N_SIM = 5000
EDGE_DAYS = (14, 16)  # [refractory, refractory + 2 days)
# (label, period in days, phase origin)
CYCLES = (
    ("weekday", 7, pd.Timestamp("2020-03-02")),  # a Monday
    ("day of year", 365.2425, pd.Timestamp("2020-01-01")),
    ("14-day", 14, TEST_START),
    ("30-day", 30, TEST_START),
)


def latest_detection_runs() -> pd.DataFrame:
    runs = pd.read_csv(RUNS_CSV)
    rows = runs[runs["split_id"] == SPLIT_ID]
    if rows.empty:
        raise SystemExit(f"no rows for split_id={SPLIT_ID!r} -- run the post-tune detectors first")
    latest = rows[rows["timestamp"] == rows.groupby(["method", "region"])["timestamp"].transform("max")]
    return latest.drop_duplicates(["method", "region"])[["method", "region", "config_hash"]]


def percentile(value: float, null: np.ndarray) -> float:
    """Share of random sequences at or below `value` (NaN-safe)."""
    null = null[~np.isnan(null)]
    return float("nan") if np.isnan(value) or len(null) == 0 else float((null <= value).mean())


def assess(timestamps: pd.DatetimeIndex, span_days: float, accepted: bool) -> dict:
    days = np.asarray((timestamps - TEST_START) / pd.Timedelta(days=1), dtype=float)
    min_gap = REFRACTORY_PERIOD / pd.Timedelta(days=1) if accepted else 0.0
    row = {"alarms": len(days)}
    if len(days) < 3:
        return row

    sims = evaluation.simulate_random_alarm_days(len(days), span_days, min_gap, N_SIM, seed=0)
    cv = evaluation.calculate_gap_cv(days)
    cv_null = np.array([evaluation.calculate_gap_cv(s) for s in sims])
    gaps = evaluation.calculate_alarm_gaps(days)
    row |= {
        "mean gap (days)": round(float(gaps.mean()), 1),
        "gap CV": round(cv, 2),
        "CV pct vs random": round(percentile(cv, cv_null), 3),
    }

    if accepted:
        edge = evaluation.calculate_gap_share(days, *EDGE_DAYS)
        edge_null = np.array([evaluation.calculate_gap_share(s, *EDGE_DAYS) for s in sims])
        edge_pct = percentile(edge, edge_null)
        row |= {"gaps 14-16 d": round(edge, 2), "edge pct vs random": round(edge_pct, 3)}

    for label, period, origin in CYCLES:
        r, p = evaluation.calculate_phase_concentration(timestamps, period, origin)
        row[f"{label} R (p)"] = f"{r:.2f} ({p:.3f})"
    return row


def main() -> None:
    spans = {}
    for region in REGIONS:
        _, _, test = loader.load(region)
        end = pd.Timestamp(test["SETTLEMENTDATE"].max())
        spans[region] = (end - TEST_START) / pd.Timedelta(days=1)

    rows = []
    for run in latest_detection_runs().sort_values(["method", "region"]).itertuples():
        for label, path, accepted in (
            ("raw", raw_detections_path(run.config_hash, run.region), False),
            ("accepted", accepted_detections_path(run.config_hash, run.region), True),
        ):
            timestamps = pd.DatetimeIndex(pd.read_csv(path, parse_dates=["timestamp"])["timestamp"]).sort_values()
            rows.append(
                {"detector": run.method, "region": run.region, "alarm set": label}
                | assess(timestamps, spans[run.region], accepted)
            )

    table = pd.DataFrame(rows)
    out = output_path(TABLES_DIR, "detection_periodicity", name="detection_periodicity.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")
    image = save_table_image(
        table.fillna("—"),
        out.with_suffix(".png"),
        title="Detection periodicity: are alarms tied to the clock or calendar?",
        subtitle=(
            "Standard-half-hourly post-tuning detectors, TEST period. Gap CV (sd / mean of days between "
            f"alarms) as a percentile of {N_SIM} random sequences with the same alarm count (accepted: the "
            "same 14-day minimum gap); edge share = gaps in [14, 16) days; R (p) = Rayleigh phase "
            "concentration. Limited power with few alarms."
        ),
    )
    print(f"wrote {image}")


if __name__ == "__main__":
    main()
