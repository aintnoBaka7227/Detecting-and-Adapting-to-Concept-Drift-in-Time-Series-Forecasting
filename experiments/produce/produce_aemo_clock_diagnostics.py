"""Clock-behaviour diagnostics for every post-tuning AEMO detector run
(req. 6): raw signal count, accepted detection count, accepted
detections/year, consecutive-gap statistics (median, most frequent), and
checks for detector-specific "clock-like" firing:

    - Page-Hinkley firing mainly every `min_instances` samples;
    - KSWIN firing mainly every `window_size` samples;
    - ADWIN gaps mainly multiples of 32 samples;

plus a cross-region (NSW1 vs. SA1) similarity check on accepted-detection
dates/counts. Reads the raw/accepted detection dumps
detection_artifacts.py persisted and each run's config.json -- no
detector is re-run here.

Saved to results/tables/aemo_clock_diagnostics.csv and
aemo_clock_diagnostics_cross_region.csv (derived tables, not new
runs.csv columns).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.results_io import RUNS_CSV, RUNS_DIR, TABLES_DIR
from experiments.run.detection.detection_artifacts import accepted_detections_path, raw_detections_path

OUTPUT = TABLES_DIR / "aemo_clock_diagnostics.csv"
CROSS_REGION_OUTPUT = TABLES_DIR / "aemo_clock_diagnostics_cross_region.csv"

SPLIT_IDS = (
    "aemo_detect_raw_pre_tune_v1",
    "aemo_detect_raw_post_tune_v1",
    "aemo_detect_standard_daily_pre_tune_v1",
    "aemo_detect_standard_daily_post_tune_v1",
    "aemo_detect_standard_half_hourly_pre_tune_v1",
    "aemo_detect_standard_half_hourly_post_tune_v1",
)

PERIOD_TOLERANCE_FRACTION = 0.15  # +/- 15% of the period being checked
PERIOD_TOLERANCE_FLOOR_DAYS = 0.25  # never tighter than 6 hours
CLOCK_LIKE_THRESHOLD = 0.5
IDENTICAL_THRESHOLD = 0.9


def _gaps_in_days(timestamps) -> np.ndarray:
    ts = pd.DatetimeIndex(sorted(pd.to_datetime(list(timestamps))))
    if len(ts) < 2:
        return np.array([])
    return np.diff(ts.values).astype("timedelta64[s]").astype(float) / 86400.0


def _samples_to_days(n_samples: float, cadence: str) -> float:
    return n_samples / 48.0 if cadence == "half_hourly" else float(n_samples)


def _period_match_fraction(gaps_days: np.ndarray, period_days: float | None) -> float:
    """Fraction of gaps within +/-tolerance of an integer multiple of
    `period_days`. Tolerance scales with the period itself (a fixed
    absolute tolerance would trivially "match" everything for a short
    period like ADWIN's 32-sample ~0.67-day check under half-hourly
    cadence, and be needlessly strict for a long one like a 450-sample
    KSWIN window)."""
    if len(gaps_days) == 0 or not period_days or period_days <= 0:
        return float("nan")
    tolerance_days = max(period_days * PERIOD_TOLERANCE_FRACTION, PERIOD_TOLERANCE_FLOOR_DAYS)
    remainder = np.abs(gaps_days % period_days)
    near_period = np.minimum(remainder, period_days - remainder) <= tolerance_days
    return float(near_period.mean())


def _detector_period_check(method: str, config: dict, cadence: str, gaps_days: np.ndarray) -> dict:
    """Explicitly checks the three named clock-like failure modes."""
    if method == "page_hinkley":
        period_days = _samples_to_days(config.get("min_instances"), cadence)
    elif method == "kswin":
        period_days = _samples_to_days(config.get("window_size"), cadence)
    elif method == "adwin":
        period_days = _samples_to_days(32, cadence)
    else:
        period_days = None

    fraction = _period_match_fraction(gaps_days, period_days)
    return {
        "known_period_days_checked": period_days,
        "fraction_gaps_matching_known_period": fraction,
        "flag_clock_like": bool(fraction >= CLOCK_LIKE_THRESHOLD) if not np.isnan(fraction) else False,
    }


def _infer_cadence(config: dict, split_id: str) -> str:
    if "cadence" in config:
        return config["cadence"]
    return "daily" if "daily" in split_id else "half_hourly"


def load_runs_for_split(split_id: str) -> pd.DataFrame:
    runs = pd.read_csv(RUNS_CSV)
    det = runs[
        (runs["group"] == "detection") & (runs["dataset"] == "aemo") & (runs["split_id"] == split_id)
    ]
    if det.empty:
        return det
    latest_timestamp = det.groupby(["method", "region"])["timestamp"].transform("max")
    return det[det["timestamp"] == latest_timestamp]


def build_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    accepted_by_split: dict[str, dict[tuple[str, str], pd.DatetimeIndex]] = {}

    for split_id in SPLIT_IDS:
        det = load_runs_for_split(split_id)
        if det.empty:
            continue

        for (method, region), group in det.groupby(["method", "region"]):
            config_hash = group["config_hash"].iloc[0]
            config_path = RUNS_DIR / config_hash / "config.json"
            if not config_path.exists():
                continue
            config = json.loads(config_path.read_text())
            cadence = _infer_cadence(config, split_id)

            raw_path = raw_detections_path(config_hash, region)
            accepted_path = accepted_detections_path(config_hash, region)
            if not raw_path.exists() or not accepted_path.exists():
                continue

            raw_ts = pd.to_datetime(pd.read_csv(raw_path)["timestamp"])
            accepted_ts = pd.to_datetime(pd.read_csv(accepted_path)["timestamp"])
            gaps_days = _gaps_in_days(accepted_ts)

            fa_year = group.loc[group["metric_name"] == "accepted_detections_per_year", "metric_value"]

            row = {
                "split_id": split_id,
                "detector": method,
                "region": region,
                "cadence": cadence,
                "raw_signal_count": len(raw_ts),
                "accepted_detection_count": len(accepted_ts),
                "accepted_detections_per_year": float(fa_year.iloc[0]) if len(fa_year) else float("nan"),
                "median_gap_days": float(np.median(gaps_days)) if len(gaps_days) else float("nan"),
                "most_frequent_gap_days": (
                    float(pd.Series(np.round(gaps_days)).mode().iloc[0]) if len(gaps_days) else float("nan")
                ),
                "n_gaps": len(gaps_days),
            }
            row.update(_detector_period_check(method, config, cadence, gaps_days))
            all_rows.append(row)

            accepted_by_split.setdefault(split_id, {})[(method, region)] = accepted_ts

    cross_rows = []
    for split_id, by_method_region in accepted_by_split.items():
        methods = {m for m, _ in by_method_region}
        for method in sorted(methods):
            sa1 = by_method_region.get((method, "SA1"))
            nsw1 = by_method_region.get((method, "NSW1"))
            if sa1 is None or nsw1 is None:
                continue
            dates_sa1 = {pd.Timestamp(t).normalize() for t in sa1}
            dates_nsw1 = {pd.Timestamp(t).normalize() for t in nsw1}
            union = dates_sa1 | dates_nsw1
            jaccard = len(dates_sa1 & dates_nsw1) / len(union) if union else float("nan")
            same_count = len(sa1) == len(nsw1)
            cross_rows.append(
                {
                    "split_id": split_id,
                    "detector": method,
                    "sa1_count": len(sa1),
                    "nsw1_count": len(nsw1),
                    "same_count": same_count,
                    "date_jaccard_similarity": jaccard,
                    "flag_suspiciously_identical": bool(same_count and jaccard >= IDENTICAL_THRESHOLD),
                }
            )

    return pd.DataFrame(all_rows), pd.DataFrame(cross_rows)


def main() -> None:
    diagnostics, cross_region = build_diagnostics()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(OUTPUT, index=False)
    cross_region.to_csv(CROSS_REGION_OUTPUT, index=False)

    print(diagnostics.to_string(index=False))
    print()
    print(cross_region.to_string(index=False))

    if not diagnostics.empty:
        flagged = diagnostics[diagnostics["flag_clock_like"]]
        if not flagged.empty:
            print("\nWARNING: clock-like firing detected:")
            print(
                flagged[
                    ["split_id", "detector", "region", "fraction_gaps_matching_known_period"]
                ].to_string(index=False)
            )

    if not cross_region.empty:
        cross_flagged = cross_region[cross_region["flag_suspiciously_identical"]]
        if not cross_flagged.empty:
            print("\nWARNING: suspiciously identical NSW1/SA1 detections:")
            print(cross_flagged.to_string(index=False))

    print(f"\nwrote {OUTPUT}")
    print(f"wrote {CROSS_REGION_OUTPUT}")


if __name__ == "__main__":
    main()
