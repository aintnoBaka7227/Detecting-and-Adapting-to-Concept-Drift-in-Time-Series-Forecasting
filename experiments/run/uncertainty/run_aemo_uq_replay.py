"""Step 6 / T5, F4 -- prediction intervals for every adaptation run.

No model is refitted. For the latest logged run of every (model, arm,
region, seed) this replays the saved per-timestamp curve (actual,
forecast) in chronological order and issues a 90% interval for every TEST
half-hour with two methods, both calibrated on that model x region x
seed's Calibration-window residuals (run_aemo_calibration_forecasts.py):

    fixed_split_conformal  one radius, set once, never updated;
    agaci                  AgACI over a rolling 7-day error buffer
                           (drift_lab.uncertainty.AgACI): one interval per
                           half-hour, built from every outcome up to the
                           previous half-hour, then fed its own outcome
                           before the next is issued. The point forecast
                           is the saved one and is never revised.

Both methods are driven only through the shared UQ interface
(base.py): `quantify` one half-hour, then `observe` its outcome.

Arm D is replayed at the 180-day window only, as in T4. A run with no
calibration file is skipped.

Outputs per run:

    results/runs/<adaptation hash>/intervals_aemo_<region>_<seed>.csv
        timestamp, regime, actual, forecast, both methods' bounds and
        rolling_coverage_7d_<uq method>: each method's coverage over the
        last 7 days (336 half-hours, the window of F1's rolling MAE), read
        by F4 (NaN for the first 335 half-hours);
    runs.csv rows under group "uncertainty", split_id
        "aemo_uq_<detector>_half_hourly_v1", method
        "<model>/<arm>/<uq method>": coverage, mean_interval_width,
        normalised_interval_width, worst_24h_coverage and
        n_complete_windows per regime (evaluation.
        calculate_uq_metrics_by_regime) and over the whole stream, plus
        pinball_loss and width_scale over the whole stream.

Worst 24h coverage uses rolling windows of 48 consecutive half-hours
lying entirely inside one regime; timestamps are passed so a window that
spans a gap is not eligible. n_complete_windows counts the eligible
(overlapping) windows.

Normalised width divides by the region's mean TRAIN demand
(evaluation.calculate_width_scale), computed once per region and shared
by every model, arm, UQ method, seed and regime.

AgACI aggregates in R (opera), one call per half-hour: about a minute per
run, so runs are replayed in parallel worker processes.

Usage:
    python -m experiments.run.uncertainty.run_aemo_uq_replay
    python -m experiments.run.uncertainty.run_aemo_uq_replay --model xgboost --region SA1 --workers 4
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from drift_lab import evaluation
from drift_lab.aemo import loader
from drift_lab.uncertainty import AgACI, FixedSplitConformal, UncertaintyQuantifier
from experiments import results_io
from experiments.produce.table_adaptation_common import D_PREFIX, latest_adaptation_rows
from experiments.run.adaptation.run_aemo_adaptation_arms import arm_d
from experiments.run_harness import build_row
from experiments.trigger_detectors import (
    CONTROL_ARMS,
    CONTROL_DETECTOR,
    TRIGGER_DETECTORS,
    uq_split_id,
)

TARGET_ALPHA = 0.10
ISSUANCE_STEP = 1  # forecasts per issuance: one half-hour, feedback after each
WORST_COVERAGE_WINDOW = 48  # 24 hours
ROLLING_COVERAGE_WINDOW = 336  # 7 days, F1's rolling-MAE window; F4 plots it
REPORTED_D_ARM = arm_d(180)
FIXED, AGACI = "fixed_split_conformal", "agaci"


def replay(
    uq: UncertaintyQuantifier, forecast: np.ndarray, actual: np.ndarray, residuals: np.ndarray
):
    """Issue every interval through the shared UQ interface: `quantify` one
    half-hour, then `observe` its outcome before the next (base.py)."""
    lower, upper = np.empty(len(actual)), np.empty(len(actual))
    for start in range(0, len(actual), ISSUANCE_STEP):
        block = slice(start, start + ISSUANCE_STEP)
        issued = uq.quantify(forecast[block], residuals)
        lower[block], upper[block] = issued.lower, issued.upper
        uq.observe(actual[block])
    return lower, upper


def add_rolling_coverage(intervals: pd.DataFrame) -> pd.DataFrame:
    """Add each UQ method's rolling_coverage_7d_<method> column, in place,
    over the full chronological stream (evaluation.calculate_rolling_coverage)."""
    for uq_method in (FIXED, AGACI):
        intervals[f"rolling_coverage_7d_{uq_method}"] = evaluation.calculate_rolling_coverage(
            intervals["actual"],
            intervals[f"lower_{uq_method}"],
            intervals[f"upper_{uq_method}"],
            window=ROLLING_COVERAGE_WINDOW,
        ).to_numpy()
    return intervals


def width_scale(region: str) -> float:
    """Normalised-width denominator: the region's mean TRAIN demand."""
    train, _, _ = loader.load(region)
    return evaluation.calculate_width_scale(train["TOTALDEMAND"])


def metric_rows(
    common: dict, actual, lower, upper, regimes, scale: float, timestamps
) -> list[dict]:
    rows = [
        build_row(common, "coverage", evaluation.calculate_empirical_coverage(actual, lower, upper), "full"),
        build_row(common, "mean_interval_width", evaluation.calculate_mean_interval_width(lower, upper), "full"),
        build_row(
            common,
            "normalised_interval_width",
            evaluation.calculate_normalised_interval_width(lower, upper, scale),
            "full",
        ),
        build_row(common, "width_scale", scale, "full"),
        build_row(
            common,
            "worst_24h_coverage",
            evaluation.calculate_worst_rolling_coverage(
                actual, lower, upper, WORST_COVERAGE_WINDOW, timestamps=timestamps
            ),
            "full",
        ),
        build_row(
            common,
            "n_complete_windows",
            evaluation.calculate_complete_window_count(
                len(actual), WORST_COVERAGE_WINDOW, timestamps=timestamps
            ),
            "full",
        ),
        build_row(
            common,
            "pinball_loss",
            evaluation.calculate_interval_pinball_loss(actual, lower, upper, alpha=TARGET_ALPHA),
            "full",
        ),
    ]
    by_regime = evaluation.calculate_uq_metrics_by_regime(
        actual,
        lower,
        upper,
        regimes,
        window=WORST_COVERAGE_WINDOW,
        width_scale=scale,
        timestamps=timestamps,
    )
    for regime, metrics in by_regime.items():
        name = regime.replace("_", "-")  # runs.csv spelling: pre-drift / drift / post-drift
        rows.append(build_row(common, "coverage", metrics["coverage"], name))
        rows.append(build_row(common, "mean_interval_width", metrics["mean_interval_width"], name))
        rows.append(
            build_row(common, "normalised_interval_width", metrics["normalised_interval_width"], name)
        )
        rows.append(build_row(common, "worst_24h_coverage", metrics["worst_rolling_coverage"], name))
        rows.append(build_row(common, "n_complete_windows", metrics["n_complete_windows"], name))
        rows.append(build_row(common, "n_observations", metrics["n_observations"], name))
    return rows


def replay_one(
    method: str, region: str, seed: int | None, source_hash: str, scale: float, split_id: str
) -> tuple[str, list[dict]]:
    """Replay one adaptation run; returns a progress note and its runs.csv rows."""
    model = method.split("/", 1)[0]
    label = f"{method} {region} seed={seed}"
    calibration = results_io.calibration_path(model, region, seed)
    if not calibration.exists():
        return f"{label}: skipped, no calibration file", []

    residuals = pd.read_csv(calibration)["residual"].to_numpy()
    curve = pd.read_csv(
        results_io.curve_path(source_hash, "aemo", region, seed), parse_dates=["timestamp"]
    )
    actual, forecast = curve["actual"].to_numpy(), curve["forecast"].to_numpy()

    # Uncalibrated: the first quantify() calibrates on `residuals`.
    fixed = FixedSplitConformal(alpha=TARGET_ALPHA)
    fixed_lower, fixed_upper = replay(fixed, forecast, actual, residuals)
    agaci = AgACI(target_alpha=TARGET_ALPHA)
    started = time.perf_counter()
    agaci_lower, agaci_upper = replay(agaci, forecast, actual, residuals)
    agaci_seconds = time.perf_counter() - started

    intervals = curve[["timestamp", "regime", "actual", "forecast"]].copy()
    intervals[f"lower_{FIXED}"], intervals[f"upper_{FIXED}"] = fixed_lower, fixed_upper
    intervals[f"lower_{AGACI}"], intervals[f"upper_{AGACI}"] = agaci_lower, agaci_upper
    add_rolling_coverage(intervals)
    intervals.to_csv(results_io.intervals_path(source_hash, "aemo", region, seed), index=False)

    methods = {
        FIXED: (fixed_lower, fixed_upper, fixed.describe(), 0.0),
        AGACI: (agaci_lower, agaci_upper, agaci.describe(), agaci_seconds),
    }
    rows: list[dict] = []
    for uq_method, (lower, upper, description, seconds) in methods.items():
        config = {
            "source_config_hash": source_hash,
            "issuance_step": ISSUANCE_STEP,
            "worst_coverage_window": WORST_COVERAGE_WINDOW,
            "uq": description,
        }
        chash = results_io.config_hash(config)
        results_io.dump_config(chash, config)
        common = {
            "group": "uncertainty",
            "method": f"{method}/{uq_method}",
            "dataset": "aemo",
            "region": region,
            "seed": seed,
            "config_hash": chash,
            "split_id": split_id,
            "n_retrains": 0,
            "train_samples": 0,
            "wall_clock_s": seconds,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        rows.extend(
            metric_rows(
                common, actual, lower, upper, curve["regime"].to_numpy(), scale, curve["timestamp"]
            )
        )
    return f"{label}: replayed in {agaci_seconds:.0f}s", rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model")
    parser.add_argument("--region")
    parser.add_argument("--detector", choices=TRIGGER_DETECTORS, default=None, help="default: all")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    runs = []
    for detector in [args.detector] if args.detector else TRIGGER_DETECTORS:
        rows = latest_adaptation_rows(detector).drop_duplicates(["method", "region", "seed_key"])
        if detector != CONTROL_DETECTOR:  # the shared controls are replayed in the control batch
            rows = rows[~rows["arm"].isin(CONTROL_ARMS)]
        runs.append(rows.assign(split_id=uq_split_id(detector)))
    runs = pd.concat(runs)
    other_windows = runs["arm"].str.startswith(D_PREFIX) & (runs["arm"] != REPORTED_D_ARM)
    runs = runs[~other_windows]
    if args.model:
        runs = runs[runs["model"] == args.model]
    if args.region:
        runs = runs[runs["region"] == args.region]
    scales = {region: width_scale(region) for region in runs["region"].unique()}
    jobs = [
        (
            run.method,
            run.region,
            None if run.seed_key == -1 else int(run.seed_key),
            run.config_hash,
            scales[run.region],
            run.split_id,
        )
        for run in runs.itertuples()
    ]
    print(f"replaying {len(jobs)} runs with {args.workers} workers", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(replay_one, *job) for job in jobs]
        for done, future in enumerate(as_completed(futures), start=1):
            note, rows = future.result()
            if rows:
                results_io.append_runs(rows)
            print(f"[{done}/{len(jobs)}] {note}", flush=True)


if __name__ == "__main__":
    main()
