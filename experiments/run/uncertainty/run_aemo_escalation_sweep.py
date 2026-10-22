"""Step 6 / T6, F5 -- escalation threshold sweep for every UQ run.

No interval is re-issued: escalation never feeds back into the UQ state, so
sweeping the saved bounds equals re-issuing at each threshold. For each
signal in SIGNALS ("the interval width, or the nonconformity score"), every
distinct score is tried as tau with "escalate <=> score > tau"
(evaluation.calculate_escalation_curve):
    width                upper - lower, the UQ classes' own flag rule;
    nonconformity_<L>    mean |actual - forecast| over the previous L
                         half-hours (30min = 1, 1d = 48, 7d = 336), lagged so
                         forecast t sees errors up to t-1; the first TEST
                         forecasts carry in the Calibration window's errors.
A constant-width interval (fixed split conformal) has only 0% and 100%
review under width; a nonconformity signal gives it a full curve.

Outputs per UQ run:
    results/runs/<escalation hash>/escalation_curve_aemo_<region>_<seed>.csv
        review rate and retained coverage at every threshold, per regime;
    runs.csv, group "escalation", split_id escalation_split_id(detector,
        signal): review rate for 0.90 / 0.95 (NaN = unreachable) and coverage
        at 5 / 10 / 20% review, per regime and overall.
MIN_RETAINED (provisional) is the fewest kept forecasts that may reach a target.

Usage:
    python -m experiments.run.uncertainty.run_aemo_escalation_sweep
    python -m experiments.run.uncertainty.run_aemo_escalation_sweep --signals nonconformity_1d
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

import pandas as pd

from drift_lab import evaluation
from drift_lab.evaluation.evaluation import (
    ESCALATION_REVIEW_BUDGETS,
    ESCALATION_TARGET_COVERAGES,
    ESCALATION_WIDTH_DECIMALS,
)
from experiments import results_io
from experiments.produce.produce_table_t5 import latest_uq_rows
from experiments.run_harness import build_row
from experiments.trigger_detectors import (
    CONTROL_ARMS,
    CONTROL_DETECTOR,
    TRIGGER_DETECTORS,
    escalation_split_id,
)

RULE = "escalate when upper - lower > threshold (MW)"
# Escalation signal -> look-back in half-hours (None: the interval width).
SIGNALS = {"width": None, "nonconformity_30min": 1, "nonconformity_1d": 48, "nonconformity_7d": 336}
# How each signal's rule reads in a caption.
SIGNAL_RULES = {
    "width": "Escalate when the 90% interval is wider than τ",
    "nonconformity_30min": "Escalate when the previous half-hour's |actual − forecast| exceeds τ",
    "nonconformity_1d": "Escalate when the mean |actual − forecast| over the previous day (48 half-hours) exceeds τ",
    "nonconformity_7d": "Escalate when the mean |actual − forecast| over the previous week (336 half-hours) exceeds τ",
}
MIN_RETAINED = 1  # provisional: any non-empty retained set


def sweep_curves(actual, lower, upper, regimes, score=None) -> pd.DataFrame:
    """Full-stream curve plus one curve per regime, on one threshold grid."""
    full = evaluation.calculate_escalation_curve(actual, lower, upper, score=score)
    full.insert(0, "regime", "full")
    by_regime = evaluation.calculate_escalation_curves_by_regime(
        actual, lower, upper, regimes, score=score
    )
    return pd.concat([full, by_regime], ignore_index=True)


def nonconformity_score(run, seed, intervals: pd.DataFrame, lookback: int):
    """The signal for every TEST forecast, carrying in the Calibration window."""
    calibration = pd.read_csv(
        results_io.calibration_path(run.model, run.region, seed), float_precision="round_trip"
    )
    return evaluation.calculate_nonconformity_score(
        intervals["actual"],
        intervals["forecast"],
        lookback,
        history_true=calibration["actual"],
        history_pred=calibration["forecast"],
    )


def sweep_one(run, signal: str) -> list[dict]:
    """Sweep one UQ run with one escalation signal; returns its runs.csv rows."""
    seed = None if run.seed_key == -1 else int(run.seed_key)
    uq_config = json.loads((results_io.RUNS_DIR / run.config_hash / "config.json").read_text())
    source_hash = uq_config["source_config_hash"]
    intervals = pd.read_csv(results_io.intervals_path(source_hash, "aemo", run.region, seed))
    lookback = SIGNALS[signal]
    score = None if lookback is None else nonconformity_score(run, seed, intervals, lookback)
    curves = sweep_curves(
        intervals["actual"],
        intervals[f"lower_{run.uq}"],
        intervals[f"upper_{run.uq}"],
        intervals["regime"],
        score,
    )

    config = {
        "uq_config_hash": run.config_hash,
        "intervals_config_hash": source_hash,
        "rule": RULE,
        "width_decimals": ESCALATION_WIDTH_DECIMALS,
        "target_coverages": list(ESCALATION_TARGET_COVERAGES),
        "review_budgets": list(ESCALATION_REVIEW_BUDGETS),
        "min_retained": MIN_RETAINED,
    }
    if lookback is not None:  # width keeps its original config, so its hashes stand
        config.update(
            {
                "signal": signal,
                "rule": "escalate when mean |actual - forecast| over the previous "
                "lookback half-hours > threshold (MW)",
                "lookback_half_hours": lookback,
                "carry_in": "end of the Calibration window",
            }
        )
    chash = results_io.config_hash(config)
    results_io.dump_config(chash, config)
    curves.to_csv(results_io.escalation_curve_path(chash, "aemo", run.region, seed), index=False)

    common = {
        "group": "escalation",
        "method": run.method,
        "dataset": "aemo",
        "region": run.region,
        "seed": seed,
        "config_hash": chash,
        "split_id": run.split_id,
        "n_retrains": 0,
        "train_samples": 0,
        "wall_clock_s": 0.0,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    rows = []
    for regime, curve in curves.groupby("regime", sort=False):
        name = regime.replace("_", "-")  # runs.csv spelling: pre-drift / drift / post-drift
        points = evaluation.calculate_escalation_operating_points(curve, min_retained=MIN_RETAINED)
        points["n_observations"] = int(curve["n_observations"].iloc[0])
        rows.extend(build_row(common, metric, value, name) for metric, value in points.items())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model")
    parser.add_argument("--region")
    parser.add_argument("--detector", choices=TRIGGER_DETECTORS, default=None, help="default: all")
    parser.add_argument(
        "--signals", default=",".join(SIGNALS), help=f"comma list from {tuple(SIGNALS)}; default all"
    )
    args = parser.parse_args()
    signals = [s.strip() for s in args.signals.split(",") if s.strip()]
    unknown = sorted(set(signals) - set(SIGNALS))
    if unknown:
        parser.error(f"unknown signals {unknown}")

    runs = []
    for detector in [args.detector] if args.detector else TRIGGER_DETECTORS:
        rows = latest_uq_rows(detector).drop_duplicates(["method", "region", "seed_key"])
        if detector != CONTROL_DETECTOR:  # the shared controls are swept in the control batch
            rows = rows[~rows["arm"].isin(CONTROL_ARMS)]
        for signal in signals:
            runs.append(rows.assign(signal=signal, split_id=escalation_split_id(detector, signal)))
    runs = pd.concat(runs)
    if args.model:
        runs = runs[runs["model"] == args.model]
    if args.region:
        runs = runs[runs["region"] == args.region]
    print(f"sweeping {len(runs)} UQ runs", flush=True)

    rows: list[dict] = []
    for done, run in enumerate(runs.itertuples(), start=1):
        rows.extend(sweep_one(run, run.signal))
        print(f"[{done}/{len(runs)}] {run.signal} {run.method} {run.region} seed={run.seed_key}", flush=True)
    results_io.append_runs(rows)
    print(f"logged {len(rows)} rows to {results_io.RUNS_CSV}")


if __name__ == "__main__":
    main()
