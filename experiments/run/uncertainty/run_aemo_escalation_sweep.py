"""Step 6 / T6, F5 -- width-threshold escalation sweep for every UQ run.

No interval is re-issued. For the latest replay of every (model, arm,
region, UQ method, seed) logged by run_aemo_uq_replay.py, this reads the
saved intervals (located through the UQ run's config.json ->
source_config_hash) and sweeps the escalation rule the UQ classes apply to
their own flag:

    escalate  <=>  upper - lower > tau

over every distinct width tau (evaluation.calculate_escalation_curve).
Escalation hands a forecast to a human and never feeds back into the UQ
state, so the bounds do not depend on tau and sweeping the saved bounds is
the same as re-issuing at each tau.

Outputs per UQ run:

    results/runs/<escalation hash>/escalation_curve_aemo_<region>_<seed>.csv
        regime (full / pre_drift / drift / post_drift), threshold,
        review_rate, retained_coverage, n_retained, n_reviewed,
        n_observations. Regimes share the full stream's threshold grid;
    runs.csv rows under group "escalation", split_id
        "aemo_escalation_<detector>_half_hourly_v1", method
        "<model>/<arm>/<uq method>", config_hash = the escalation hash
        above (so producers can find the curve file):
        review_rate_for_coverage_0.90 / _0.95 (NaN = unreachable),
        coverage_at_review_0.05 / _0.10 / _0.20 with the review rate
        actually attained (actual_review_at_*), n_operating_points and
        n_observations, per regime and over the full stream.

A constant-width method (fixed split conformal) has one width, so its only
operating points are 0% and 100% review; that is logged as is.

MIN_RETAINED is provisional: a target coverage counts as reached only when
at least this many forecasts are kept.

Usage:
    python -m experiments.run.uncertainty.run_aemo_escalation_sweep
    python -m experiments.run.uncertainty.run_aemo_escalation_sweep --model xgboost --region SA1
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
MIN_RETAINED = 1  # provisional: any non-empty retained set


def sweep_curves(actual, lower, upper, regimes) -> pd.DataFrame:
    """Full-stream curve plus one curve per regime, on one threshold grid."""
    full = evaluation.calculate_escalation_curve(actual, lower, upper)
    full.insert(0, "regime", "full")
    by_regime = evaluation.calculate_escalation_curves_by_regime(actual, lower, upper, regimes)
    return pd.concat([full, by_regime], ignore_index=True)


def sweep_one(run) -> list[dict]:
    """Sweep one UQ run; returns its runs.csv rows."""
    seed = None if run.seed_key == -1 else int(run.seed_key)
    uq_config = json.loads((results_io.RUNS_DIR / run.config_hash / "config.json").read_text())
    source_hash = uq_config["source_config_hash"]
    intervals = pd.read_csv(results_io.intervals_path(source_hash, "aemo", run.region, seed))
    curves = sweep_curves(
        intervals["actual"],
        intervals[f"lower_{run.uq}"],
        intervals[f"upper_{run.uq}"],
        intervals["regime"],
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
    args = parser.parse_args()

    runs = []
    for detector in [args.detector] if args.detector else TRIGGER_DETECTORS:
        rows = latest_uq_rows(detector).drop_duplicates(["method", "region", "seed_key"])
        if detector != CONTROL_DETECTOR:  # the shared controls are swept in the control batch
            rows = rows[~rows["arm"].isin(CONTROL_ARMS)]
        runs.append(rows.assign(split_id=escalation_split_id(detector)))
    runs = pd.concat(runs)
    if args.model:
        runs = runs[runs["model"] == args.model]
    if args.region:
        runs = runs[runs["region"] == args.region]
    print(f"sweeping {len(runs)} UQ runs", flush=True)

    rows: list[dict] = []
    for done, run in enumerate(runs.itertuples(), start=1):
        rows.extend(sweep_one(run))
        print(f"[{done}/{len(runs)}] {run.method} {run.region} seed={run.seed_key}", flush=True)
    results_io.append_runs(rows)
    print(f"logged {len(rows)} rows to {results_io.RUNS_CSV}")


if __name__ == "__main__":
    main()
