"""Final synthetic-only detector tuning against the false-alarm budget.

One tuning pass, not split by deployment cadence. `SAMPLES_PER_YEAR` is
fixed at the synthetic generator's own native assumption -- `make_series`
already models `_PERIOD = 48`, i.e. half-hourly data:

    SAMPLES_PER_YEAR = 48 * 365

(A per-cadence split was tried and reverted: it forced re-imagining the
same fixed N=20,000 series as spanning a different number of years
depending on which cadence was being scored -- e.g. treating it as ~55
simulated years under a "daily" assumption -- which produced misleading
budget comparisons rather than a genuine cadence-specific result. The
AEMO side's own annualisation, `accepted_detections_per_year` in
run_harness.py, is unaffected by this: it's computed from real elapsed
calendar days, not from a samples-per-year assumption, so it never
depended on this cadence split in the first place.)

Synthetic benchmark (unchanged): N=20,000, noise=1.0, kinds = none /
sudden / gradual / recurring, the shared five seeds, detectors = ADWIN /
KSWIN / Page-Hinkley. No AEMO data or AEMO events are used here.

Eligibility per detector configuration -- checked per row, across every
detector and every seed, not as an average:
    every no-drift seed has exactly zero false alarms (`none_clean`,
        strict -- there is no tolerance window to argue about on a pure
        no-drift series, so any detection there is unambiguously wrong)
    every row's false-alarms/year (no-drift AND every drift scenario)
        is <= 2
    total missed drifts across sudden + gradual + recurring == 0

Winner selection per detector, among eligible configs only:
    1. lowest mean detection delay on drift scenarios (none excluded)
    2. lowest mean no-drift false alarms/year
    3. lowest maximum no-drift false alarms/year
    4. deterministic parameter ordering (canonical JSON string)
(Criteria 2-3 are usually a tie among eligible configs, since
`none_clean` forces every eligible config's no-drift rate to exactly 0 --
criterion 1, then 4, does the actual tie-breaking in practice.)
A detector with no eligible configuration is recorded with status
"no_eligible_configuration" rather than a failing fallback.

Runtime: a fresh detector instance is created for every single
(configuration, kind, seed) combination -- detector state is never
reused across calls. Within one configuration's unit, all five no-drift
seeds run first; a unit that fails `none_clean` skips its drift
scenarios entirely, and a unit stops at its first missed drift or first
over-budget row since it can no longer become eligible either way.
Independent units run in worker processes via ProcessPoolExecutor;
workers only return data, never write a file -- the parent process is
the only writer of runs.csv, config.json and the checkpoint files. A
completed unit's checkpoint file (under results/runs/<config_hash>/)
lets a re-run skip units that are already done instead of recomputing
and re-logging them -- but a checkpoint encodes the eligibility rule
that was active when it was written, so a rule change requires clearing
old checkpoints, not just re-running.

Outputs (unchanged filenames):
    results/tables/fine_tune_on_synthetic.csv          -- every row
    results/tables/fine_tune_on_synthetic_winners.csv  -- one row per
        eligible detector

Scalar metrics (n_detections, n_false_alarms, false_alarms_per_year,
missed_detections, detection_delay, runtime) are written to runs.csv
through the shared harness (experiments.run_harness.record_run), which
still uses `drift_lab.evaluation.evaluate_detections` as the one trusted
metric implementation -- this script never recomputes a metric it also
sends through record_run.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from drift_lab.config import SEEDS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series
from experiments import results_io
from experiments.run_harness import record_run

DRIFT_KINDS = ("sudden", "gradual", "recurring")
KINDS = ("none", *DRIFT_KINDS)

DETECTOR_NAMES = ("adwin", "kswin", "page_hinkley")

N = 20_000
NOISE = 1.0

SAMPLES_PER_YEAR = 48 * 365

FALSE_ALARM_BUDGET_PER_YEAR = 2.0

OUTPUT = results_io.TABLES_DIR / "fine_tune_on_synthetic.csv"
WINNERS_OUTPUT = results_io.TABLES_DIR / "fine_tune_on_synthetic_winners.csv"


def false_alarms_per_year(n_false_alarms: int, n_observations: int) -> float:
    return n_false_alarms / n_observations * SAMPLES_PER_YEAR


def candidate_sweeps():
    """Yield (sweep_name, detector_class, kwargs) for every candidate
    configuration. Classes + kwargs only (not instances), so each worker
    builds its own fresh instances -- nothing built here crosses into a
    worker process and gets reused."""

    for delta in (0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002):
        yield ("adwin_budget_delta", ADWINDetector, {"delta": delta})

    for alpha in (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05):
        for window_size in (300, 450):
            for stat_size in (40, 44, 48, 50, 55, 60, 65):
                yield (
                    "kswin_budget_grid",
                    KSWINDetector,
                    {
                        "alpha": alpha,
                        "window_size": window_size,
                        "stat_size": stat_size,
                        "seed": 42,
                    },
                )

    for delta in (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05):
        for min_instances in (20, 30, 50):
            for threshold in (200.0, 300.0, 400.0, 500.0, 750.0, 1000.0):
                yield (
                    "page_hinkley_budget_grid",
                    PageHinkleyDetector,
                    {
                        "min_instances": min_instances,
                        "delta": delta,
                        "threshold": threshold,
                    },
                )


def full_config(kwargs: dict) -> dict:
    """Detector kwargs plus samples_per_year -- the identity a
    config_hash is computed from."""
    return {**kwargs, "samples_per_year": SAMPLES_PER_YEAR}


def _run_one(detector_class, kwargs: dict, kind: str, seed: int) -> dict:
    """Fresh detector, fresh series lookup, one (kind, seed) call."""
    series, truth = make_series(kind=kind, n=N, noise=NOISE, seed=seed)

    detector = detector_class(**kwargs)
    t0 = time.perf_counter()
    detected = detector.detect(series)
    wall_clock_s = time.perf_counter() - t0

    evaluation = evaluate_detections(
        detected_changepoints=detected,
        true_changepoints=truth,
        n_observations=len(series),
        drift_type=kind,
    )
    n_false = len(evaluation["false_alarm_indices"])

    return {
        "kind": kind,
        "seed": seed,
        "detected": list(detected),
        "truth": list(truth),
        "n_observations": len(series),
        "n_detections": len(detected),
        "n_false_alarms": n_false,
        "false_alarms_per_year": false_alarms_per_year(n_false, len(series)),
        "missed_detections": evaluation["missed_detections"],
        "detection_delay": evaluation["detection_delay"],
        "wall_clock_s": wall_clock_s,
    }


def evaluate_unit(sweep_name: str, detector_class, kwargs: dict) -> dict:
    """Runs inside a worker process. Pure computation: no file I/O, no
    record_run, no shared state -- only returns data for the parent to
    log and persist.

    Runtime shortcuts applied here, in order:
      1. every no-drift seed runs first;
      2. if any no-drift seed has a false alarm, drift scenarios are
         skipped entirely for this unit;
      3. otherwise drift scenarios run kind by kind, seed by seed, and
         stop at the first missed drift or over-budget row, since one
         already fails eligibility regardless of what the remaining
         seeds/kinds would have shown.
    """
    method = detector_class.name
    rows: list[dict] = []

    none_fa_per_year = []
    for seed in SEEDS:
        row = _run_one(detector_class, kwargs, "none", seed)
        rows.append(row)
        none_fa_per_year.append(row["false_alarms_per_year"])

    mean_none_fa_year = sum(none_fa_per_year) / len(none_fa_per_year)
    max_none_fa_year = max(none_fa_per_year)

    # Strict: every single no-drift seed must have exactly zero false
    # alarms. There is no tolerance window to argue about on a pure
    # no-drift series -- any detection there is unambiguously wrong, so
    # this is a hard per-seed gate, not folded into the annualised
    # budget below. (`none_clean` implies the budget trivially, since
    # 0 <= FALSE_ALARM_BUDGET_PER_YEAR.)
    none_clean = all(row["n_false_alarms"] == 0 for row in rows)

    stopped_early = False
    if none_clean:
        for kind in DRIFT_KINDS:
            if stopped_early:
                break
            for seed in SEEDS:
                row = _run_one(detector_class, kwargs, kind, seed)
                rows.append(row)
                # The <=2/year budget applies per row here -- every
                # detector, every seed, every kind -- not just as a mean
                # over the no-drift seeds. One missed drift or one
                # over-budget row already rules the config out, so
                # evaluation stops at the first of either.
                if (
                    row["missed_detections"] > 0
                    or row["false_alarms_per_year"] > FALSE_ALARM_BUDGET_PER_YEAR
                ):
                    stopped_early = True
                    break

    eligible = none_clean and not stopped_early

    return {
        "sweep": sweep_name,
        "method": method,
        "kwargs": kwargs,
        "eligible": eligible,
        "mean_none_false_alarms_per_year": mean_none_fa_year,
        "max_none_false_alarms_per_year": max_none_fa_year,
        "rows": rows,
    }


# --- parent-only I/O: checkpoints, runs.csv logging, output tables --------


def checkpoint_path(config_hash: str):
    return results_io.RUNS_DIR / config_hash / "fine_tune_checkpoint.json"


def load_checkpoint(config_hash: str) -> dict | None:
    path = checkpoint_path(config_hash)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_checkpoint(config_hash: str, unit: dict) -> None:
    path = checkpoint_path(config_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(unit))


def log_unit_to_runs_csv(config_hash: str, config: dict, unit: dict) -> None:
    """The only place this script calls record_run -- always in the
    parent process, never inside a worker."""
    for row in unit["rows"]:
        record_run(
            method=unit["method"],
            dataset=f"synthetic_{row['kind']}",
            seed=row["seed"],
            config=config,
            wall_clock_s=row["wall_clock_s"],
            split_id=f"synth_n{N}_finetune_{row['kind']}",
            detection=(row["detected"], row["truth"], row["n_observations"]),
            samples_per_year=SAMPLES_PER_YEAR,
        )


def select_winners(completed_units: dict[str, dict]) -> pd.DataFrame:
    """One winner per detector among eligible units only. Pure function
    of `completed_units`, so re-running it on the same input is the
    determinism check `validate_before_saving` relies on."""

    by_method: dict[str, list[dict]] = {}
    for config_hash, unit in completed_units.items():
        if not unit["eligible"]:
            continue
        drift_delays = [row["detection_delay"] for row in unit["rows"] if row["kind"] != "none"]
        mean_delay = sum(drift_delays) / len(drift_delays) if drift_delays else float("nan")
        by_method.setdefault(unit["method"], []).append(
            {
                "config_hash": config_hash,
                "kwargs": unit["kwargs"],
                "mean_detection_delay": mean_delay,
                "mean_none_false_alarms_per_year": unit["mean_none_false_alarms_per_year"],
                "max_none_false_alarms_per_year": unit["max_none_false_alarms_per_year"],
            }
        )

    winners = []
    for method in DETECTOR_NAMES:
        candidates = by_method.get(method, [])

        if not candidates:
            winners.append(
                {
                    "detector": method,
                    "config_hash": None,
                    "detector_parameters": None,
                    "mean_detection_delay": None,
                    "mean_none_false_alarms_per_year": None,
                    "max_none_false_alarms_per_year": None,
                    "status": "no_eligible_configuration",
                }
            )
            continue

        ranked = sorted(
            candidates,
            key=lambda c: (
                c["mean_detection_delay"],
                c["mean_none_false_alarms_per_year"],
                c["max_none_false_alarms_per_year"],
                json.dumps(c["kwargs"], sort_keys=True),
            ),
        )
        winner = ranked[0]
        winners.append(
            {
                "detector": method,
                "config_hash": winner["config_hash"],
                "detector_parameters": json.dumps(winner["kwargs"], sort_keys=True),
                "mean_detection_delay": winner["mean_detection_delay"],
                "mean_none_false_alarms_per_year": winner["mean_none_false_alarms_per_year"],
                "max_none_false_alarms_per_year": winner["max_none_false_alarms_per_year"],
                "status": "selected",
            }
        )

    return pd.DataFrame(winners)


def validate_before_saving(
    full_results: pd.DataFrame,
    winners: pd.DataFrame,
    completed_units: dict[str, dict],
) -> None:
    """Every check runs before the winners file is written; a failure
    raises instead of silently saving a questionable pick."""

    assert set(full_results["detector"]) == set(DETECTOR_NAMES), "all three detectors must be present"
    assert set(full_results["seed"]) == set(SEEDS), "all five seeds must have been used"

    selected = winners[winners["status"] == "selected"]
    for _, winner in selected.iterrows():
        unit = completed_units[winner["config_hash"]]

        kinds_present = {row["kind"] for row in unit["rows"]}
        assert kinds_present == set(KINDS), (
            f"winner {winner['detector']} is missing kinds "
            f"{set(KINDS) - kinds_present}, so it cannot truly be eligible"
        )
        assert all(row["missed_detections"] == 0 for row in unit["rows"]), (
            f"winner {winner['detector']} has a missed drift"
        )
        assert all(
            row["n_false_alarms"] == 0 for row in unit["rows"] if row["kind"] == "none"
        ), (
            f"winner {winner['detector']} has a false alarm on the no-drift stream"
        )
        assert all(
            row["false_alarms_per_year"] <= FALSE_ALARM_BUDGET_PER_YEAR
            for row in unit["rows"]
        ), (
            f"winner {winner['detector']} exceeds the false-alarm budget on at least one row"
        )
        # Every row was produced by _run_one(), which always builds
        # `detector_class(**kwargs)` fresh immediately before the single
        # .detect() call it's used for -- no detector instance is ever
        # stored and reused across rows, units, kinds or seeds.

    # Determinism: selecting winners twice from the same completed units
    # must produce the identical table.
    winners_again = select_winners(completed_units)
    pd.testing.assert_frame_equal(
        winners.reset_index(drop=True), winners_again.reset_index(drop=True)
    )


def main() -> None:
    candidates = list(candidate_sweeps())

    completed_units: dict[str, dict] = {}
    pending = []

    for sweep_name, detector_class, kwargs in candidates:
        config = full_config(kwargs)
        config_hash = results_io.config_hash(config)

        cached = load_checkpoint(config_hash)
        if cached is not None:
            completed_units[config_hash] = cached
        else:
            pending.append((config_hash, config, sweep_name, detector_class, kwargs))

    if pending:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_unit, sweep_name, detector_class, kwargs): (
                    config_hash,
                    config,
                )
                for config_hash, config, sweep_name, detector_class, kwargs in pending
            }
            for future in as_completed(futures):
                config_hash, config = futures[future]
                unit = future.result()

                log_unit_to_runs_csv(config_hash, config, unit)
                save_checkpoint(config_hash, unit)
                completed_units[config_hash] = unit

    result_rows = []
    for config_hash, unit in completed_units.items():
        for row in unit["rows"]:
            result_rows.append(
                {
                    "detector": unit["method"],
                    "detector_parameters": json.dumps(unit["kwargs"], sort_keys=True),
                    "config_hash": config_hash,
                    "kind": row["kind"],
                    "seed": row["seed"],
                    "detections": row["n_detections"],
                    "false_alarms": row["n_false_alarms"],
                    "false_alarms_per_year": row["false_alarms_per_year"],
                    "missed_drifts": row["missed_detections"],
                    "detection_delay": row["detection_delay"],
                    "eligibility_status": "eligible" if unit["eligible"] else "ineligible",
                }
            )

    full_results = pd.DataFrame(result_rows)
    results_io.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    full_results.to_csv(OUTPUT, index=False)

    winners = select_winners(completed_units)
    validate_before_saving(full_results, winners, completed_units)
    winners.to_csv(WINNERS_OUTPUT, index=False)


if __name__ == "__main__":
    main()
