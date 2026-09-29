"""Synthetic-only detector tuning against the false-alarm budget.

Constrained grid search: every candidate config runs on the synthetic
benchmark (N=20,000, noise=1.0, kinds none/sudden/gradual/recurring, the
shared five seeds). No AEMO data is used. False alarms are annualised at
48*365 samples/year (the generator is half-hourly).

Eligible configs must, on every seed:
    - raise zero false alarms on the no-drift series;
    - stay at <= 2 false alarms/year on every row;
    - miss no drift.
Winner per detector: fewest false alarms (mean false alarms/year over all
rows -- no-drift and every drift scenario), then shortest mean detection
delay, then parameter order. No eligible config ->
status "no_eligible_configuration".

Configs run in parallel worker processes; only the parent writes files.
Per-config checkpoints (results/runs/<config_hash>/) let a re-run skip
finished configs -- clear them if the eligibility rules change.

Outputs:
    results/tables/fine_tune_on_synthetic.csv          every row
    results/tables/fine_tune_on_synthetic_winners.csv  one row per detector
    runs.csv                                           per-row metrics via record_run
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

# delta / alpha act through a logarithm, so they are swept in 10x steps.
# KSWIN alpha starts near 2 / 17,520: the per-test level that keeps <= 2
# false alarms/year when testing once per half-hour.
ADWIN_DELTAS = tuple(10.0**-k for k in range(3, 10))  # 1e-3 ... 1e-9
ADWIN_CLOCKS = (32, 48, 336)  # check every 16 h (river default), 1 day, 1 week
ADWIN_MIN_WINDOW_LENGTHS = (5, 48, 336, 1344)  # river default, 1 day, 1 week, 4 weeks
KSWIN_ALPHAS = tuple(10.0**-k for k in range(4, 10))  # 1e-4 ... 1e-9
# KSWIN sizes in whole days of half-hours. The reference window covers at
# least a full weekly cycle and at most ~a month (longer spans mix seasonal
# demand levels); the recent sample holds whole days.
KSWIN_STAT_SIZES = (96, 144, 240, 336, 672)  # 2, 3, 5 days, 1, 2 weeks
KSWIN_WINDOW_SIZES = (336, 480, 672, 1344)  # 7, 10, 14, 28 days

OUTPUT = results_io.TABLES_DIR / "fine_tune_on_synthetic.csv"
WINNERS_OUTPUT = results_io.TABLES_DIR / "fine_tune_on_synthetic_winners.csv"


def false_alarms_per_year(n_false_alarms: int, n_observations: int) -> float:
    return n_false_alarms / n_observations * SAMPLES_PER_YEAR


def candidate_sweeps():
    """Yield (sweep_name, detector_class, kwargs) per candidate; workers
    build fresh instances."""

    for delta in ADWIN_DELTAS:
        for clock in ADWIN_CLOCKS:
            for min_window_length in ADWIN_MIN_WINDOW_LENGTHS:
                yield (
                    "adwin_budget_grid",
                    ADWINDetector,
                    {"delta": delta, "clock": clock, "min_window_length": min_window_length},
                )

    size_pairs = [
        (window_size, stat_size)
        for window_size in KSWIN_WINDOW_SIZES
        for stat_size in KSWIN_STAT_SIZES
        if window_size >= 2 * stat_size  # river samples the reference from the rest
    ]
    for alpha in KSWIN_ALPHAS:
        for window_size, stat_size in size_pairs:
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
    """Detector kwargs plus samples_per_year, hashed into config_hash."""
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
    """Worker-side evaluation of one config (no file I/O). Runs no-drift
    seeds first and stops at the first failure, since one failure already
    makes the config ineligible."""
    method = detector_class.name
    rows: list[dict] = []

    none_fa_per_year = []
    for seed in SEEDS:
        row = _run_one(detector_class, kwargs, "none", seed)
        rows.append(row)
        none_fa_per_year.append(row["false_alarms_per_year"])

    mean_none_fa_year = sum(none_fa_per_year) / len(none_fa_per_year)
    max_none_fa_year = max(none_fa_per_year)

    # Any alarm on a no-drift series is wrong, so this gate is per seed.
    none_clean = all(row["n_false_alarms"] == 0 for row in rows)

    stopped_early = False
    if none_clean:
        for kind in DRIFT_KINDS:
            if stopped_early:
                break
            for seed in SEEDS:
                row = _run_one(detector_class, kwargs, kind, seed)
                rows.append(row)
                # Budget is per row, not averaged.
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
    """Log one config's rows to runs.csv (parent process only)."""
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
    """One winner per detector among eligible configs (pure function)."""

    by_method: dict[str, list[dict]] = {}
    for config_hash, unit in completed_units.items():
        if not unit["eligible"]:
            continue
        drift_delays = [row["detection_delay"] for row in unit["rows"] if row["kind"] != "none"]
        mean_delay = sum(drift_delays) / len(drift_delays) if drift_delays else float("nan")
        fa_per_year = [row["false_alarms_per_year"] for row in unit["rows"]]
        by_method.setdefault(unit["method"], []).append(
            {
                "config_hash": config_hash,
                "kwargs": unit["kwargs"],
                "mean_false_alarms_per_year": sum(fa_per_year) / len(fa_per_year),
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
                    "mean_false_alarms_per_year": None,
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
                c["mean_false_alarms_per_year"],
                c["mean_detection_delay"],
                json.dumps(c["kwargs"], sort_keys=True),
            ),
        )
        winner = ranked[0]
        winners.append(
            {
                "detector": method,
                "config_hash": winner["config_hash"],
                "detector_parameters": json.dumps(winner["kwargs"], sort_keys=True),
                "mean_false_alarms_per_year": winner["mean_false_alarms_per_year"],
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
    """Re-check winners before saving; raise rather than save a bad pick."""

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

    # Selection must be deterministic.
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
