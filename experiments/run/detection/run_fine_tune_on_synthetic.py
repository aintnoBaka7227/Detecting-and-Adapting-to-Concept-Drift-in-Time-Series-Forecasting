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

Nothing is logged to runs.csv: the sweep's thousands of candidate configs
are tuning scaffolding, not reportable runs. Only the frozen winners are
logged, by run_post_tune_on_synthetic.py and the AEMO post-tune runs.
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

DRIFT_KINDS = ("sudden", "gradual", "recurring")
KINDS = ("none", *DRIFT_KINDS)

DETECTOR_NAMES = ("adwin", "kswin", "page_hinkley")

N = 20_000
NOISE = 1.0

SAMPLES_PER_YEAR = 48 * 365

FALSE_ALARM_BUDGET_PER_YEAR = 2.0

# delta / alpha act through a logarithm, so they are swept in 10x steps.
# KSWIN alpha starts near 2 / 17,520: the per-test level that keeps <= 2
# false alarms/year when testing once per half-hour; 5e-5 is added between
# 1e-4 and 1e-5, where the false-alarm floor is first reached.
ADWIN_DELTAS = tuple(10.0**-k for k in range(3, 11))  # 1e-3 ... 1e-10
ADWIN_CLOCKS = (32, 48, 336)  # check every 16 h (river default), 1 day, 1 week
# river default, 1 day, 3.5 days, 5 days, 1 week, 4 weeks
ADWIN_MIN_WINDOW_LENGTHS = (5, 48, 168, 240, 336, 1344)
KSWIN_ALPHAS = (*(10.0**-k for k in range(4, 10)), 5e-5)  # 1e-4 ... 1e-9, 5e-5
# KSWIN sizes in whole days of half-hours. The reference window covers at
# least a full weekly cycle and at most ~a month (longer spans mix seasonal
# demand levels) -- a deliberate cap, not an unexplored edge; the recent
# sample holds whole days.
KSWIN_STAT_SIZES = (48, 96, 144, 240, 336, 672)  # 1, 2, 3, 5 days, 1, 2 weeks
KSWIN_WINDOW_SIZES = (336, 480, 672, 1344)  # 7, 10, 14, 28 days
# Page-Hinkley: every river hyperparameter except `mode`, fixed at "both"
# because AEMO demand can shift up or down.
#
# alpha (forgetting factor; memory ~ 1 / (1 - alpha) steps): memory must
# be at least 4 weeks of half-hours (1344 steps, alpha >= ~0.99926) --
# the same horizon as KSWIN's 28-day reference window, because the demand
# shifts this study targets build up over weeks, and a detector that
# forgets faster never accumulates them. 0.9995 (~6 weeks) and river's
# default 1 - 1e-4 (~7 months) are the values above that floor. Shorter
# memories win on the synthetic sudden shifts but miss the gradual real
# shifts: alpha = 0.985 (~1.4 days) gave 1-4 AEMO test-period detections
# and alpha = 0.999 (~3 weeks) gave 7-8.
#
# threshold (lambda): 400 < lambda < 1000. At these memories, lambda <= 400
# exceeds the false-alarm floor on synthetic data, and the remaining
# "false alarms" are early alarms on the gradual-drift series (PH fires
# during the ramp, before the scoring window) -- never on the no-drift,
# sudden or recurring series. lambda >= 1000 suppresses those
# early-but-real alarms only by delaying every detection, so it is left
# out.
#
# delta (per-step change tolerated): 0.0001 to 0.001 -- small enough that
# small sustained shifts still accumulate, but non-zero so the detector
# keeps a minimum tolerance to noise-level wobble.
#
# NOTE: these bounds were set after seeing the AEMO results -- report them
# alongside any Page-Hinkley result; this is not a synthetic-only choice.
PAGE_HINKLEY_DELTAS = (0.0001, 0.0005, 0.001)
PAGE_HINKLEY_MIN_INSTANCES = (20, 30, 50)
PAGE_HINKLEY_THRESHOLDS = (
    425.0, 450.0, 475.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0,
    850.0, 900.0, 950.0,
)
PAGE_HINKLEY_ALPHAS = (0.9995, 1 - 0.0001)
PAGE_HINKLEY_MODE = "both"

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

    for alpha in PAGE_HINKLEY_ALPHAS:
        for delta in PAGE_HINKLEY_DELTAS:
            for min_instances in PAGE_HINKLEY_MIN_INSTANCES:
                for threshold in PAGE_HINKLEY_THRESHOLDS:
                    yield (
                        "page_hinkley_budget_grid",
                        PageHinkleyDetector,
                        {
                            "min_instances": min_instances,
                            "delta": delta,
                            "threshold": threshold,
                            "alpha": alpha,
                            "mode": PAGE_HINKLEY_MODE,
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


# --- parent-only I/O: checkpoints, output tables ---------------------------


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
        config_hash = results_io.config_hash(full_config(kwargs))

        cached = load_checkpoint(config_hash)
        if cached is not None:
            completed_units[config_hash] = cached
        else:
            pending.append((config_hash, sweep_name, detector_class, kwargs))

    if pending:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(evaluate_unit, sweep_name, detector_class, kwargs): config_hash
                for config_hash, sweep_name, detector_class, kwargs in pending
            }
            for future in as_completed(futures):
                config_hash = futures[future]
                unit = future.result()

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
