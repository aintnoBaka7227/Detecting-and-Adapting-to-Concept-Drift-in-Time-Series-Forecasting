"""Final synthetic-only detector tuning against the false-alarm budget.

Acceptance criteria:
1. <= 2 false alarms per simulated year.
2. No missed known drift events.
3. Zero false alarms on the "none" (no-drift) stream. Unlike sudden/
   gradual/recurring, there is no tolerance window to argue about here --
   any detection on a pure no-drift series is unambiguously wrong, so it
   is a hard gate rather than folded into the annualised rate budget
   (a config could satisfy the <=2/year budget while still firing on
   every single no-drift run tested).

Grid search per detector, not one-parameter-at-a-time: ADWIN has only one
tunable parameter (`delta`) so its sweep already covers its full space;
KSWIN (`alpha`, `window_size`, `stat_size`) and Page-Hinkley
(`min_instances`, `delta`, `threshold`) each get a real Cartesian grid so
parameter interactions aren't missed by holding two of three knobs fixed.

The winning config per detector is selected automatically (lowest mean
detection delay among configs that pass all three criteria on every
drift type and seed) and written to
results/tables/fine_tune_on_synthetic_winners.csv, instead of a human
eyeballing the full sweep table and hand-copying literals into
frozen_detector_configs.py.

AEMO data and AEMO events are not used for tuning.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from drift_lab.config import SEEDS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.evaluation.evaluation import evaluate_detections
from drift_lab.synthetic.generator import make_series
from experiments.run_harness import config_of

KINDS = ("none", "sudden", "gradual", "recurring")

N = 20_000
NOISE = 1.0
SPLIT_ID = "synth_n20000_tuning"

SAMPLES_PER_YEAR = 48 * 365
FALSE_ALARM_BUDGET_PER_YEAR = 2.0

REPO_ROOT = Path(__file__).resolve().parents[3]
TABLES_DIR = REPO_ROOT / "results" / "tables"
OUTPUT = TABLES_DIR / "fine_tune_on_synthetic.csv"
WINNERS_OUTPUT = TABLES_DIR / "fine_tune_on_synthetic_winners.csv"


def false_alarms_per_year(
    n_false_alarms: int,
    n_observations: int,
) -> float:
    simulated_years = n_observations / SAMPLES_PER_YEAR
    return float(n_false_alarms / simulated_years)


def candidate_sweeps():
    """Yield every candidate config to evaluate, one detector at a time."""

    # ADWIN's only knob. Widened past the earlier 0.00025-0.001 range in
    # both directions: down to 0.0001 (more sensitive, in case it still
    # clears the budget with lower delay) and up to the class default
    # 0.002 (kept as a reference point).
    for delta in (0.0001, 0.00025, 0.0005, 0.00075, 0.001, 0.0015, 0.002):
        yield (
            "adwin_budget_delta",
            ADWINDetector(delta=delta),
        )

    # KSWIN: a real (alpha, window_size, stat_size) grid instead of only
    # stat_size. alpha widened well past the earlier 0.005-only value in
    # both directions (more sensitive to more conservative); stat_size
    # extended past 50 since the false-alarm count on "none" was still
    # falling, not flat, at the old sweep's upper end.
    for alpha in (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05):
        for window_size in (300, 450):
            for stat_size in (40, 44, 48, 50, 55, 60, 65):
                yield (
                    "kswin_budget_grid",
                    KSWINDetector(
                        alpha=alpha,
                        window_size=window_size,
                        stat_size=stat_size,
                        seed=42,
                    ),
                )

    # Page-Hinkley: a real (min_instances, delta, threshold) grid instead
    # of only threshold. delta widened the same way as KSWIN's alpha.
    for delta in (0.0005, 0.001, 0.005, 0.01, 0.02, 0.05):
        for min_instances in (20, 30, 50):
            for threshold in (200.0, 300.0, 400.0, 500.0, 750.0, 1000.0):
                yield (
                    "page_hinkley_budget_grid",
                    PageHinkleyDetector(
                        min_instances=min_instances,
                        delta=delta,
                        threshold=threshold,
                    ),
                )


def evaluate_candidate(detector, sweep_name: str) -> list[dict]:
    """Run one candidate config across every (kind, seed), unscored --
    `accepted` is filled in afterwards once every row for this config is
    in, since the "none must be clean" gate needs all of a config's
    "none"-kind rows before it can be applied to any of that config's
    rows."""

    detector_config = config_of(detector)

    print(f"\n=== {detector.name} | {sweep_name} | {detector_config} ===")

    rows = []

    for kind in KINDS:
        for seed in SEEDS:
            series, truth = make_series(kind=kind, n=N, noise=NOISE, seed=seed)

            detected = detector.detect(series)

            evaluation = evaluate_detections(
                detected_changepoints=detected,
                true_changepoints=truth,
                n_observations=len(series),
                drift_type=kind,
            )

            n_false = len(evaluation["false_alarm_indices"])
            fa_year = false_alarms_per_year(n_false, len(series))

            rows.append(
                {
                    "method": detector.name,
                    "sweep": sweep_name,
                    "config": str(detector_config),
                    "drift_type": kind,
                    "seed": seed,
                    "n_detections": len(detected),
                    "n_false_alarms": n_false,
                    "false_alarms_per_year": fa_year,
                    "budget_met": fa_year <= FALSE_ALARM_BUDGET_PER_YEAR,
                    "detection_delay": evaluation["detection_delay"],
                    "missed_detections": evaluation["missed_detections"],
                }
            )

            print(
                f"  {kind:9s} seed={seed} det={len(detected):3d} "
                f"FA/year={fa_year:6.2f} miss={evaluation['missed_detections']}"
            )

    return rows


def apply_acceptance(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the `none_clean` and final `accepted` columns.

    `none_clean` is a per-(method, config) property -- true only if every
    seed's "none"-kind run for that config had zero false alarms -- so it
    has to be computed after every row for a config exists, then broadcast
    back onto all of that config's rows (not just its "none" ones), since
    acceptance is a property of the config as a whole.
    """
    none_clean = (
        frame[frame["drift_type"] == "none"]
        .groupby(["method", "config"])["n_false_alarms"]
        .apply(lambda alarms: bool((alarms == 0).all()))
        .rename("none_clean")
    )
    frame = frame.join(none_clean, on=["method", "config"])
    frame["accepted"] = (
        frame["budget_met"] & (frame["missed_detections"] == 0) & frame["none_clean"]
    )
    return frame


def select_winners(frame: pd.DataFrame) -> pd.DataFrame:
    """Pick one config per detector: lowest mean detection delay among
    configs accepted on every (kind, seed) row. Falls back to the config
    with the fewest total false alarms on "none" (then lowest worst-case
    false-alarms/year) if nothing fully clears every criterion, and flags
    that fallback clearly rather than silently picking a partial pass."""

    none_totals = (
        frame[frame["drift_type"] == "none"]
        .groupby(["method", "config"])["n_false_alarms"]
        .sum()
        .rename("none_total_false_alarms")
    )

    per_config = (
        frame.groupby(["method", "sweep", "config"], dropna=False)
        .agg(
            all_runs_accepted=("accepted", "all"),
            worst_false_alarms_per_year=("false_alarms_per_year", "max"),
            total_missed_detections=("missed_detections", "sum"),
            mean_detection_delay=("detection_delay", "mean"),
        )
        .reset_index()
        .merge(none_totals, on=["method", "config"], how="left")
    )

    winners = []
    for method, group in per_config.groupby("method"):
        fully_accepted = group[group["all_runs_accepted"]]

        if not fully_accepted.empty:
            winner = fully_accepted.loc[fully_accepted["mean_detection_delay"].idxmin()].copy()
            winner["fallback"] = False
        else:
            print(
                f"\nWARNING: no {method} config passed every criterion "
                "(budget + no misses + none_clean) on every (kind, seed) run. "
                "Falling back to the cleanest partial candidate -- review "
                f"{OUTPUT} before trusting this pick."
            )
            ranked = group.sort_values(
                ["none_total_false_alarms", "worst_false_alarms_per_year"]
            )
            winner = ranked.iloc[0].copy()
            winner["fallback"] = True

        winners.append(winner)

    return pd.DataFrame(winners).reset_index(drop=True)


def main() -> None:
    rows = []
    for sweep_name, detector in candidate_sweeps():
        rows.extend(evaluate_candidate(detector, sweep_name))

    frame = apply_acceptance(pd.DataFrame(rows))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT, index=False)
    print(f"\nSaved: {OUTPUT}")

    print("\n=== CONFIGURATION ACCEPTANCE SUMMARY ===")
    summary = (
        frame.groupby(["method", "sweep", "config"], dropna=False)
        .agg(
            worst_false_alarms_per_year=("false_alarms_per_year", "max"),
            total_missed_detections=("missed_detections", "sum"),
            none_clean=("none_clean", "first"),
            all_runs_accepted=("accepted", "all"),
            mean_detection_delay=("detection_delay", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))

    winners = select_winners(frame)
    winners.to_csv(WINNERS_OUTPUT, index=False)

    print("\n=== SELECTED FROZEN CONFIGS ===")
    print(
        winners[
            [
                "method",
                "config",
                "mean_detection_delay",
                "worst_false_alarms_per_year",
                "fallback",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved: {WINNERS_OUTPUT}")


if __name__ == "__main__":
    main()
