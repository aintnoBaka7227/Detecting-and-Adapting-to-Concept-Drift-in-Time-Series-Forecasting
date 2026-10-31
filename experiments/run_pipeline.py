"""One command for the whole pipeline, from data checks to escalation.

    Data -> Detect -> Forecast -> Adapt -> Uncertainty -> Escalate -> Produce

Nothing new is computed here. Each step is an existing `run_*`, `check_*` or
`produce_*` script, run in dependency order as `python -m <module>` in its
own process, so every script behaves exactly as it does when run by hand
(logging to results/runs.csv through record_run, writing its own tables and
figures). This file only decides the order, passes the arguments, stops at
the first failure and prints what it did. The order is the one in the
README's "Execution order" table.

Stages, in order:

    data         QC of the seasonal residual stream the detectors read
    synthetic    synthetic benchmark: generator plots, frozen detector run
                 (the multi-hour tuning sweep only with --retune)
    detect       detectors on AEMO, the change points that trigger arms C and
                 D, and the periodicity check
    forecast     frozen baselines (F1) and the Calibration-window forecasts
                 that forecaster selection and the conformal intervals use
    adapt        the adaptation arms, per detector x model x region x seed,
                 then the re-scoring that adds pinball loss      (heavy)
    uncertainty  90% intervals (fixed split conformal, AgACI)    (heavy, needs R)
    escalation   escalation threshold sweep                      (heavy)
    produce      every table and figure: T1-T7, F1-F6, the Arm D sweep and
                 the forecaster-selection table

Every run_* step appends its rows to results/runs.csv (the shared, tracked
ledger), so re-running a stage adds a newer batch; the produce_* scripts read
the latest batch. Use `--stages produce` to rebuild tables and figures
without touching the ledger.

The heavy stages are skipped unless asked for, because they take hours
(NHITS / XGBoost x 2 regions x 5 seeds x 2 detectors). Name a heavy stage, or
pass --include-heavy.

Usage (from the repo root):

    python -m experiments.run_pipeline --list              # show every step
    python -m experiments.run_pipeline --dry-run           # print the plan only
    python -m experiments.run_pipeline                     # data, synthetic, detect, forecast, produce
    python -m experiments.run_pipeline --stages produce    # rebuild tables/figures only
    python -m experiments.run_pipeline --stages adapt --models xgboost --regions SA1 --seeds 1
    python -m experiments.run_pipeline --stages adapt --detectors page_hinkley --models dhr_arima
    python -m experiments.run_pipeline --stages all --include-heavy   # everything, hours
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from drift_lab.config import REGIONS, SEEDS
from experiments.trigger_detectors import TRIGGER_DETECTORS

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "results" / "runs" / "pipeline_last_run.json"  # gitignored

MODELS = ("nhits", "xgboost", "dhr_arima")
SEEDED_MODELS = ("nhits", "xgboost")  # dhr_arima is deterministic: no --seed
RUN = "experiments.run"
PRODUCE = "experiments.produce"


@dataclass(frozen=True)
class Step:
    stage: str
    module: str
    args: tuple[str, ...] = ()
    note: str = ""

    @property
    def command(self) -> list[str]:
        return [sys.executable, "-m", self.module, *self.args]

    @property
    def label(self) -> str:
        return " ".join([self.module.removeprefix("experiments."), *self.args])


@dataclass
class Stage:
    name: str
    description: str
    heavy: bool = False
    default: bool = True
    steps: list[Step] = field(default_factory=list)


def _seeds_for(model, seeds):
    """Stochastic models run once per seed, dhr_arima once (it takes no --seed)."""
    return seeds if model in SEEDED_MODELS else (None,)


def adaptation_steps(models, regions, seeds, arms=None, detectors=None) -> list[Step]:
    """One run_aemo_adaptation_arms call per [detector x] model x region (x seed
    for the stochastic models). `arms` is the runner's own comma list (e.g.
    "D_drift_recent_30d,D_drift_recent_60d"); None = the runner's default
    (all four arms for ADWIN, only C and D for the other detectors, because A
    and B use no detector). `detectors` None leaves --detector to the runner's
    default (ADWIN)."""
    steps = []
    for detector in detectors or (None,):
        for model in models:
            for region in regions:
                for seed in _seeds_for(model, seeds):
                    args = []
                    if detector is not None:
                        args += ["--detector", detector]
                    args += ["--model", model, "--region", region]
                    if seed is not None:
                        args += ["--seed", str(seed)]
                    if arms:
                        args += ["--arms", arms]
                    steps.append(
                        Step("adapt", f"{RUN}.adaptation.run_aemo_adaptation_arms", tuple(args))
                    )
    return steps


def calibration_steps(models, regions, seeds) -> list[Step]:
    """Calibration-window forecasts, one call per model x region (x seed)."""
    steps = []
    for model in models:
        for region in regions:
            for seed in _seeds_for(model, seeds):
                args = ["--model", model, "--region", region]
                if seed is not None:
                    args += ["--seed", str(seed)]
                steps.append(
                    Step("forecast", f"{RUN}.forecasting.run_aemo_calibration_forecasts", tuple(args))
                )
    return steps


def per_detector_steps(stage, module, detectors, models, regions) -> list[Step]:
    """A uncertainty / escalation call per detector x model x region."""
    return [
        Step(stage, module, ("--detector", d, "--model", m, "--region", r))
        for d in detectors
        for m in models
        for r in regions
    ]


PRODUCERS = (
    "detection.produce_table_t1_post_tune",
    "detection.produce_figure_f2_synthetic_post_tune",
    "detection.produce_table_t2_standard_half_hourly_post_tune",
    "detection.produce_figure_f2_aemo_standard_half_hourly_post_tune",
    "forecasting.produce_figure_f1",
    "forecasting.produce_table_forecaster_selection",
    "adaptation.produce_table_t3",
    "adaptation.produce_table_t4",
    "adaptation.produce_figure_f3",
    "adaptation.produce_table_arm_d_sweep",
    "uncertainty.produce_table_t5",
    "uncertainty.produce_figure_f4",
    "escalation.produce_table_t6",
    "escalation.produce_figure_f5",
    "comparison.produce_table_t7",
    "comparison.produce_figure_f6",
)


def build_stages(
    models,
    regions,
    seeds,
    arms=None,
    retune=False,
    detectors=TRIGGER_DETECTORS,
) -> list[Stage]:
    detect = f"{RUN}.detection"
    return [
        Stage(
            "data",
            "QC of the seasonal residual stream: does the TRAIN-fitted profile generalise?",
            steps=[
                Step("data", f"{detect}.check_seasonal_residuals"),
                Step("data", f"{detect}.check_residual_generalisation"),
            ],
        ),
        Stage(
            "synthetic",
            "Synthetic benchmark: run the frozen detector configs "
            "(add --retune to redo the tuning sweep first)",
            steps=[
                Step("synthetic", f"{detect}.run_synthetic_generator"),
                *(
                    [Step("synthetic", f"{detect}.run_fine_tune_on_synthetic",
                          note="multi-hour sweep; the winners are already committed")]
                    if retune
                    else []
                ),
                Step("synthetic", f"{detect}.run_post_tune_on_synthetic"),
            ],
        ),
        Stage(
            "detect",
            "Detectors on AEMO, the change points that trigger arms C and D, periodicity check",
            steps=[
                Step("detect", f"{detect}.run_aemo_detectors_standard_half_hourly_post_tune"),
                Step("detect", f"{detect}.run_detector_changepoints"),
                Step("detect", f"{detect}.check_detection_periodicity"),
            ],
        ),
        Stage(
            "forecast",
            "Frozen baselines (F1), then the Calibration-window forecasts "
            "(forecaster selection, conformal residuals)",
            steps=[
                Step("forecast", f"{RUN}.forecasting.run_aemo_baselines"),
                Step("forecast", f"{RUN}.forecasting.run_aemo_nhits"),
                *calibration_steps(models, regions, seeds),
            ],
        ),
        Stage(
            "adapt",
            "The adaptation arms for each trigger detector (A never, B every 30 days, "
            "C full history, D recent window), then the re-scoring",
            heavy=True,
            steps=[
                *adaptation_steps(models, regions, seeds, arms, detectors),
                Step("adapt", f"{RUN}.adaptation.rescore_adaptation_runs"),
            ],
        ),
        Stage(
            "uncertainty",
            "90% intervals on every adaptation run: fixed split conformal and AgACI (needs R + opera)",
            heavy=True,
            steps=per_detector_steps(
                "uncertainty", f"{RUN}.uncertainty.run_aemo_uq_replay", detectors, models, regions
            ),
        ),
        Stage(
            "escalation",
            "Escalation threshold sweep over the saved intervals",
            heavy=True,
            steps=per_detector_steps(
                "escalation", f"{RUN}.uncertainty.run_aemo_escalation_sweep", detectors, models, regions
            ),
        ),
        Stage(
            "produce",
            "Tables T1-T7 and figures F1-F6 from results/runs.csv",
            steps=[Step("produce", f"{PRODUCE}.{name}") for name in PRODUCERS],
        ),
    ]


def select_stages(stages, requested, include_heavy):
    """Returns (to_run, skipped). `requested` is a list of names, or ["all"]
    for every stage, or None for the defaults."""
    by_name = {s.name: s for s in stages}
    if requested is None:
        chosen = [s for s in stages if s.default]
        explicit = False
    elif requested == ["all"]:
        chosen = list(stages)
        explicit = False
    else:
        unknown = [n for n in requested if n not in by_name]
        if unknown:
            raise SystemExit(f"unknown stage(s) {unknown}; choose from {list(by_name)} or 'all'")
        chosen = [by_name[n] for n in requested]
        explicit = True
    # Naming a heavy stage explicitly counts as asking for it.
    to_run = [s for s in chosen if include_heavy or explicit or not s.heavy]
    skipped = [s for s in chosen if s not in to_run]
    # Always run in pipeline order, whatever order the user typed.
    order = {s.name: i for i, s in enumerate(stages)}
    to_run.sort(key=lambda s: order[s.name])
    return to_run, skipped


def run_step(step: Step) -> dict:
    print(f"\n>>> {step.label}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run(step.command, cwd=REPO_ROOT)
    seconds = round(time.perf_counter() - t0, 1)
    status = "ok" if result.returncode == 0 else f"failed (exit {result.returncode})"
    print(f"<<< {step.label}: {status} in {seconds}s", flush=True)
    if result.returncode != 0 and step.stage == "produce":
        print(
            "    hint: this producer reads the per-run dumps in results/runs/<config_hash>/ "
            "(gitignored: curves, detections, intervals). They exist only on a machine that ran "
            "the matching run_* steps. Only T2, T5, T6, T7 and the forecaster-selection table "
            "rebuild from results/runs.csv alone.",
            flush=True,
        )
    if result.returncode != 0 and step.stage == "uncertainty":
        print("    hint: AgACI needs R and the opera package (see README, Setup).", flush=True)
    return {"step": step.label, "stage": step.stage, "status": status, "seconds": seconds}


def write_log(records: list[dict]) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {"finished_at": datetime.now(timezone.utc).isoformat(), "steps": records}
        LOG_PATH.write_text(json.dumps(payload, indent=2))
    except OSError as exc:  # the log is a convenience, never a reason to fail
        print(f"(could not write {LOG_PATH}: {exc})")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the whole pipeline (Data -> Detect -> Forecast -> Adapt -> "
        "Uncertainty -> Escalate -> Produce)."
    )
    p.add_argument(
        "--stages",
        nargs="+",
        help="stages to run, or 'all' (default: data synthetic detect forecast produce)",
    )
    p.add_argument(
        "--include-heavy",
        action="store_true",
        help="also run the multi-hour stages when running the defaults or 'all'",
    )
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--regions", nargs="+", choices=REGIONS, default=list(REGIONS))
    p.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    p.add_argument(
        "--detectors",
        nargs="+",
        choices=TRIGGER_DETECTORS,
        default=list(TRIGGER_DETECTORS),
        help="trigger detectors for the adapt, uncertainty and escalation stages",
    )
    p.add_argument(
        "--arms",
        help="adapt stage only: comma list passed to the runner's --arms "
        "(default: the runner's own default), e.g. D_drift_recent_30d,D_drift_recent_60d",
    )
    p.add_argument(
        "--retune",
        action="store_true",
        help="synthetic stage: also redo the detector tuning sweep (hours; the tuned "
        "winners are already committed in results/tables/)",
    )
    p.add_argument("--dry-run", action="store_true", help="print the plan and run nothing")
    p.add_argument("--list", action="store_true", help="list every stage and step, then exit")
    p.add_argument(
        "--keep-going",
        action="store_true",
        help="continue after a failed step (default: stop at the first failure)",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    stages = build_stages(
        args.models, args.regions, args.seeds, args.arms, args.retune, tuple(args.detectors)
    )

    if args.list:
        for s in stages:
            flags = ", ".join(f for f, on in (("heavy", s.heavy), ("off by default", not s.default)) if on)
            print(f"\n[{s.name}] {s.description}" + (f"  ({flags})" if flags else ""))
            for step in s.steps:
                print(f"    {step.label}")
        return 0

    to_run, skipped = select_stages(stages, args.stages, args.include_heavy)

    print("Pipeline plan")
    for s in to_run:
        print(f"  RUN   {s.name:<12} {len(s.steps):>3} step(s)  {s.description}")
    for s in skipped:
        print(f"  SKIP  {s.name:<12} {len(s.steps):>3} step(s)  heavy: add --include-heavy or name it")
    if any(s.name != "produce" for s in to_run):
        print(
            "  NOTE  run_* steps append rows to results/runs.csv (the shared, tracked ledger); "
            "the produce_* scripts read the latest rows"
        )

    if args.dry_run:
        for s in to_run:
            for step in s.steps:
                print(f"    {' '.join(step.command[1:])}")
        return 0

    records: list[dict] = []
    failed = False
    for s in to_run:
        print(f"\n===== stage: {s.name} =====", flush=True)
        for step in s.steps:
            record = run_step(step)
            records.append(record)
            if record["status"] != "ok":
                failed = True
                if not args.keep_going:
                    write_log(records)
                    print(f"\nStopped at the first failure: {step.label}")
                    return 1
    write_log(records)

    print("\nSummary")
    for r in records:
        print(f"  {r['status']:<18} {r['seconds']:>8}s  {r['step']}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
