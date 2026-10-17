"""One command for the whole pipeline up to adaptation.

    Data -> Detect -> Adapt            (this script)
                      -> Quantify (+ Escalate)   (plugs in later, see UNCERTAINTY)

Nothing new is computed here. Each step is an existing `run_*` or
`produce_*` script, run in dependency order as `python -m <module>` in its
own process, so every script behaves exactly as it does when run by hand
(logging to results/runs.csv through record_run, writing its own tables and
figures). This file only decides the order, passes the arguments, stops at
the first failure and prints what it did.

Stages, in order:

    forecast     frozen baselines on AEMO (F1 curves)
    synthetic    synthetic benchmark: generator plots, detector tuning, frozen run
    detect       detectors on AEMO (raw / daily / half-hourly) + ADWIN change
                 points that trigger arms C and D
    adapt        the four adaptation arms, per model x region x seed
    produce      tables T1-T4 and figures F1-F3 from runs.csv

Every run_* step appends its rows to results/runs.csv (the shared, tracked
ledger), so re-running a stage adds a newer batch; the produce_* scripts read
the latest batch. Use `--stages produce` to rebuild tables and figures
without touching the ledger.

Heavy stages are skipped unless asked for, because they take hours:
`synthetic` (the tuning sweep) and `adapt` (NHITS / XGBoost x 2 regions x 5
seeds). `produce` only reads what is already in results/runs.csv, so a
default run is quick and safe to repeat.

Uncertainty and escalation are not part of this script yet. When they are
ready, add their steps to UNCERTAINTY below and they run after `adapt`
(calibration forecasts and re-scoring already exist for the conformal
intervals: see UNCERTAINTY_READY_STEPS). `--stages uncertainty` runs those.

Usage (from the repo root):

    python -m experiments.run_pipeline --list              # show every step
    python -m experiments.run_pipeline --dry-run           # print the plan only
    python -m experiments.run_pipeline                     # forecast + detect + produce
    python -m experiments.run_pipeline --stages produce    # rebuild tables/figures only
    python -m experiments.run_pipeline --stages adapt --models xgboost --regions SA1 --seeds 1
    python -m experiments.run_pipeline --stages adapt --models nhits --arms D_drift_recent_30d,D_drift_recent_60d,D_drift_recent_120d
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


def adaptation_steps(models, regions, seeds, arms=None) -> list[Step]:
    """One run_aemo_adaptation_arms call per model x region (x seed for the
    stochastic models). dhr_arima takes no seed. `arms` is the runner's own
    comma list (e.g. "D_drift_recent_30d,D_drift_recent_60d"); None = all arms."""
    steps = []
    for model in models:
        for region in regions:
            for seed in seeds if model in SEEDED_MODELS else (None,):
                args = ["--model", model, "--region", region]
                if seed is not None:
                    args += ["--seed", str(seed)]
                if arms:
                    args += ["--arms", arms]
                steps.append(
                    Step("adapt", f"{RUN}.adaptation.run_aemo_adaptation_arms", tuple(args))
                )
    return steps


def uncertainty_ready_steps(models) -> list[Step]:
    """What already exists for the conformal intervals (T3 pinball loss)."""
    steps = [
        Step("uncertainty", f"{RUN}.forecasting.run_aemo_calibration_forecasts", ("--model", m))
        for m in models
    ]
    steps.append(Step("uncertainty", f"{RUN}.adaptation.rescore_adaptation_runs"))
    return steps


# Add the uncertainty / escalation steps here when they are done, as
# Step("uncertainty", "<module>", ("--arg", "value")). They run after `adapt`.
UNCERTAINTY: list[Step] = []


def build_stages(models, regions, seeds, arms=None) -> list[Stage]:
    return [
        Stage(
            "forecast",
            "Frozen baselines on AEMO",
            steps=[Step("forecast", f"{RUN}.forecasting.run_aemo_baselines")],
        ),
        Stage(
            "synthetic",
            "Synthetic benchmark: tune detectors, then run the frozen configs",
            heavy=True,
            steps=[
                Step("synthetic", f"{RUN}.detection.run_synthetic_generator"),
                Step("synthetic", f"{RUN}.detection.run_fine_tune_on_synthetic"),
                Step("synthetic", f"{RUN}.detection.run_post_tune_on_synthetic"),
            ],
        ),
        Stage(
            "detect",
            "Detectors on AEMO, then the ADWIN change points that trigger arms C and D",
            steps=[
                Step("detect", f"{RUN}.detection.run_aemo_detectors_raw_post_tune"),
                Step("detect", f"{RUN}.detection.run_aemo_detectors_standard_daily_post_tune"),
                Step(
                    "detect", f"{RUN}.detection.run_aemo_detectors_standard_half_hourly_post_tune"
                ),
                Step("detect", f"{RUN}.detection.run_adwin_changepoints"),
            ],
        ),
        Stage(
            "adapt",
            "The four adaptation arms (A never, B every 30 days, C full history, D recent window)",
            heavy=True,
            steps=adaptation_steps(models, regions, seeds, arms),
        ),
        Stage(
            "uncertainty",
            "Calibration forecasts + conformal re-scoring (pinball loss), then UQ / escalation",
            heavy=True,
            default=False,
            steps=uncertainty_ready_steps(models) + UNCERTAINTY,
        ),
        Stage(
            "produce",
            "Tables T1-T4 and figures F1-F3 from results/runs.csv",
            steps=[
                Step("produce", f"{PRODUCE}.produce_table_t1_post_tune"),
                Step("produce", f"{PRODUCE}.produce_figure_f1"),
                Step("produce", f"{PRODUCE}.produce_figure_f2_synthetic_post_tune"),
                Step("produce", f"{PRODUCE}.produce_figure_f2_aemo_standard_half_hourly_post_tune"),
                Step("produce", f"{PRODUCE}.produce_table_t2_all_streams_post_tune"),
                Step("produce", f"{PRODUCE}.produce_table_t3"),
                Step("produce", f"{PRODUCE}.produce_table_t4"),
                Step("produce", f"{PRODUCE}.produce_figure_f3"),
            ],
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
            "    hint: figures and T1 read the per-run dumps in results/runs/<config_hash>/ "
            "(gitignored). They exist only on a machine that ran the matching run_* steps. "
            "T2-T4 need only results/runs.csv.",
            flush=True,
        )
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
        description="Run the pipeline up to adaptation (Data -> Detect -> Adapt)."
    )
    p.add_argument(
        "--stages",
        nargs="+",
        help="stages to run, or 'all' (default: forecast detect produce)",
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
        "--arms",
        help="adapt stage only: comma list passed to the runner's --arms "
        "(default: all arms), e.g. D_drift_recent_30d,D_drift_recent_60d,D_drift_recent_120d",
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
    stages = build_stages(args.models, args.regions, args.seeds, args.arms)

    if args.list:
        for s in stages:
            flags = ", ".join(
                f
                for f, on in (("heavy", s.heavy), ("off by default", not s.default))
                if on
            )
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
    if not UNCERTAINTY:
        print("  NOTE  uncertainty / escalation steps are not wired in yet (UNCERTAINTY is empty)")

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
