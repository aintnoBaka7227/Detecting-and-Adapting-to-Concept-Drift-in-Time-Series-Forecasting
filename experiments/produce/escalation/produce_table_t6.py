"""T6 -- escalation operating points: how much human review is needed.

One table per detector x escalation signal (width or a nonconformity
score), read from run_aemo_escalation_sweep.py's rows. Columns: review rate
for 0.90 / 0.95 (the smallest review rate whose kept forecasts reach it) and
coverage at 5 / 10 / 20% review. Across seeds:
    "4%" or "4% ± 2%"                    every seed reaches it (± if they differ)
    "0% in 1/5 seeds, else unreachable"  only some seeds do
    "unreachable"                        none does
A budget is "not attainable" when no threshold reviews within
ATTAIN_TOLERANCE of it (e.g. width for fixed split conformal: 0% or 100%
only). Global regimes; arm D at 180 days.

Outputs go to <table|figure>/<detector>/[<model>/[<signal>/]] (results_io.output_path);
arms A and B are the shared controls, C and D that detector's runs.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from experiments.produce.adaptation.table_adaptation_common import REGIMES, arm_sort_key
from experiments.produce.escalation.produce_figure_f5 import latest_escalation_rows
from experiments.produce.shared.table_image import save_table_image
from experiments.results_io import TABLE6_DIR, output_path
from experiments.run.uncertainty.run_aemo_escalation_sweep import (
    MIN_RETAINED,
    SIGNAL_RULES,
    SIGNALS,
)
from experiments.trigger_detectors import DETECTOR_LABELS, TRIGGER_DETECTORS

UQ_LABELS = {"fixed_split_conformal": "fixed split conformal", "agaci": "AgACI"}
TARGETS = (0.90, 0.95)
BUDGETS = (0.05, 0.10, 0.20)
RUN_KEYS = ["model", "arm", "region", "uq", "seed_key"]
# A budget is attained when a run has an operating point this close to it.
ATTAIN_TOLERANCE = 0.005


def percent(value: float) -> str:
    """0.0 -> "0%", 0.046 -> "4.6%"."""
    return f"{value * 100:.1f}".removesuffix(".0") + "%"


def spread(values: pd.Series, fmt) -> str:
    """Mean, with "± sd" only when the seeds differ at the shown precision."""
    mean = fmt(values.mean())
    if len(values) < 2 or fmt(values.std()) in (fmt(0.0), fmt(-0.0)):
        return mean
    return f"{mean} ± {fmt(values.std())}"


def review_rate_cell(values: pd.Series) -> str:
    """Review rate for a target across seeds (NaN = that seed never reaches it)."""
    reached = values.dropna()
    if reached.empty:
        return "unreachable"
    cell = spread(reached, percent)
    if len(reached) < len(values):
        cell += f" in {len(reached)}/{len(values)} seeds, else unreachable"
    return cell


def coverage_cell(coverage: pd.Series, attained: pd.Series, budget: float) -> str:
    """Coverage @ budget over the seeds that attain it; "not attainable" when
    none do."""
    ok = (attained - budget).abs() <= ATTAIN_TOLERANCE
    if not ok.any():
        return "not attainable"
    cell = spread(coverage[ok], lambda v: f"{v:.3f}")
    if not ok.all():
        cell += f" in {int(ok.sum())}/{len(ok)} seeds, else not attainable"
    return cell


def build_table(detector: str, signal: str) -> pd.DataFrame:
    rows = latest_escalation_rows(detector, signal)
    rows = rows[rows["regime"].isin(REGIMES)]
    # One row per run x regime; NaN review rates (unreachable) must survive.
    per_run = rows.set_index([*RUN_KEYS, "regime", "metric_name"])["metric_value"].unstack().reset_index()

    table_rows = []
    for (model, arm, region, uq, regime), g in per_run.groupby(["model", "arm", "region", "uq", "regime"]):
        row = {"model": model, "arm": arm, "region": region, "UQ method": UQ_LABELS.get(uq, uq), "regime": regime}
        for target in TARGETS:
            row[f"review rate for {target:.2f}"] = review_rate_cell(
                g[f"review_rate_for_coverage_{target:.2f}"]
            )
        for budget in BUDGETS:
            row[f"coverage @ {budget:.0%}"] = coverage_cell(
                g[f"coverage_at_review_{budget:.2f}"], g[f"actual_review_at_{budget:.2f}"], budget
            )
        row["seeds"] = int(g["seed_key"].nunique())
        row["_order"] = (*arm_sort_key(arm), list(UQ_LABELS).index(uq), REGIMES.index(regime))
        table_rows.append(row)
    table = pd.DataFrame(table_rows).sort_values(["model", "region", "_order"])
    return table.drop(columns="_order").reset_index(drop=True)


def main() -> None:
    TABLE6_DIR.mkdir(parents=True, exist_ok=True)
    for detector, signal in product(TRIGGER_DETECTORS, SIGNALS):
        write_tables(detector, signal)


def write_tables(detector: str, signal: str) -> None:
    table = build_table(detector, signal)
    stem = f"table_t6_escalation_operating_points_{detector}_{signal}"
    out = output_path(TABLE6_DIR, detector, name=f"{stem}.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")
    for model, rows in table.groupby("model"):
        image = save_table_image(
            rows.reset_index(drop=True),
            output_path(TABLE6_DIR, detector, model, signal, name=f"{stem}_{model}.png"),
            title=(
                f"T6 — Escalation operating points, {model}, {DETECTOR_LABELS[detector]} trigger, "
                f"{signal.replace('_', ' ')}"
            ),
            subtitle=(
                f"{SIGNAL_RULES[signal]} (τ swept over every observed value); "
                "coverage of the forecasts kept, per global regime. Review rate for a target: smallest "
                f"review rate reaching it with ≥{MIN_RETAINED} forecast kept; \"unreachable\" when no "
                "review rate does. Coverage @ b: coverage of the forecasts kept when b of them are "
                "reviewed; \"not attainable\" when no threshold reviews that share (e.g. width for "
                "fixed split conformal: every interval has the same width, so only 0% or 100%). "
                "Values are the mean across seeds, ± sd when the seeds differ; \"in k/n seeds\" when "
                "only some seeds reach or attain it. dhr_arima is deterministic: 1 run."
            ),
        )
        print(f"wrote {image}")


if __name__ == "__main__":
    main()
