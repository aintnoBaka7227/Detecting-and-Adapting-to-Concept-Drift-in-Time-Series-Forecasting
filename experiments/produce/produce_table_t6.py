"""T6 -- escalation operating points: how much human review is needed.

Columns (required-outputs T6): model, arm, region, UQ method, regime,
review rate for 0.90, review rate for 0.95, coverage @ 5%, @ 10%, @ 20%,
seeds. Read from the escalation rows logged by run_aemo_escalation_sweep.py
(rule: escalate when the 90% interval is wider than tau, tau swept over
every issued width; evaluation.calculate_escalation_operating_points).

Review rate for a target is the smallest review rate whose retained
coverage reaches it. Across seeds:

    every seed reaches it   "4%" when the seeds agree, else "4% ± 2%"
    some seeds do           "0% in 1/5 seeds, else unreachable" (the rate
                            is the mean over the seeds that reach it)
    no seed does            "unreachable"

Coverage @ b is the retained coverage at review rate b, "0.927" when the
seeds agree to three decimals, else "0.927 ± 0.002". A run "attains" b when
it has an operating point within ATTAIN_TOLERANCE of b; otherwise the cell
is "not attainable". Fixed split conformal has a constant width, so its
only operating points are 0% and 100% review: it never attains 5/10/20%,
and its review rate for a target is 0% (already met) or unreachable.

A single run (dhr_arima is deterministic) is shown without a spread.

Regimes are the global rule, as in T5. Arm D is reported at the 180-day
window only, as in T4/T5.

Writes one CSV with every row and one table image per model.

Every output is written once per trigger detector (experiments.trigger_detectors:
ADWIN, Page-Hinkley) with the detector in the file name; arms A and B are the
shared controls, arms C and D are that detector's runs.
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.produce_figure_f5 import latest_escalation_rows
from experiments.produce.table_adaptation_common import REGIMES, arm_sort_key
from experiments.produce.table_image import save_table_image
from experiments.results_io import TABLE6_DIR
from experiments.run.uncertainty.run_aemo_escalation_sweep import MIN_RETAINED
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


def build_table(detector: str) -> pd.DataFrame:
    rows = latest_escalation_rows(detector)
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
    for detector in TRIGGER_DETECTORS:
        write_tables(detector)


def write_tables(detector: str) -> None:
    table = build_table(detector)
    out = TABLE6_DIR / f"table_t6_escalation_operating_points_{detector}.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")
    for model, rows in table.groupby("model"):
        image = save_table_image(
            rows.reset_index(drop=True),
            TABLE6_DIR / f"table_t6_escalation_operating_points_{detector}_{model}.png",
            title=f"T6 — Escalation operating points, {model}, {DETECTOR_LABELS[detector]} trigger",
            subtitle=(
                "Escalate when the 90% interval is wider than τ (τ swept over every issued width); "
                "coverage of the forecasts kept, per global regime. Review rate for a target: smallest "
                f"review rate reaching it with ≥{MIN_RETAINED} forecast kept; \"unreachable\" when no "
                "review rate does. Coverage @ b: coverage of the forecasts kept when b of them are "
                "reviewed; \"not attainable\" when no threshold reviews that share (fixed split "
                "conformal gives every interval the same width, so it can review only 0% or 100%). "
                "Values are the mean across seeds, ± sd when the seeds differ; \"in k/n seeds\" when "
                "only some seeds reach or attain it. dhr_arima is deterministic: 1 run."
            ),
        )
        print(f"wrote {image}")


if __name__ == "__main__":
    main()
