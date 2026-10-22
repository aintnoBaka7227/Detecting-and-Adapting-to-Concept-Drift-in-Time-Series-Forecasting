"""T6 -- escalation operating points: how much human review is needed.

For every logged run this reads the per-timestamp curve (actual, regime,
lower, upper -- written by rescore_adaptation_runs.py), sweeps an
interval-width threshold to escalate (send to human review) the widest
forecasts first, and reads three things off each regime's curve with the
shared `drift_lab.evaluation` functions:

    review rate for 0.90 / 0.95   smallest share of forecasts that must be
                                  reviewed for the rest to reach that coverage
    coverage @ 5% / 10% / 20%     coverage of the forecasts kept when at
                                  most that share is reviewed

A target no threshold reaches is "unreachable" -- a finding, not a missing
cell. Across seeds: mean ± sd of the reachable runs, with "(k/n)" when only
some seeds reach it. Nothing is refitted.

Split conformal gives every forecast the same interval width, so a width
threshold escalates all forecasts or none: the only review rates on its
curve are 0% and 100%, and coverage @ x% equals plain coverage.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.evaluation.evaluation import (
    ESCALATION_REVIEW_BUDGETS,
    ESCALATION_TARGET_COVERAGES,
    FORECAST_REGIMES,
    calculate_escalation_curves_by_regime,
    calculate_escalation_operating_points,
)
from experiments import results_io
from experiments.produce.table_adaptation_common import (
    arm_label,
    arm_sort_key,
    latest_adaptation_rows,
    mae_by_run,
    official_d_arms,
)
from experiments.produce.table_image import save_table_image

UQ_METHOD = "split conformal (90%)"
REGIME_ORDER = tuple(FORECAST_REGIMES)


def operating_points_by_run(latest: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per (model, arm, region, seed, regime) of operating points;
    also the number of runs skipped for lacking interval columns."""
    rows, skipped = [], 0
    keys = ["model", "arm", "region", "seed_key", "config_hash"]
    for (model, arm, region, seed_key, chash), _ in latest.groupby(keys):
        seed = None if seed_key == -1 else int(seed_key)
        path = results_io.curve_path(chash, "aemo", region, seed)
        curve = pd.read_csv(path) if path.exists() else None
        if curve is None or not {"lower", "upper"} <= set(curve.columns):
            skipped += 1
            continue
        labels = curve["regime"].astype(str).str.replace("-", "_", regex=False)
        keep = labels.isin(REGIME_ORDER).to_numpy()  # drop "unassigned"
        curves = calculate_escalation_curves_by_regime(
            curve["actual"][keep], curve["lower"][keep], curve["upper"][keep], labels[keep]
        )
        for regime, regime_curve in curves.groupby("regime"):
            points = calculate_escalation_operating_points(regime_curve.reset_index(drop=True))
            rows.append({"model": model, "arm": arm, "region": region,
                         "seed_key": seed_key, "regime": regime, **points})
    return pd.DataFrame(rows), skipped


def _percent(values: pd.Series, digits: int = 1) -> str:
    """Review rates as a percentage; NaN means unreachable for that run."""
    reached = values.dropna()
    if reached.empty:
        return "unreachable"
    text = f"{100 * reached.mean():.{digits}f}%"
    if len(reached) > 1:
        text += f" ± {100 * reached.std():.{digits}f}"
    if len(reached) < len(values):
        text += f" ({len(reached)}/{len(values)} seeds)"
    return text


def _coverage(values: pd.Series, digits: int = 3) -> str:
    valid = values.dropna()
    if valid.empty:
        return "n/a"
    if len(valid) == 1:
        return f"{valid.iloc[0]:.{digits}f} ± n/a"
    return f"{valid.mean():.{digits}f} ± {valid.std():.{digits}f}"


def build_table(latest: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    latest = latest_adaptation_rows() if latest is None else latest
    per_run, skipped = operating_points_by_run(latest)
    if per_run.empty:
        return per_run, skipped
    official = official_d_arms(mae_by_run(latest))
    rows = []
    for (model, arm, region, regime), g in per_run.groupby(["model", "arm", "region", "regime"]):
        row = {
            "model": model,
            "arm": arm_label(arm, model, region, official),
            "region": region,
            "UQ method": UQ_METHOD,
            "regime": regime.replace("_", "-"),
        }
        for target in ESCALATION_TARGET_COVERAGES:
            row[f"review rate for {target:.2f}"] = _percent(
                g[f"review_rate_for_coverage_{target:.2f}"]
            )
        for budget in ESCALATION_REVIEW_BUDGETS:
            row[f"coverage @ {budget:.0%}"] = _coverage(g[f"coverage_at_review_{budget:.2f}"])
        row["review rates on curve"] = int(g["n_operating_points"].max())
        row["seeds"] = int(g["seed_key"].nunique())
        row["_order"] = (arm_sort_key(arm), REGIME_ORDER.index(regime))
        rows.append(row)
    table = pd.DataFrame(rows)
    table = table.sort_values(["model", "region", "_order"]).drop(columns="_order")
    return table.reset_index(drop=True), skipped


def main() -> None:
    results_io.TABLE6_DIR.mkdir(parents=True, exist_ok=True)
    table, skipped = build_table()
    if table.empty:
        raise SystemExit(
            "no run has interval columns -- run rescore_adaptation_runs.py first"
        )
    out = results_io.TABLE6_DIR / "table_t6_escalation_operating_points.csv"
    table.to_csv(out, index=False)
    image = save_table_image(
        table,
        results_io.TABLE6_DIR / "table_t6_escalation_operating_points.png",
        title="T6 — Escalation operating points, split conformal",
        subtitle=(
            "Escalate widest intervals first; coverage is of the forecasts kept. "
            "'unreachable' = no threshold reaches the target; mean ± sd across seeds."
        ),
    )
    print(table.to_string(index=False))
    if skipped:
        print(f"\n{skipped} run(s) skipped: curve has no interval columns")
    print(f"\nwrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
