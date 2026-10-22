"""T5 -- coverage and interval width by regime, for split conformal.

Reads the per-timestamp curve of the latest logged run of every (model,
arm, region, seed) -- actual, regime, lower, upper, written by
rescore_adaptation_runs.py -- and scores the 90% split conformal
interval with the shared `drift_lab.evaluation.calculate_uq_metrics_by_regime`.
Nothing is refitted. Runs whose curve has no interval columns are skipped
and counted.

Columns (required-outputs T5): UQ method, regime, coverage, mean width,
normalised width (mean width / mean TRAIN demand of the region), worst 24h
coverage, as mean ± sd across seeds. Worst 24h coverage is the lowest
48-step rolling coverage among windows lying wholly inside the regime on
consecutive half-hours; `complete 24h windows` is how many such windows
exist (they overlap, so it is not a count of independent days).
"""

from __future__ import annotations

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.evaluation import calculate_uq_metrics_by_regime, calculate_width_scale
from experiments import results_io
from experiments.produce.table_adaptation_common import (
    arm_label,
    arm_sort_key,
    latest_adaptation_rows,
    mae_by_run,
    mean_sd,
    official_d_arms,
)
from experiments.produce.table_image import save_table_image
from experiments.run.adaptation.run_aemo_adaptation_arms import TARGET_COLUMN

UQ_METHOD = "split conformal (90%)"
REGIME_ORDER = ("pre_drift", "drift", "post_drift")
WINDOW = 48  # 24 hours of half-hours


def _train_width_scale(region: str) -> float:
    train, _, _ = loader.load(region)
    return calculate_width_scale(train[TARGET_COLUMN])


def uq_by_run(latest: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One row per (model, arm, region, seed, regime); also the number of
    runs skipped for lacking interval columns."""
    scales: dict[str, float] = {}
    rows, skipped = [], 0
    keys = ["model", "arm", "region", "seed_key", "config_hash"]
    for (model, arm, region, seed_key, chash), _ in latest.groupby(keys):
        seed = None if seed_key == -1 else int(seed_key)
        path = results_io.curve_path(chash, "aemo", region, seed)
        curve = pd.read_csv(path, parse_dates=["timestamp"]) if path.exists() else None
        if curve is None or not {"lower", "upper"} <= set(curve.columns):
            skipped += 1
            continue
        if region not in scales:
            scales[region] = _train_width_scale(region)
        metrics = calculate_uq_metrics_by_regime(
            curve["actual"],
            curve["lower"],
            curve["upper"],
            curve["regime"],
            window=WINDOW,
            width_scale=scales[region],
            timestamps=curve["timestamp"],
        )
        for regime, m in metrics.items():
            rows.append({"model": model, "arm": arm, "region": region,
                         "seed_key": seed_key, "regime": regime, **m})
    return pd.DataFrame(rows), skipped


def build_table(latest: pd.DataFrame | None = None) -> tuple[pd.DataFrame, int]:
    latest = latest_adaptation_rows() if latest is None else latest
    per_run, skipped = uq_by_run(latest)
    if per_run.empty:
        return per_run, skipped
    official = official_d_arms(mae_by_run(latest))
    rows = []
    for (model, arm, region, regime), g in per_run.groupby(["model", "arm", "region", "regime"]):
        rows.append(
            {
                "model": model,
                "arm": arm_label(arm, model, region, official),
                "region": region,
                "UQ method": UQ_METHOD,
                "regime": regime.replace("_", "-"),
                "coverage": mean_sd(g["coverage"], 3),
                "mean width (MW)": mean_sd(g["mean_interval_width"], 1),
                "normalised width": mean_sd(g["normalised_interval_width"], 3),
                "worst 24h coverage": mean_sd(g["worst_rolling_coverage"], 3),
                "complete 24h windows": int(g["n_complete_windows"].iloc[0]),
                "seeds": int(g["seed_key"].nunique()),
                "_order": (arm_sort_key(arm), REGIME_ORDER.index(regime)),
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(["model", "region", "_order"]).drop(columns="_order")
    return table.reset_index(drop=True), skipped


def main() -> None:
    results_io.TABLE5_DIR.mkdir(parents=True, exist_ok=True)
    table, skipped = build_table()
    if table.empty:
        raise SystemExit(
            "no run has interval columns -- run run_aemo_calibration_forecasts.py "
            "then rescore_adaptation_runs.py first"
        )
    out = results_io.TABLE5_DIR / "table_t5_coverage_width.csv"
    table.to_csv(out, index=False)
    image = save_table_image(
        table,
        results_io.TABLE5_DIR / "table_t5_coverage_width.png",
        title="T5 — Coverage and interval width by regime, split conformal",
        subtitle=(
            "Nominal coverage 0.90; normalised width = mean width / mean TRAIN demand; "
            "worst 24h = lowest 48-step rolling coverage inside the regime; mean ± sd across seeds."
        ),
    )
    print(table.to_string(index=False))
    if skipped:
        print(f"\n{skipped} run(s) skipped: curve has no interval columns")
    print(f"\nwrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
