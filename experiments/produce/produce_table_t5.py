"""T5 -- coverage and interval width by regime, for split conformal.

Reads the per-timestamp curve of the latest logged run of every (model,
arm, region, seed) -- actual, lower, upper, written by
rescore_adaptation_runs.py -- and scores the 90% split conformal interval
with the shared `drift_lab.evaluation.calculate_uq_metrics_by_regime`.
Nothing is refitted. Runs whose curve has no interval columns are skipped
and counted.

Regimes are recomputed here from the curve timestamps, not read from the
curve file, under two rules:

    global       (default, the official table) `assign_timestamp_global_regime`:
                 every TEST timestamp gets exactly one regime -- pre-drift
                 before the first event, drift inside an event's drift window
                 plus the 7-day end buffer, post-drift afterwards.
    event_local  (comparison) `assign_event_local_regime`: only timestamps
                 inside an event's 7-day pre-drift / drift / post-drift
                 windows are labelled; the rest are "unassigned" and belong
                 to no regime.

Columns (required-outputs T5): UQ method, regime, coverage, mean width,
normalised width (mean width / mean TRAIN demand of the region), complete
24h windows, worst 24h coverage, as mean ± sd across seeds.

"complete 24h windows" sits immediately before "worst 24h coverage" on
purpose: it is the number of 48-step windows that lie wholly inside the
regime on consecutive half-hours, and the worst coverage is the minimum
over exactly those windows. Windows overlap, so the count is not a number
of independent days. A regime with fewer than --min-windows complete
windows has too few to compare across 2020-2023; its worst 24h coverage is
shown as "n/a" and the regime is listed in a warning.
"""

from __future__ import annotations

import argparse

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV
from drift_lab.evaluation import calculate_uq_metrics_by_regime, calculate_width_scale
from drift_lab.evaluation.evaluation import (
    UNASSIGNED_REGIME,
    assign_event_local_regime,
    assign_timestamp_global_regime,
    build_event_windows,
)
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
from experiments.run.adaptation.run_aemo_adaptation_arms import MODELS, TARGET_COLUMN

UQ_METHOD = "split conformal (90%)"
REGIME_ORDER = ("pre_drift", "drift", "post_drift")
WINDOW = 48  # 24 hours of half-hours
REGIME_RULES = ("global", "event_local")


def _train_width_scale(region: str) -> float:
    train, _, _ = loader.load(region)
    return calculate_width_scale(train[TARGET_COLUMN])


def regime_labels(timestamps: pd.Series, region: str, rule: str) -> pd.Series:
    """Regime of every curve timestamp under `rule` ("global" or "event_local").

    Under the event-local rule a timestamp outside every event window is
    "unassigned": it stays in the stream (so a 24h window never jumps over
    it) but belongs to no regime.
    """
    if rule not in REGIME_RULES:
        raise ValueError(f"regimes must be one of {REGIME_RULES}")
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    windows = build_event_windows(events, region)
    assign = assign_timestamp_global_regime if rule == "global" else assign_event_local_regime
    labelled = assign(timestamps, windows).set_index("timestamp")["regime"]
    stamps = pd.DatetimeIndex(timestamps)
    return pd.Series(labelled.reindex(stamps).fillna(UNASSIGNED_REGIME).to_numpy())


def uq_by_run(latest: pd.DataFrame, rule: str = "global") -> tuple[pd.DataFrame, int]:
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
            regime_labels(curve["timestamp"], region, rule),
            window=WINDOW,
            width_scale=scales[region],
            timestamps=curve["timestamp"],
        )
        for regime, m in metrics.items():
            rows.append({"model": model, "arm": arm, "region": region,
                         "seed_key": seed_key, "regime": regime, **m})
    return pd.DataFrame(rows), skipped


def build_table(
    latest: pd.DataFrame | None = None,
    rule: str = "global",
    min_windows: int = 1,
    models: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, int]:
    """`models` limits the table to those forecasters (None: every model)."""
    latest = latest_adaptation_rows() if latest is None else latest
    if models:
        latest = latest[latest["model"].isin(models)]
    per_run, skipped = uq_by_run(latest, rule)
    if per_run.empty:
        return per_run, skipped
    official = official_d_arms(mae_by_run(latest))
    rows = []
    for (model, arm, region, regime), g in per_run.groupby(["model", "arm", "region", "regime"]):
        n_windows = int(g["n_complete_windows"].min())
        enough = n_windows >= min_windows
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
                "complete 24h windows": n_windows,
                "worst 24h coverage": (
                    mean_sd(g["worst_rolling_coverage"], 3) if enough else "n/a"
                ),
                "seeds": int(g["seed_key"].nunique()),
                "_order": (arm_sort_key(arm), REGIME_ORDER.index(regime)),
            }
        )
    table = pd.DataFrame(rows)
    table = table.sort_values(["model", "region", "_order"]).drop(columns="_order")
    return table.reset_index(drop=True), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce T5 (coverage and width by regime).")
    parser.add_argument(
        "--regimes",
        choices=REGIME_RULES,
        default="global",
        help="global: every TEST timestamp gets a regime (official, default); "
        "event_local: only timestamps in an event's 7-day windows (comparison)",
    )
    parser.add_argument(
        "--min-windows",
        type=int,
        default=1,
        help="fewest complete 24h windows a regime needs to report a worst 24h coverage",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODELS,
        default=None,
        help="only these forecasters, e.g. --models xgboost (default: every model)",
    )
    args = parser.parse_args()

    results_io.TABLE5_DIR.mkdir(parents=True, exist_ok=True)
    table, skipped = build_table(
        rule=args.regimes,
        min_windows=args.min_windows,
        models=tuple(args.models) if args.models else None,
    )
    if table.empty:
        raise SystemExit(
            "no run has interval columns -- run run_aemo_calibration_forecasts.py "
            "then rescore_adaptation_runs.py first"
        )
    stem = (
        "table_t5_coverage_width"
        + ("_event_local" if args.regimes == "event_local" else "")
        + (f"_{'_'.join(args.models)}" if args.models else "")
    )
    out = results_io.TABLE5_DIR / f"{stem}.csv"
    table.to_csv(out, index=False)
    scope = (
        "every TEST timestamp has a regime"
        if args.regimes == "global"
        else "regimes cover only timestamps in an event's 7-day windows"
    )
    image = save_table_image(
        table,
        results_io.TABLE5_DIR / f"{stem}.png",
        title=f"T5 — Coverage and interval width by regime ({args.regimes.replace('_', '-')}), split conformal",
        subtitle=(
            f"Nominal coverage 0.90; {scope}; normalised width = mean width / mean TRAIN demand; "
            "worst 24h = lowest 48-step rolling coverage over the complete windows shown; "
            "mean ± sd across seeds."
        ),
    )
    print(table.to_string(index=False))
    thin = table[table["complete 24h windows"] < args.min_windows]
    if not thin.empty:
        print(f"\nWARNING: {len(thin)} row(s) have fewer than {args.min_windows} complete 24h windows:")
        print(thin[["model", "arm", "region", "regime", "complete 24h windows"]].to_string(index=False))
    if skipped:
        print(f"\n{skipped} run(s) skipped: curve has no interval columns")
    print(f"\nwrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
