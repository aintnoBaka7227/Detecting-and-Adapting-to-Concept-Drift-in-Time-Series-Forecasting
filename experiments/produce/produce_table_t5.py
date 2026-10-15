"""T5 -- coverage and interval width by regime, per UQ method.

Columns (required-outputs T5): model, arm, region, UQ method, regime,
coverage, mean width, normalised width, worst 24h coverage, complete 24h
windows, seeds. Values
are mean ± sd across seeds of each run's metric
(evaluation.calculate_uq_metrics_by_regime, logged by
run_aemo_uq_replay.py): coverage and width pooled over the regime's
timestamps, and the lowest coverage of any window of 48 consecutive
half-hours lying entirely inside the regime ("N/A" when no window does).
Complete 24h windows counts those eligible windows; they overlap, so it is
not a number of independent days. Regimes are the global rule: every TEST
timestamp has one. Intervals are 90%
and issued one half-hour ahead. Arm D is reported at the 180-day window
only, as in T4.

Normalised width is mean width over the region's mean TRAIN demand
(evaluation.calculate_normalised_interval_width), one fixed denominator per
region logged by the replay as width_scale; both denominators are stated in
the table subtitle.

Writes one CSV with every row and one table image per model.
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_adaptation_common import REGIMES, arm_sort_key, mean_sd
from experiments.produce.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLE5_DIR
from experiments.run.uncertainty.run_aemo_uq_replay import AGACI, FIXED, SPLIT_ID

UQ_LABELS = {FIXED: "fixed split conformal", AGACI: "AgACI"}
METRICS = (
    "coverage",
    "mean_interval_width",
    "normalised_interval_width",
    "worst_24h_coverage",
    "n_complete_windows",
)
RUN_KEYS = ["model", "arm", "region", "uq", "seed_key"]


def latest_uq_rows() -> pd.DataFrame:
    """runs.csv rows of the latest replay of every (model, arm, region, UQ
    method, seed); method is "<model>/<arm>/<uq method>"."""
    hint = "run run_aemo_uq_replay.py first"
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- {hint}")
    runs = pd.read_csv(RUNS_CSV)
    rows = runs[(runs["split_id"] == SPLIT_ID) & (runs["group"] == "uncertainty")].copy()
    if rows.empty:
        raise SystemExit(f"no rows for split_id={SPLIT_ID!r} -- {hint}")
    parts = rows["method"].str.split("/", n=2, expand=True)
    rows["model"], rows["arm"], rows["uq"] = parts[0], parts[1], parts[2]
    rows["seed_key"] = rows["seed"].fillna(-1)
    latest = rows.groupby(RUN_KEYS)["timestamp"].transform("max")
    return rows[rows["timestamp"] == latest]


def width_scales(rows: pd.DataFrame) -> dict[str, float]:
    """Each region's normalised-width denominator, as logged by the replay."""
    scales = rows[rows["metric_name"] == "width_scale"].groupby("region")["metric_value"]
    if (scales.nunique() > 1).any():
        raise SystemExit("width_scale differs between runs of one region -- rerun run_aemo_uq_replay.py")
    return scales.first().to_dict()


def build_table() -> tuple[pd.DataFrame, dict[str, float]]:
    rows = latest_uq_rows()
    scales = width_scales(rows)
    rows = rows[rows["metric_name"].isin(METRICS) & rows["regime"].isin(REGIMES)]
    per_run = (
        rows.pivot_table(index=[*RUN_KEYS, "regime"], columns="metric_name", values="metric_value")
        .reindex(columns=METRICS)
        .reset_index()
    )

    table_rows = []
    for (model, arm, region, uq, regime), g in per_run.groupby(
        ["model", "arm", "region", "uq", "regime"]
    ):
        worst = mean_sd(g["worst_24h_coverage"], 3)
        table_rows.append(
            {
                "model": model,
                "arm": arm,
                "region": region,
                "UQ method": UQ_LABELS.get(uq, uq),
                "regime": regime,
                "coverage": mean_sd(g["coverage"], 3),
                "mean width (MW)": mean_sd(g["mean_interval_width"], 0),
                "normalised width": mean_sd(g["normalised_interval_width"], 3),
                "worst 24h coverage": "N/A" if worst == "n/a" else worst,
                "complete 24h windows": round(g["n_complete_windows"].mean()),
                "seeds": int(g["seed_key"].nunique()),
                "_order": (*arm_sort_key(arm), list(UQ_LABELS).index(uq), REGIMES.index(regime)),
            }
        )
    table = pd.DataFrame(table_rows).sort_values(["model", "region", "_order"])
    return table.drop(columns="_order").reset_index(drop=True), scales


def main() -> None:
    TABLE5_DIR.mkdir(parents=True, exist_ok=True)
    table, scales = build_table()
    denominators = ", ".join(f"{region} {scale:,.1f} MW" for region, scale in scales.items()) or "n/a"
    out = TABLE5_DIR / "table_t5_coverage_and_width.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nnormalised width denominators (mean TRAIN demand): {denominators}")
    print(f"wrote {out}")
    for model, rows in table.groupby("model"):
        image = save_table_image(
            rows.reset_index(drop=True),
            TABLE5_DIR / f"table_t5_coverage_and_width_{model}.png",
            title=f"T5 — Coverage and interval width by regime, {model}",
            subtitle=(
                "90% intervals issued one half-hour ahead; global regimes; coverage and width "
                "pooled within each regime; worst 24h = lowest window of 48 consecutive "
                "half-hours inside the regime (windows overlap; count shown); "
                f"normalised width = MW / mean TRAIN demand ({denominators}); "
                "mean ± sd across seeds (dhr_arima is deterministic: 1 seed)."
            ),
        )
        print(f"wrote {image}")


if __name__ == "__main__":
    main()
