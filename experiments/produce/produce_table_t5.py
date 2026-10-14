"""T5 -- coverage and interval width by regime, per UQ method.

Columns (required-outputs T5): model, arm, region, UQ method, regime,
coverage, mean width, normalised width, worst 24h coverage, seeds. Values
are mean ± sd across seeds of each run's metric
(evaluation.calculate_uq_metrics_by_regime, logged by
run_aemo_uq_replay.py): coverage and width pooled over the regime's
timestamps, and the lowest coverage of any 48-half-hour window lying
entirely inside the regime ("N/A" when no window does). Intervals are 90%
and issued one half-hour ahead. Arm D is reported at the 180-day window
only, as in T4.

Normalised width is left blank: its denominator is not yet agreed.

Writes one CSV with every row and one table image per model.
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_adaptation_common import REGIMES, arm_sort_key, mean_sd
from experiments.produce.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLE5_DIR
from experiments.run.uncertainty.run_aemo_uq_replay import AGACI, FIXED, SPLIT_ID

UQ_LABELS = {FIXED: "fixed split conformal", AGACI: "AgACI"}
METRICS = ("coverage", "mean_interval_width", "worst_24h_coverage")
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


def build_table() -> pd.DataFrame:
    rows = latest_uq_rows()
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
                "normalised width": "",
                "worst 24h coverage": "N/A" if worst == "n/a" else worst,
                "seeds": int(g["seed_key"].nunique()),
                "_order": (*arm_sort_key(arm), list(UQ_LABELS).index(uq), REGIMES.index(regime)),
            }
        )
    table = pd.DataFrame(table_rows).sort_values(["model", "region", "_order"])
    return table.drop(columns="_order").reset_index(drop=True)


def main() -> None:
    TABLE5_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    out = TABLE5_DIR / "table_t5_coverage_and_width.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out}")
    for model, rows in table.groupby("model"):
        image = save_table_image(
            rows.reset_index(drop=True),
            TABLE5_DIR / f"table_t5_coverage_and_width_{model}.png",
            title=f"T5 — Coverage and interval width by regime, {model}",
            subtitle=(
                "90% intervals issued one half-hour ahead; coverage and width pooled within each "
                "regime; worst 24h = lowest 48-half-hour window inside the regime; "
                "mean ± sd across seeds."
            ),
        )
        print(f"wrote {image}")


if __name__ == "__main__":
    main()
