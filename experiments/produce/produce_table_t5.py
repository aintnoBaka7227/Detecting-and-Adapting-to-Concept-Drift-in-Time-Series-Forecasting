"""T5 -- coverage and interval width by regime, per UQ method.

Columns: model, arm, region, UQ method, regime, coverage, mean width,
normalised width (mean width / the region's mean TRAIN demand), worst 24h
coverage (lowest window of 48 half-hours inside the regime; "N/A" if none),
complete 24h windows (overlapping, not independent days) and seeds. Values
are mean ± sd across seeds of the metrics logged by run_aemo_uq_replay.py;
global regimes; arm D at 180 days.

Outputs go to <table|figure>/<detector>/[<model>/[<signal>/]] (results_io.output_path);
arms A and B are the shared controls, C and D that detector's runs.
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_adaptation_common import REGIMES, arm_sort_key, mean_sd
from experiments.produce.table_image import save_table_image
from experiments.results_io import RUNS_CSV, TABLE5_DIR, output_path
from experiments.run.uncertainty.run_aemo_uq_replay import AGACI, FIXED
from experiments.trigger_detectors import (
    CONTROL_DETECTOR,
    DETECTOR_LABELS,
    TRIGGER_DETECTORS,
    detector_rows,
    uq_split_id,
)

UQ_LABELS = {FIXED: "fixed split conformal", AGACI: "AgACI"}
METRICS = (
    "coverage",
    "mean_interval_width",
    "normalised_interval_width",
    "worst_24h_coverage",
    "n_complete_windows",
)
RUN_KEYS = ["model", "arm", "region", "uq", "seed_key"]


def latest_uq_rows(detector: str = CONTROL_DETECTOR) -> pd.DataFrame:
    """runs.csv rows of the latest replay of every (model, arm, region, UQ
    method, seed) for one trigger detector (its arms C and D plus the
    shared controls A and B); method is "<model>/<arm>/<uq method>"."""
    hint = "run run_aemo_uq_replay.py first"
    if not RUNS_CSV.exists():
        raise SystemExit(f"{RUNS_CSV} not found -- {hint}")
    rows = detector_rows(pd.read_csv(RUNS_CSV), "uncertainty", uq_split_id, detector).copy()
    if rows.empty:
        raise SystemExit(f"no uncertainty rows for detector {detector!r} -- {hint}")
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


def build_table(detector: str) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = latest_uq_rows(detector)
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
    for detector in TRIGGER_DETECTORS:
        write_tables(detector)


def write_tables(detector: str) -> None:
    table, scales = build_table(detector)
    denominators = ", ".join(f"{region} {scale:,.1f} MW" for region, scale in scales.items()) or "n/a"
    out = output_path(TABLE5_DIR, detector, name=f"table_t5_coverage_and_width_{detector}.csv")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"\nnormalised width denominators (mean TRAIN demand): {denominators}")
    print(f"wrote {out}")
    for model, rows in table.groupby("model"):
        image = save_table_image(
            rows.reset_index(drop=True),
            output_path(TABLE5_DIR, detector, model, name=f"table_t5_coverage_and_width_{detector}_{model}.png"),
            title=f"T5 — Coverage and interval width by regime, {model}, {DETECTOR_LABELS[detector]} trigger",
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
