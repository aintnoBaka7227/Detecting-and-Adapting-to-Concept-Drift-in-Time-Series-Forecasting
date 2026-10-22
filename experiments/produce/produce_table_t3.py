"""T3 -- forecast accuracy by regime for every adaptation arm.

Columns: model, arm, region, MAE pre-drift / drift / post-drift (mean ± sd
across seeds, pooled within each regime), pinball loss of the 90% fixed
split conformal interval over all TEST timestamps (from
rescore_adaptation_runs.py; "pending UQ" if not yet rescored) and seeds.
Arm D at 180 days. Two versions: global regimes (official) and event-local
(only the 7 days before, during and 7 days after an event).

Outputs go to <table|figure>/<detector>/[<model>/] (results_io.output_path);
arms A and B are the shared controls, C and D that detector's runs.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from experiments.produce.table_adaptation_common import (
    REGIMES,
    RUN_KEYS,
    arm_sort_key,
    drop_unreported_d_arms,
    latest_adaptation_rows,
    mae_by_run,
    mae_by_run_event_local,
    mean_sd,
    pinball_by_run,
)
from experiments.produce.table_image import save_table_image
from experiments.results_io import TABLE3_DIR, output_path
from experiments.trigger_detectors import DETECTOR_LABELS, TRIGGER_DETECTORS

VERSIONS = {
    "global": ("table_t3_forecast_accuracy", "MAE pooled within each global regime"),
    "event_local": (
        "table_t3_forecast_accuracy_event_local",
        (
            "MAE pooled within each event-local regime (7 days before, during, 7 days after "
            "each event; other timestamps left out)"
        ),
    ),
}


def build_table(detector: str, regimes: str = "global") -> pd.DataFrame:
    latest = drop_unreported_d_arms(latest_adaptation_rows(detector))
    by_run = mae_by_run if regimes == "global" else mae_by_run_event_local
    per_run = by_run(latest).merge(pinball_by_run(latest), on=RUN_KEYS, how="left")
    rows = []
    for (model, arm, region), g in per_run.groupby(["model", "arm", "region"]):
        pinball = mean_sd(g["pinball_loss"])
        rows.append(
            {
                "model": model,
                "arm": arm,
                "region": region,
                **{f"MAE {regime} (MW)": mean_sd(g[regime]) for regime in REGIMES},
                "pinball loss (MW)": "pending UQ" if pinball == "n/a" else pinball,
                "seeds": int(g["seed_key"].nunique()),
                "_order": arm_sort_key(arm),
            }
        )
    table = pd.DataFrame(rows).sort_values(["model", "region", "_order"])
    return table.drop(columns="_order").reset_index(drop=True)


def main() -> None:
    TABLE3_DIR.mkdir(parents=True, exist_ok=True)
    for detector, (regimes, (stem, regime_note)) in product(TRIGGER_DETECTORS, VERSIONS.items()):
        stem = f"{stem}_{detector}"
        table = build_table(detector, regimes)
        out = output_path(TABLE3_DIR, detector, name=f"{stem}.csv")
        table.to_csv(out, index=False)
        image = save_table_image(
            table,
            output_path(TABLE3_DIR, detector, name=f"{stem}.png"),
            title=f"T3 — Forecast accuracy by regime ({regimes.replace('_', '-')}), four adaptation arms, {DETECTOR_LABELS[detector]} trigger",
            subtitle=(
                f"{regime_note}; pinball loss of the 90% fixed split conformal interval "
                "over all TEST timestamps; mean ± sd across seeds."
            ),
        )
        print(f"\n[{detector} {regimes}]\n{table.to_string(index=False)}")
        print(f"wrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
