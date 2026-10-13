"""T3 -- forecast accuracy by regime for every adaptation arm.

Columns (required-outputs T3): model, arm, region, MAE pre-drift / drift /
post-drift (mean ± sd across seeds, errors pooled within each regime),
pinball loss, seeds. Pinball loss is the 90% interval score (mean of the
q = 0.05 and q = 0.95 losses) over every TEST timestamp, with bounds from
fixed split conformal (rescore_adaptation_runs.py); an arm with no
intervals logged reads "pending UQ". Every arm D window is listed; the
official one is marked "(selected)".
"""

from __future__ import annotations

import pandas as pd

from experiments.produce.table_adaptation_common import (
    REGIMES,
    arm_label,
    arm_sort_key,
    latest_adaptation_rows,
    mae_by_run,
    mean_sd,
    official_d_arms,
    pinball_by_run,
)
from experiments.produce.table_image import save_table_image
from experiments.results_io import TABLE3_DIR


def build_table() -> pd.DataFrame:
    latest = latest_adaptation_rows()
    per_run = mae_by_run(latest).merge(
        pinball_by_run(latest), on=["model", "arm", "region", "seed_key"], how="left"
    )
    official = official_d_arms(per_run)
    rows = []
    for (model, arm, region), g in per_run.groupby(["model", "arm", "region"]):
        pinball = mean_sd(g["pinball_loss"])
        rows.append(
            {
                "model": model,
                "arm": arm_label(arm, model, region, official),
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
    table = build_table()
    out = TABLE3_DIR / "table_t3_forecast_accuracy.csv"
    table.to_csv(out, index=False)
    image = save_table_image(
        table,
        TABLE3_DIR / "table_t3_forecast_accuracy.png",
        title="T3 — Forecast accuracy by regime, four adaptation arms",
        subtitle=(
            "MAE pooled within each documented-event regime; pinball loss of the 90% "
            "fixed split conformal interval over all TEST timestamps; mean ± sd across seeds."
        ),
    )
    print(table.to_string(index=False))
    print(f"\nwrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
