"""T4 -- what each adaptation arm cost.

Columns (required-outputs T4): model, arm, region, retrains, cumulative
train samples, wall clock (s) (cumulative TEST retrain fit time), drift-MAE
gain vs arm A and gain per retrain (evaluation.calculate_adaptation_gain,
paired with arm A on region and seed; negative = better than never
retraining), seeds. Every arm D window is listed; the official one is
marked "(selected)".
"""

from __future__ import annotations

import pandas as pd

from drift_lab.evaluation import calculate_adaptation_gain
from experiments.produce.table_adaptation_common import (
    arm_label,
    arm_sort_key,
    latest_adaptation_rows,
    mae_by_run,
    mean_sd,
    official_d_arms,
)
from experiments.produce.table_image import save_table_image
from experiments.results_io import TABLE4_DIR
from experiments.run.adaptation.run_aemo_adaptation_arms import ARM_A


def build_table() -> pd.DataFrame:
    per_run = mae_by_run(latest_adaptation_rows())
    official = official_d_arms(per_run)
    arm_a = per_run[per_run["arm"] == ARM_A].set_index(["model", "region", "seed_key"])["drift"]

    gains = []
    for run in per_run.itertuples():
        base = arm_a.get((run.model, run.region, run.seed_key))
        if base is None:
            gains.append((float("nan"), float("nan")))
            continue
        gain = calculate_adaptation_gain(run.drift, base, int(run.n_retrains))
        gains.append(
            (gain["drift_mae_difference_vs_arm_a"], gain["drift_mae_difference_per_retrain"])
        )
    per_run[["gain", "gain_per_retrain"]] = pd.DataFrame(gains, index=per_run.index)

    rows = []
    for (model, arm, region), g in per_run.groupby(["model", "arm", "region"]):
        rows.append(
            {
                "model": model,
                "arm": arm_label(arm, model, region, official),
                "region": region,
                "retrains": mean_sd(g["n_retrains"], 0),
                "cumulative train samples": mean_sd(g["train_samples"], 0),
                "wall clock (s)": mean_sd(g["wall_clock_s"], 0),
                "drift MAE gain vs arm A (MW)": mean_sd(g["gain"]),
                "gain per retrain (MW)": mean_sd(g["gain_per_retrain"], 2),
                "seeds": int(g["seed_key"].nunique()),
                "_order": arm_sort_key(arm),
            }
        )
    table = pd.DataFrame(rows).sort_values(["model", "region", "_order"])
    return table.drop(columns="_order").reset_index(drop=True)


def main() -> None:
    TABLE4_DIR.mkdir(parents=True, exist_ok=True)
    table = build_table()
    out = TABLE4_DIR / "table_t4_adaptation_cost.csv"
    table.to_csv(out, index=False)
    image = save_table_image(
        table,
        TABLE4_DIR / "table_t4_adaptation_cost.png",
        title="T4 — Adaptation cost, four adaptation arms",
        subtitle="Gain = drift-regime MAE(arm) − MAE(arm A), paired by region and seed; negative is better.",
    )
    print(table.to_string(index=False))
    print(f"\nwrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
