"""T4 -- what each adaptation arm cost.

Columns (required-outputs T4): model, arm, region, retrains, cumulative
train samples, wall clock (s) (cumulative TEST retrain fit time), drift-MAE
gain vs arm A and gain per retrain (evaluation.calculate_adaptation_gain,
paired with arm A on region and seed; negative = better than never
retraining), seeds. Arm D is reported at the 180-day window only, as in
T3 and T5.

Two versions are written, differing only in which drift regime the gain
uses (TEAM_DECISIONS.md 5.3): global (official) and event-local
(comparison). Retrains and cost are the same in both.

Every output is written once per trigger detector (experiments.trigger_detectors:
ADWIN, Page-Hinkley) with the detector in the file name; arms A and B are the
shared controls, arms C and D are that detector's runs.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from drift_lab.evaluation import calculate_adaptation_gain
from experiments.produce.table_adaptation_common import (
    arm_sort_key,
    drop_unreported_d_arms,
    latest_adaptation_rows,
    mae_by_run,
    mae_by_run_event_local,
    mean_sd,
)
from experiments.produce.table_image import save_table_image
from experiments.results_io import TABLE4_DIR, output_path
from experiments.run.adaptation.run_aemo_adaptation_arms import ARM_A
from experiments.trigger_detectors import DETECTOR_LABELS, TRIGGER_DETECTORS

VERSIONS = {
    "global": ("table_t4_adaptation_cost", "global drift regime"),
    "event_local": ("table_t4_adaptation_cost_event_local", "event-local drift regime (event dates only)"),
}


def build_table(detector: str, regimes: str = "global") -> pd.DataFrame:
    latest = drop_unreported_d_arms(latest_adaptation_rows(detector))
    by_run = mae_by_run if regimes == "global" else mae_by_run_event_local
    per_run = by_run(latest)
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
                "arm": arm,
                "region": region,
                "retrains": round(g["n_retrains"].mean()),
                "cumulative train samples": round(g["train_samples"].mean()),
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
    for detector, (regimes, (stem, regime_note)) in product(TRIGGER_DETECTORS, VERSIONS.items()):
        stem = f"{stem}_{detector}"
        table = build_table(detector, regimes)
        out = output_path(TABLE4_DIR, detector, name=f"{stem}.csv")
        table.to_csv(out, index=False)
        image = save_table_image(
            table,
            output_path(TABLE4_DIR, detector, name=f"{stem}.png"),
            title=f"T4 — Adaptation cost ({regimes.replace('_', '-')}), four adaptation arms, {DETECTOR_LABELS[detector]} trigger",
            subtitle=(
                f"Gain = MAE(arm) − MAE(arm A) over the {regime_note}, paired by region and "
                "seed; negative is better."
            ),
        )
        print(f"\n[{detector} {regimes}]\n{table.to_string(index=False)}")
        print(f"wrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
