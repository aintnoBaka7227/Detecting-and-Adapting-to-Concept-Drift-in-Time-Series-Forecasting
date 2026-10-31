"""Arm D sweep -- forecast accuracy by regime for every arm D window.

T3's format restricted to arm D (retrain on each accepted alarm, on the last
30 / 60 / 120 / 180 days): model, window, region, MAE pre-drift / drift /
post-drift (mean ± sd across seeds), pinball loss of the 90% fixed split
conformal interval ("pending UQ" unless every seed is rescored), seeds, and a last
"winner" column. Per model x region the winner is the window with the lowest
mean drift MAE; "tie" means the runner-up is within one seed sd (the larger
of the two). Same two versions as T3: global regimes (official) and
event-local. The winner is picked on TEST-period drift MAE.

Outputs go to results/tables/arm_d_sweep/<detector>/.

Usage:
    python -m experiments.produce.adaptation.produce_table_arm_d_sweep
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from experiments.produce.adaptation.produce_table_t3 import VERSIONS
from experiments.produce.adaptation.table_adaptation_common import (
    D_PREFIX,
    REGIMES,
    RUN_KEYS,
    arm_sort_key,
    latest_adaptation_rows,
    mae_by_run,
    mae_by_run_event_local,
    mean_sd,
    pinball_by_run,
)
from experiments.produce.shared.table_image import save_table_image
from experiments.results_io import ARM_D_SWEEP_DIR, output_path
from experiments.trigger_detectors import DETECTOR_LABELS, TRIGGER_DETECTORS


def window_label(arm: str) -> str:
    return f"{arm.removeprefix(D_PREFIX).removesuffix('d')} days"


def winners(per_run: pd.DataFrame) -> dict[tuple[str, str], str]:
    """(model, region) -> winner text for the best window's row."""
    stats = (
        per_run.groupby(["model", "region", "arm"])["drift"]
        .agg(["mean", "std"])
        .reset_index()
    )
    picks = {}
    for (model, region), g in stats.groupby(["model", "region"]):
        ranked = g.sort_values("mean").reset_index(drop=True)
        best, second = ranked.iloc[0], ranked.iloc[1]
        tolerance = pd.Series([best["std"], second["std"]]).max()
        tie = pd.notna(tolerance) and second["mean"] - best["mean"] <= tolerance
        picks[(model, region)] = (
            best["arm"],
            f"✓ (tie with {window_label(second['arm'])})" if tie else "✓",
        )
    return picks


def build_table(detector: str, regimes: str = "global") -> pd.DataFrame:
    latest = latest_adaptation_rows(detector)
    latest = latest[latest["arm"].str.startswith(D_PREFIX)]
    by_run = mae_by_run if regimes == "global" else mae_by_run_event_local
    per_run = by_run(latest).merge(pinball_by_run(latest), on=RUN_KEYS, how="left")
    best = winners(per_run)
    rows = []
    for (model, arm, region), g in per_run.groupby(["model", "arm", "region"]):
        scored = int(g["pinball_loss"].notna().sum())
        if scored == len(g):
            pinball = mean_sd(g["pinball_loss"])
        else:
            pinball = "pending UQ" if scored == 0 else f"pending UQ ({scored}/{len(g)} seeds)"
        winner_arm, mark = best[(model, region)]
        rows.append(
            {
                "model": model,
                "arm D window": window_label(arm),
                "region": region,
                **{f"MAE {regime} (MW)": mean_sd(g[regime]) for regime in REGIMES},
                "pinball loss (MW)": pinball,
                "seeds": int(g["seed_key"].nunique()),
                "winner": mark if arm == winner_arm else "",
                "_order": arm_sort_key(arm),
            }
        )
    table = pd.DataFrame(rows).sort_values(["model", "region", "_order"])
    return table.drop(columns="_order").reset_index(drop=True)


def main() -> None:
    for detector, (regimes, (stem, regime_note)) in product(
        TRIGGER_DETECTORS, VERSIONS.items()
    ):
        stem = (
            stem.replace("table_t3_forecast_accuracy", "table_arm_d_sweep")
            + f"_{detector}"
        )
        table = build_table(detector, regimes)
        out = output_path(ARM_D_SWEEP_DIR, detector, name=f"{stem}.csv")
        table.to_csv(out, index=False)
        image = save_table_image(
            table,
            out.with_suffix(".png"),
            title=(
                f"Arm D sweep — Forecast accuracy by regime ({regimes.replace('_', '-')}), "
                f"retrain window 30 / 60 / 120 / 180 days, {DETECTOR_LABELS[detector]} trigger"
            ),
            subtitle=(
                f"{regime_note}; pinball loss of the 90% fixed split conformal interval over all TEST "
                "timestamps; mean ± sd across seeds. Winner: per model and region, the window with the "
                "lowest mean drift MAE (TEST period); tie = runner-up within one seed sd."
            ),
        )
        print(f"\n[{detector} {regimes}]\n{table.to_string(index=False)}")
        print(f"wrote {out}\nwrote {image}")


if __name__ == "__main__":
    main()
