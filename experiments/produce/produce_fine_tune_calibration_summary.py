"""Produce the detector calibration summary from the synthetic tuning sweep.

One row per swept config of every detector: its parameters, whether it
passed the eligibility rules, why not if it failed, false alarms/year and
mean delay per drift kind, and whether it is the frozen winner. Reads
results/tables/fine_tune_on_synthetic.csv and ..._winners.csv only; it does
not re-run any detector.

The sweep stops a config at its first failing row, so per-kind columns are
blank for kinds it never reached, and the all-row averages are given for
eligible configs only (the same averages the winner is ranked on).
"""

from __future__ import annotations

import json

import pandas as pd

from experiments import results_io
from experiments.run.detection.run_fine_tune_on_synthetic import (
    DETECTOR_NAMES,
    DRIFT_KINDS,
    FALSE_ALARM_BUDGET_PER_YEAR,
    KINDS,
    OUTPUT as SWEEP_CSV,
    WINNERS_OUTPUT as WINNERS_CSV,
)

OUTPUT = results_io.TABLES_DIR / "fine_tune_calibration_summary.csv"

# Parameter columns in display order; a detector leaves the others blank.
PARAMETER_COLUMNS = (
    "delta", "clock", "min_window_length",  # ADWIN (delta shared with Page-Hinkley)
    "alpha", "window_size", "stat_size",  # KSWIN (alpha shared with Page-Hinkley)
    "threshold", "min_instances", "mode",  # Page-Hinkley
)


def fail_reason(rows: pd.DataFrame) -> str:
    """Why a config is ineligible, from the row the sweep stopped on."""
    none_rows = rows[rows["kind"] == "none"]
    if (none_rows["false_alarms"] > 0).any():
        return "false alarm on no-drift"
    last = rows.iloc[-1]
    if last["missed_drifts"] > 0:
        return f"missed {last['kind']}"
    if last["false_alarms_per_year"] > FALSE_ALARM_BUDGET_PER_YEAR:
        return f"over false-alarm budget on {last['kind']}"
    return "unknown"


def summarise_config(rows: pd.DataFrame, selected_hashes: set[str]) -> dict:
    first = rows.iloc[0]
    eligible = first["eligibility_status"] == "eligible"
    summary = {"detector": first["detector"], **json.loads(first["detector_parameters"])}
    summary.pop("seed", None)  # KSWIN's fixed internal seed, not a tuned value
    summary["eligible"] = eligible
    summary["fail_reason"] = "" if eligible else fail_reason(rows)

    for kind in KINDS:
        kind_rows = rows[rows["kind"] == kind]
        summary[f"fa_per_year_{kind}"] = kind_rows["false_alarms_per_year"].mean() if len(kind_rows) else None
    summary["fa_per_year_all_rows"] = rows["false_alarms_per_year"].mean() if eligible else None

    for kind in DRIFT_KINDS:
        kind_rows = rows[rows["kind"] == kind]
        summary[f"delay_{kind}"] = kind_rows["detection_delay"].mean() if len(kind_rows) else None
    drift_rows = rows[rows["kind"] != "none"]
    summary["mean_delay"] = drift_rows["detection_delay"].mean() if eligible else None

    summary["config_hash"] = first["config_hash"]
    summary["selected"] = first["config_hash"] in selected_hashes
    return summary


def build_table() -> pd.DataFrame:
    sweep = pd.read_csv(SWEEP_CSV)
    winners = pd.read_csv(WINNERS_CSV)
    selected_hashes = set(winners.loc[winners["status"] == "selected", "config_hash"])

    # sort=False keeps the sweep's row order, so iloc[-1] is the stopping row.
    summary = pd.DataFrame(
        [summarise_config(rows, selected_hashes) for _, rows in sweep.groupby("config_hash", sort=False)]
    )

    # The flagged winner must still rank first among eligible configs (fewest
    # false alarms/year, then shortest mean delay, as select_winners ranks);
    # otherwise the winners file is out of date. Ties are left to the sweep.
    for detector in DETECTOR_NAMES:
        eligible = summary[(summary["detector"] == detector) & summary["eligible"]]
        if eligible.empty:
            continue
        fewest = eligible[eligible["fa_per_year_all_rows"] == eligible["fa_per_year_all_rows"].min()]
        best = fewest[fewest["mean_delay"] == fewest["mean_delay"].min()]
        flagged = summary[(summary["detector"] == detector) & summary["selected"]]
        if len(flagged) != 1 or flagged.iloc[0]["config_hash"] not in set(best["config_hash"]):
            raise RuntimeError(
                f"{detector}: winners file does not match the sweep's best eligible config "
                "-- re-run experiments.run.detection.run_fine_tune_on_synthetic"
            )

    parameters = [column for column in PARAMETER_COLUMNS if column in summary.columns]
    rest = [column for column in summary.columns if column not in parameters and column != "detector"]
    summary = summary[["detector", *parameters, *rest]]
    summary["detector"] = pd.Categorical(summary["detector"], categories=DETECTOR_NAMES, ordered=True)
    return summary.sort_values(["detector", *parameters], kind="stable").reset_index(drop=True)


def main() -> None:
    table = build_table()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)
    print(f"wrote {OUTPUT} ({len(table)} configs)")


if __name__ == "__main__":
    main()
