"""T1 — detection on the synthetic benchmark, aggregated across seeds.

Reads results/runs.csv (the one sanctioned top-level read) and group-bys it;
no metric is computed here that evaluation.py didn't already produce.
"""

from __future__ import annotations

import json

import pandas as pd

from experiments.results_io import FIGURES_DIR, RUNS_CSV, RUNS_DIR

THRESHOLD_LABEL_BY_METHOD = {
    "adwin": lambda c: f"delta = {c['delta']}",
    "kswin": lambda c: f"alpha = {c['alpha']}",
    "page_hinkley": lambda c: f"threshold = {c['threshold']}",
}

TABLE_COLUMNS = [
    "detector",
    "drift type",
    "delay (mean ± sd)",
    "false alarms / 10k",
    "missed",
    "threshold",
    "seeds",
]


def threshold_label_for(config_hash: str, method: str) -> str:
    config = json.loads((RUNS_DIR / config_hash / "config.json").read_text())
    label = THRESHOLD_LABEL_BY_METHOD.get(method)
    return label(config) if label else json.dumps(config)


def format_delay_summary(delay: pd.Series) -> str:
    valid = delay.dropna()
    if valid.empty:
        return "n/a"
    if len(valid) == 1:
        return f"{valid.mean():.0f} ± n/a"
    return f"{valid.mean():.0f} ± {valid.std():.0f}"


def build_table() -> pd.DataFrame:
    df = pd.read_csv(RUNS_CSV)
    det = df[(df["group"] == "detection") & df["dataset"].str.startswith("synthetic_")].copy()
    det["drift_type"] = det["dataset"].str.removeprefix("synthetic_")

    rows = []
    for (method, drift_type), g in det.groupby(["method", "drift_type"]):
        delay = g.loc[g["metric_name"] == "detection_delay", "metric_value"]
        alarms = g.loc[g["metric_name"] == "false_alarms_per_10000", "metric_value"]
        missed = g.loc[g["metric_name"] == "missed_detections", "metric_value"]
        rows.append(
            {
                "detector": method,
                "drift type": drift_type,
                "delay (mean ± sd)": format_delay_summary(delay),
                "false alarms / 10k": round(alarms.mean(), 1),
                "missed": int(missed.sum()),
                "threshold": threshold_label_for(g["config_hash"].iloc[0], method),
                "seeds": g["seed"].nunique(),
            }
        )

    return pd.DataFrame(rows, columns=TABLE_COLUMNS).sort_values(["detector", "drift type"]).reset_index(drop=True)


def main() -> None:
    table = build_table()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "table_t1_synthetic_detection.csv"
    table.to_csv(out_path, index=False)
    print(table.to_string(index=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
