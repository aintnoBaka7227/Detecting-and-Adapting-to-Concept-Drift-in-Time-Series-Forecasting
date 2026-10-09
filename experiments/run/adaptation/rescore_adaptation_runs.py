"""Re-score logged adaptation runs from their saved forecasts.

No model is refitted. For the latest logged run of every (model, arm,
region, seed) this reads the saved per-timestamp curve (actual, forecast)
and logs the run again through `record_run`, with its original config and
cost fields, after recomputing:

    regimes     the documented-event regime of every TEST timestamp, from
                the current `evaluation.assign_regime`;
    intervals   90% fixed split conformal bounds (`FixedSplitConformal`),
                calibrated once on that model x region x seed's
                Calibration-window residuals from
                run_aemo_calibration_forecasts.py. Every arm of a stream
                shares the one radius, and it is never updated from TEST
                outcomes or at a retrain. A run with no calibration file
                is re-scored without intervals (no pinball loss).

The new rows carry a later timestamp, so the T3 / T4 producers pick them
up; the curve file is rewritten in place with the new `regime`, `lower`
and `upper` columns.

Run after a change to the regime rules or the interval method, so every
run is scored the same way.

Usage:
    python -m experiments.run.adaptation.rescore_adaptation_runs
"""

from __future__ import annotations

import json

import pandas as pd

from drift_lab.uncertainty import FixedSplitConformal
from experiments import results_io
from experiments.produce.table_adaptation_common import latest_adaptation_rows
from experiments.run.adaptation.run_aemo_adaptation_arms import SPLIT_ID, regime_labels
from experiments.run_harness import record_run

INTERVAL_ALPHA = 0.10


def fixed_radius(model: str, region: str, seed: int | None) -> float | None:
    """The stream's fixed split conformal radius, or None without a
    calibration file."""
    path = results_io.calibration_path(model, region, seed)
    if not path.exists():
        return None
    residuals = pd.read_csv(path)["residual"]
    return FixedSplitConformal(alpha=INTERVAL_ALPHA).calibrate(residuals).radius


def main() -> None:
    rows = latest_adaptation_rows()
    labels_by_region: dict[str, pd.Series] = {}
    radius_by_stream: dict[tuple, float | None] = {}

    keys = ["method", "region", "seed_key", "config_hash"]
    for (method, region, seed_key, chash), run in rows.groupby(keys):
        seed = None if seed_key == -1 else int(seed_key)
        model = method.split("/", 1)[0]
        config = json.loads((results_io.RUNS_DIR / chash / "config.json").read_text())
        if results_io.config_hash(config) != chash:
            raise RuntimeError(f"{chash}: config.json no longer hashes to its folder name")

        curve = pd.read_csv(
            results_io.curve_path(chash, "aemo", region, seed), parse_dates=["timestamp"]
        )
        index = pd.DatetimeIndex(curve["timestamp"])
        retrains = pd.read_csv(
            results_io.retrains_path(chash, "aemo", region, seed), parse_dates=["timestamp"]
        )
        if region not in labels_by_region:
            labels_by_region[region] = regime_labels(index, region)
        labels = labels_by_region[region].reindex(index)

        stream = (model, region, seed)
        if stream not in radius_by_stream:
            radius_by_stream[stream] = fixed_radius(*stream)
        radius = radius_by_stream[stream]
        interval = None
        if radius is not None:
            interval = (curve["forecast"] - radius, curve["forecast"] + radius, INTERVAL_ALPHA)

        first = run.iloc[0]
        record_run(
            method=method,
            dataset="aemo",
            region=region,
            seed=seed,
            config=config,
            wall_clock_s=float(first["wall_clock_s"]),
            split_id=SPLIT_ID,
            train_samples=int(first["train_samples"]),
            n_retrains=int(first["n_retrains"]),
            forecast=(curve["actual"].to_numpy(), curve["forecast"].to_numpy(), index),
            regime_labels=labels.to_numpy(),
            retrain_timestamps=retrains["timestamp"],
            interval=interval,
        )
        note = "no calibration file" if radius is None else f"radius {radius:.1f}"
        print(f"{method} {region} seed={seed}: {note}", flush=True)


if __name__ == "__main__":
    main()
