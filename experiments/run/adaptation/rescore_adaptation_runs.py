"""Re-score logged adaptation runs from their saved forecasts.

No model is refitted. Each latest run is logged again (same config and cost
fields, a later timestamp) after recomputing its global regimes and its 90%
fixed split conformal bounds, calibrated once on that model x region x
seed's Calibration residuals (no pinball loss without a calibration file).
The curve file is rewritten with the new regime, lower and upper columns.
Run it after a regime-rule or interval change, and after new adaptation runs
(T3's pinball loss comes from here).

Usage:
    python -m experiments.run.adaptation.rescore_adaptation_runs
    python -m experiments.run.adaptation.rescore_adaptation_runs --detector page_hinkley
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from drift_lab.uncertainty import FixedSplitConformal
from experiments import results_io
from experiments.produce.adaptation.table_adaptation_common import (
    latest_adaptation_rows,
)
from experiments.run.adaptation.run_aemo_adaptation_arms import regime_labels
from experiments.run_harness import record_run
from experiments.trigger_detectors import (
    CONTROL_ARMS,
    CONTROL_DETECTOR,
    TRIGGER_DETECTORS,
    adaptation_split_id,
)

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
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--detector", choices=TRIGGER_DETECTORS, default=None, help="default: all")
    args = parser.parse_args()

    for detector in [args.detector] if args.detector else TRIGGER_DETECTORS:
        rows = latest_adaptation_rows(detector)
        if detector != CONTROL_DETECTOR:  # the shared controls are re-scored with the control batch
            rows = rows[~rows["arm"].isin(CONTROL_ARMS)]
        rescore(rows, adaptation_split_id(detector))


def rescore(rows: pd.DataFrame, split_id: str) -> None:
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
            split_id=split_id,
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
