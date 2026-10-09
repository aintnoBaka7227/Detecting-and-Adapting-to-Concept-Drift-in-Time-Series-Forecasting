"""Calibration-window forecasts for the conformal intervals (Step 6).

For every model x region x seed this fits the model on TRAIN exactly as
run_aemo_adaptation_arms.py does before any arm starts, then forecasts the
Calibration window (SPLIT["calibration"]) with the same per-model
protocol used on TEST: NHITS in 48-step blocks with real context between
blocks, xgboost / dhr_arima blind from the TRAIN fit. No Calibration
outcome is seen before its forecast is issued, and no TEST value is
touched.

Output, one file per stream:

    results/runs/calibration/calibration_<model>_<region>_<seed>.csv
        timestamp, actual, forecast, residual (= actual - forecast)

The residuals are the calibration scores of every arm of that model x
region x seed: all arms start from this same TRAIN-fitted model.

Usage:
    python -m experiments.run.forecasting.run_aemo_calibration_forecasts --model nhits
    python -m experiments.run.forecasting.run_aemo_calibration_forecasts --model xgboost --region SA1 --seed 3
"""

from __future__ import annotations

import argparse
import logging
import warnings

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS, SEEDS
from experiments import results_io
from experiments.run.adaptation.run_aemo_adaptation_arms import (
    MODELS,
    SEEDED_MODELS,
    TARGET_COLUMN,
    demand_series,
    make_model,
    predict_input,
)


def run_one(model_name: str, region: str, seed: int | None) -> None:
    train, calibration, _ = loader.load(region)
    train_y, cal_y = demand_series(train), demand_series(calibration)

    model = make_model(model_name, seed)
    model.fit(pd.DataFrame(index=train_y.index), train_y)
    forecast = model.predict(predict_input(model_name, cal_y))

    frame = pd.DataFrame(
        {
            "timestamp": cal_y.index,
            "actual": cal_y.to_numpy(),
            "forecast": forecast,
        }
    )
    frame["residual"] = frame["actual"] - frame["forecast"]
    path = results_io.calibration_path(model_name, region, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"[{model_name} {region} seed={seed}] {len(frame)} {TARGET_COLUMN} forecasts -> {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast the Calibration window per model.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--region", choices=REGIONS, default=None, help="default: every region")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None, help="default: every seed")
    args = parser.parse_args()

    seeded = args.model in SEEDED_MODELS
    if not seeded and args.seed is not None:
        parser.error(f"{args.model} is deterministic; --seed applies to {SEEDED_MODELS} only")
    if args.model == "nhits":
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    regions = [args.region] if args.region else list(REGIONS)
    seeds = [args.seed] if args.seed is not None else (list(SEEDS) if seeded else [None])
    for region in regions:
        for seed in seeds:
            run_one(args.model, region, seed)


if __name__ == "__main__":
    main()
