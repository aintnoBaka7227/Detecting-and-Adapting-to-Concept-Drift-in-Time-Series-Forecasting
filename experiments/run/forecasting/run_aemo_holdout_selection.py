"""Forecaster selection on a validation window cut from the end of TRAIN.

The supervisor's protocol: train every forecaster on TRAIN, validate on a
period taken from the end of TRAIN, pick the best, freeze it. The three
candidate forecasters (dhr_arima: deterministic, once; xgboost / nhits:
config.SEEDS) are fitted on TRAIN before VALIDATION_START and forecast the
held-out VALIDATION_START .. end of TRAIN (Nov-Dec 2019, the Calibration
window's length) with their TEST protocol: NHITS in 48-step blocks with real
context, xgboost / dhr_arima blind. Seasonal naive is the baseline, not a
candidate, so it is not run here. Neither the Calibration window (conformal residuals) nor
TEST is touched. Runs one at a time with a fixed thread count, so timings
compare; each model runs in its own process (see main).

Recorded through record_run under split_id SPLIT_ID, method "<model>":
    runs.csv              mae and rolling_mae_7d_mean / _max over the
                          validation window, fit_seconds, predict_seconds,
                          wall_clock_s = fit + predict;
    results/runs/<hash>/  config.json (model settings, protocol, windows,
                          timing conditions) and curve_aemo_<region>_<seed>.csv.
produce_table_forecaster_selection.py picks the forecaster from these rows.

Usage:
    python -m experiments.run.forecasting.run_aemo_holdout_selection
    python -m experiments.run.forecasting.run_aemo_holdout_selection --model nhits --region SA1 --seed 3
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
import warnings

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS, SEEDS
from experiments.run.adaptation.run_aemo_adaptation_arms import (
    SEEDED_MODELS,
    demand_series,
    make_model,
    predict_input,
)
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_holdout_selection_nov_dec_2019_v1"
MODELS = ("dhr_arima", "xgboost", "nhits")
VALIDATION_START = pd.Timestamp("2019-11-01")  # [Nov 1 00:00, end of TRAIN)
TORCH_THREADS = 4  # NHITS; xgboost's own n_jobs is 4 too
PROTOCOL = {
    "nhits": "48-step blocks with real context between blocks",
    "xgboost": "blind roll-forward from the fit",
    "dhr_arima": "blind roll-forward from the fit",
}


def run_one(model_name: str, region: str, seed: int | None) -> None:
    train, _, _ = loader.load(region)
    y = demand_series(train)
    fit_y, val_y = y[y.index < VALIDATION_START], y[y.index >= VALIDATION_START]

    model = make_model(model_name, seed)
    t0 = time.perf_counter()
    model.fit(pd.DataFrame(index=fit_y.index), fit_y)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    forecast = model.predict(predict_input(model_name, val_y))
    predict_seconds = time.perf_counter() - t1

    rows = record_run(
        method=model_name,
        dataset="aemo",
        region=region,
        seed=seed,
        config={
            **config_of(model),
            "model": model_name,
            "protocol": PROTOCOL[model_name],
            "evaluated_on": "validation cut from the end of TRAIN",
            "fit_window": [str(fit_y.index.min()), str(fit_y.index.max())],
            "validation_window": [str(val_y.index.min()), str(val_y.index.max())],
            "timing": {
                "device": "cpu",
                "torch_threads": TORCH_THREADS,
                "sequential": True,
            },
        },
        wall_clock_s=fit_seconds + predict_seconds,
        split_id=SPLIT_ID,
        train_samples=len(fit_y),
        forecast=(val_y.to_numpy(), forecast, val_y.index),
        extra_metrics={"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
    )
    print(
        f"[{model_name} {region} seed={seed}] {len(val_y)} validation forecasts, "
        f"fit {fit_seconds:.1f}s, predict {predict_seconds:.1f}s -> {rows['config_hash'].iloc[0]}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecaster selection on a validation window cut from TRAIN."
    )
    parser.add_argument(
        "--model", choices=MODELS, default=None, help="default: every model"
    )
    parser.add_argument(
        "--region", choices=REGIONS, default=None, help="default: every region"
    )
    parser.add_argument(
        "--seed", type=int, choices=SEEDS, default=None, help="default: every seed"
    )
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS)
    if args.seed is not None and any(m not in SEEDED_MODELS for m in models):
        parser.error(f"--seed applies to {SEEDED_MODELS} only")
    if len(models) > 1:
        # One process per model: torch (NHITS) and xgboost load clashing OpenMP
        # runtimes on macOS and crash in one process, whichever loads first.
        for model_name in models:
            command = [sys.executable, "-m", __spec__.name, "--model", model_name]
            if args.region:
                command += ["--region", args.region]
            if args.seed is not None:
                command += ["--seed", str(args.seed)]
            subprocess.run(command, check=True)
        return
    if "nhits" in models:
        import torch

        torch.set_num_threads(TORCH_THREADS)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    regions = [args.region] if args.region else list(REGIONS)
    for model_name in models:
        seeds = (
            [args.seed]
            if args.seed is not None
            else (list(SEEDS) if model_name in SEEDED_MODELS else [None])
        )
        for region in regions:
            for seed in seeds:
                run_one(model_name, region, seed)


if __name__ == "__main__":
    main()
