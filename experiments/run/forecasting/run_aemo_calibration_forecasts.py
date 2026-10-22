"""Stage 4 -- Calibration-window forecasts: forecaster selection and conformal residuals.

For every model x region x seed (nhits / xgboost: config.SEEDS; dhr_arima:
deterministic, once): fit on TRAIN, then forecast the Calibration window with
the model's TEST protocol (NHITS in 48-step blocks with real context;
xgboost / dhr_arima blind from the TRAIN fit). Models come from
run_aemo_adaptation_arms.make_model, so they are the models every later stage
uses. No Calibration outcome is seen before its forecast, and no TEST value is
touched. Runs one at a time with a fixed thread count, so timings compare.

Recorded through record_run under split_id SPLIT_ID, method "<model>":
    runs.csv              mae and rolling_mae_7d_mean / _max over the Calibration
                          window, fit_seconds (TRAIN), predict_seconds
                          (Calibration), wall_clock_s = fit + predict;
    results/runs/<hash>/  config.json (model settings, protocol, timing
                          conditions), curve_aemo_<region>_<seed>.csv (rolling
                          MAE) and calibration_forecast_aemo_<region>_<seed>.csv
                          (timestamp, actual, forecast, residual): the exact
                          residuals this run produced.
produce_table_forecaster_selection.py picks the forecaster from these rows.
The later stages (rescore, UQ replay, escalation) read the residuals through
results_io.calibration_path, which resolves the latest run's hash folder, so
a rerun of this stage is picked up by every later stage.

Usage:
    python -m experiments.run.forecasting.run_aemo_calibration_forecasts
    python -m experiments.run.forecasting.run_aemo_calibration_forecasts --model xgboost --region SA1 --seed 3
"""

from __future__ import annotations

import argparse
import logging
import time
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
from experiments.run_harness import config_of, record_run

SPLIT_ID = results_io.CALIBRATION_SPLIT_ID
TORCH_THREADS = 4  # NHITS; xgboost's own n_jobs is 4 too
PROTOCOL = {
    "nhits": "48-step blocks with real context between blocks",
    "xgboost": "blind roll-forward from the TRAIN fit",
    "dhr_arima": "blind roll-forward from the TRAIN fit",
}


def run_one(model_name: str, region: str, seed: int | None) -> None:
    train, calibration, _ = loader.load(region)
    train_y, cal_y = demand_series(train), demand_series(calibration)

    model = make_model(model_name, seed)
    t0 = time.perf_counter()
    model.fit(pd.DataFrame(index=train_y.index), train_y)
    fit_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    forecast = model.predict(predict_input(model_name, cal_y))
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
            "evaluated_on": "calibration",
            "timing": {"device": "cpu", "torch_threads": TORCH_THREADS, "sequential": True},
        },
        wall_clock_s=fit_seconds + predict_seconds,
        split_id=SPLIT_ID,
        train_samples=len(train_y),
        forecast=(cal_y.to_numpy(), forecast, cal_y.index),
        extra_metrics={"fit_seconds": fit_seconds, "predict_seconds": predict_seconds},
    )

    frame = pd.DataFrame({"timestamp": cal_y.index, "actual": cal_y.to_numpy(), "forecast": forecast})
    frame["residual"] = frame["actual"] - frame["forecast"]
    seed_token = "none" if seed is None else str(seed)
    run_dir = results_io.RUNS_DIR / rows["config_hash"].iloc[0]
    frame.to_csv(run_dir / f"calibration_forecast_aemo_{region}_{seed_token}.csv", index=False)
    print(
        f"[{model_name} {region} seed={seed}] {len(frame)} {TARGET_COLUMN} forecasts, "
        f"fit {fit_seconds:.1f}s, predict {predict_seconds:.1f}s -> {run_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration-window forecasts (stage 4).")
    parser.add_argument("--model", choices=MODELS, default=None, help="default: every model")
    parser.add_argument("--region", choices=REGIONS, default=None, help="default: every region")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=None, help="default: every seed")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODELS)
    if args.seed is not None and any(m not in SEEDED_MODELS for m in models):
        parser.error(f"--seed applies to {SEEDED_MODELS} only")
    if "nhits" in models:
        import torch

        torch.set_num_threads(TORCH_THREADS)
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")

    regions = [args.region] if args.region else list(REGIONS)
    for model_name in models:
        seeds = [args.seed] if args.seed is not None else (list(SEEDS) if model_name in SEEDED_MODELS else [None])
        for region in regions:
            for seed in seeds:
                run_one(model_name, region, seed)


if __name__ == "__main__":
    main()
