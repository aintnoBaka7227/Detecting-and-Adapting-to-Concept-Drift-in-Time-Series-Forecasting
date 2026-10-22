"""Step 3 / F1 — fit the frozen baselines on AEMO and roll them through TEST.

One `record_run` per (model, region, seed): fit on TRAIN, observe the
Calibration window, forecast TEST; metrics come from evaluation.py inside
record_run. seasonal_naive and dhr_arima are deterministic (seed=None).
xgboost is stochastic, so it runs for every config.SEEDS with
random_state = seed, built by run_aemo_adaptation_arms.make_model so the
frozen baseline is the same model as arm A. NHITS runs in run_aemo_nhits.py.

Usage:
    python -m experiments.run.forecasting.run_aemo_baselines
    python -m experiments.run.forecasting.run_aemo_baselines --models xgboost
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS, SEEDS
from drift_lab.forecasting.seasonal_naive import SeasonalNaive
from experiments.run.adaptation.run_aemo_adaptation_arms import make_model
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_frozen_v1"
MODELS = ("seasonal_naive", "xgboost", "dhr_arima")
SEEDED_MODELS = ("xgboost",)


def build(name: str, seed: int | None):
    """A fresh model; xgboost / dhr_arima exactly as the adaptation arms build them."""
    return SeasonalNaive() if name == "seasonal_naive" else make_model(name, seed)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def predict_input(model, test_y: pd.Series) -> pd.DataFrame:
    """seasonal_naive needs the real lag-48 values to walk forward with;
    xgboost / dhr_arima are frozen — no target column, no feedback."""
    if model.name == "seasonal_naive":
        return pd.DataFrame({"TOTALDEMAND": test_y.to_numpy()}, index=test_y.index)
    return pd.DataFrame(index=test_y.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen AEMO baselines (F1).")
    parser.add_argument("--models", default=",".join(MODELS), help=f"comma list from {MODELS}")
    args = parser.parse_args()
    names = [m.strip() for m in args.models.split(",") if m.strip()]
    if unknown := sorted(set(names) - set(MODELS)):
        parser.error(f"unknown models {unknown}")

    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))

        for name in names:
            for seed in SEEDS if name in SEEDED_MODELS else (None,):
                model = build(name, seed)
                t0 = time.perf_counter()
                model.fit(pd.DataFrame(index=train_y.index), train_y)
                model.observe(cal_y)
                preds = model.predict(predict_input(model, test_y))
                wall_clock_s = time.perf_counter() - t0

                record_run(
                    method=model.name,
                    dataset="aemo",
                    region=region,
                    seed=seed,
                    config=config_of(model),
                    wall_clock_s=wall_clock_s,
                    split_id=SPLIT_ID,
                    train_samples=len(train_y),
                    forecast=(test_y.to_numpy(), preds, test_y.index),
                )
                print(f"{region} {model.name} seed={seed}: wall_clock={wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()
