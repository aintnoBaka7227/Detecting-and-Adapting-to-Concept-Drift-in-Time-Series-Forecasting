"""Randomised hyperparameter search for XGBoostForecaster on the frozen split.

Tuning protocol
---------------
- The search space is sampled with a fixed seed (reproducible) and each
  candidate is fitted on the training window (2018-2019) then rolled
  through the CALIBRATION window (Jan-Feb 2020) — the naturally
  leak-free holdout between train and test. No test-period data is used
  to choose parameters.
- Candidates are scored on calibration MAE plus 7-day rolling-MAE mean
  and max (the max is the spike metric the frozen baseline spikes on).
- The winning config(s) can then be re-rolled through the full test
  stream for a before/after look at the actual rolling-MAE curve.

This is exploratory: nothing here writes to results/runs.csv (the frozen
Step 3 ledger). A summary CSV is written under results/xgboost_tuning/.

Usage
-----
    python -m experiments.run.forecasting.tune_xgboost --n-samples 24
"""

from __future__ import annotations

import argparse
import random
import time

import numpy as np
import pandas as pd

from drift_lab import evaluation
from drift_lab.aemo import loader
from drift_lab.config import REGIONS, ROLLING_WINDOW_DAYS
from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster
from experiments.results_io import RESULTS_DIR

RESULTS_DIR = RESULTS_DIR / "xgboost_tuning"

SEARCH_SPACE = {
    "learning_rate": [0.02, 0.05, 0.1],
    "max_depth": [3, 5, 7],
    "min_child_weight": [1, 5, 10],
    "gamma": [0.0, 0.3, 1.0],
    "reg_alpha": [0.0, 0.5, 2.0],
    "reg_lambda": [1.0, 3.0, 10.0],
}

FIXED = {
    "n_estimators": 1200,
    "early_stopping_rounds": 30,
    "validation_fraction": 0.15,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

ROLLING_WINDOW = 48 * ROLLING_WINDOW_DAYS


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def sample_config(rng: random.Random) -> dict:
    config = {name: rng.choice(values) for name, values in SEARCH_SPACE.items()}
    config.update(FIXED)
    return config


def score_on_calibration(model, cal_y: pd.Series) -> dict:
    t0 = time.perf_counter()
    preds = model.predict(pd.DataFrame(index=cal_y.index))
    curve = evaluation.calculate_rolling_mae(
        cal_y.to_numpy(), preds, window=ROLLING_WINDOW
    )
    curve = curve[np.isfinite(curve)]
    return {
        "mae": evaluation.calculate_mae(cal_y.to_numpy(), preds),
        "rolling_mae_7d_mean": float(np.nanmean(curve)),
        "rolling_mae_7d_max": float(np.nanmax(curve)),
        "wall_clock_s": time.perf_counter() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    configs = []

    seen = set()
    while len(configs) < args.n_samples:
        config = sample_config(rng)
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        seen.add(key)
        configs.append(config)

    data = {
        region: (
            demand_series(loader.load(region)[0]),
            demand_series(loader.load(region)[1]),
        )
        for region in REGIONS
    }

    for region, (train_y, cal_y) in data.items():
        rows = []
        for config in configs:
            model = XGBoostForecaster(
                interval="30min",
                max_lag=48,
                extra_lags=(96, 144, 336),
                random_state=42,
                **config,
            )
            model.fit(pd.DataFrame(index=train_y.index), train_y)
            metrics = score_on_calibration(model, cal_y)
            rows.append({**config, **metrics})
            print(
                f"{region}: depth={config['max_depth']} lr={config['learning_rate']} "
                f"mwc={config['min_child_weight']} gamma={config['gamma']} "
                f"a={config['reg_alpha']} l={config['reg_lambda']} -> "
                f"mae={metrics['mae']:.1f} roll_mean={metrics['rolling_mae_7d_mean']:.1f} "
                f"roll_max={metrics['rolling_mae_7d_max']:.1f}",
                flush=True,
            )

        frame = pd.DataFrame(rows).sort_values("rolling_mae_7d_max")
        frame.to_csv(RESULTS_DIR / f"{region}_search.csv", index=False)
        print(
            f"{region} best-by-spike: "
            f"{frame.iloc[0][['learning_rate', 'max_depth', 'min_child_weight', 'gamma', 'reg_alpha', 'reg_lambda', 'rolling_mae_7d_max']].to_dict()}"
        )


if __name__ == "__main__":
    main()
