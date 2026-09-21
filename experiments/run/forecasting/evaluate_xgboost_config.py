"""Full-test comparison of XGBoost configs (default vs tuned) on AEMO.

Exploratory sibling of the Step 3 frozen baselines: fits on the training
window, observes the calibration window, then rolls the frozen model
through the full test stream blind (no test actuals) — the exact
information policy of run_aemo_baselines.py. Reports MAE plus 7-day
rolling-MAE mean and max, and dumps the rolling curve so the spike
pattern can be plotted before/after tuning.

Nothing here writes to results/runs.csv.

Usage
    python -m experiments.run.forecasting.evaluate_xgboost_config default
    python -m experiments.run.forecasting.evaluate_xgboost_config tuned
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from drift_lab import evaluation
from drift_lab.aemo import loader
from drift_lab.config import REGIONS, ROLLING_WINDOW_DAYS
from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster
from experiments.results_io import RESULTS_DIR

RESULTS_DIR = RESULTS_DIR / "xgboost_tuning"

ROLLING_WINDOW = 48 * ROLLING_WINDOW_DAYS

NAMED_CONFIGS = {
    "default": {},
    "tuned": {
        "n_estimators": 1200,
        "early_stopping_rounds": 30,
        "validation_fraction": 0.15,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_child_weight": 10,
        "gamma": 0.3,
        "reg_alpha": 0.5,
        "reg_lambda": 3.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name", choices=sorted(NAMED_CONFIGS))
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))

        model = XGBoostForecaster(
            interval="30min",
            max_lag=48,
            extra_lags=(96, 144, 336),
            random_state=42,
            **NAMED_CONFIGS[args.name],
        )

        t0 = time.perf_counter()
        model.fit(pd.DataFrame(index=train_y.index), train_y)
        model.observe(cal_y)
        preds = model.predict(pd.DataFrame(index=test_y.index))
        wall_clock_s = time.perf_counter() - t0

        curve = evaluation.calculate_rolling_mae(
            test_y.to_numpy(), preds, window=ROLLING_WINDOW
        )
        curve = pd.Series(curve.to_numpy(), index=pd.Index(test_y.index, name="index"))

        curve_dir = RESULTS_DIR / "curves"
        curve_dir.mkdir(parents=True, exist_ok=True)
        curve.rename("rolling_mae_7d").to_csv(
            curve_dir / f"{args.name}_{region}.csv", header=True
        )

        mae = evaluation.calculate_mae(test_y.to_numpy(), preds)
        finite = curve[np.isfinite(curve)]
        row = {
            "config": args.name,
            "region": region,
            "mae": mae,
            "rolling_mae_7d_mean": float(np.nanmean(finite)),
            "rolling_mae_7d_max": float(np.nanmax(finite)),
            "wall_clock_s": wall_clock_s,
        }
        print(f"{args.name} {region}: {row}", flush=True)

        out = RESULTS_DIR / "full_test_eval.csv"
        previous = pd.read_csv(out) if out.exists() else pd.DataFrame()
        pd.concat([previous, pd.DataFrame([row])], ignore_index=True).to_csv(
            out, index=False
        )


if __name__ == "__main__":
    main()
