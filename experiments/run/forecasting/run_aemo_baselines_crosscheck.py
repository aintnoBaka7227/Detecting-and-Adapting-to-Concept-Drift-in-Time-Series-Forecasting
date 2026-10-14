"""Independent DHR-ARIMA and XGBoost cross-check on the frozen AEMO split.

One `record_run` call per (model, region); nothing is computed here beyond
fit/observe/predict — metrics come from `evaluation.py` inside `record_run`.
Deterministic given the frozen split, so `seed=None` (no seed fabricated).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.dhr_arima_crosscheck import DHRArimaCrosscheck
from drift_lab.forecasting.xgboost_crosscheck import XGBoostCrosscheck
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_frozen_v1"

# Independent implementations used to cross-check the existing baseline results.
MODELS = (XGBoostCrosscheck(), DHRArimaCrosscheck())


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def predict_input(model, test_y: pd.Series) -> pd.DataFrame:
    """Return timestamp-only test input for the frozen cross-check models."""
    return pd.DataFrame(index=test_y.index)


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))

        for model in MODELS:
            t0 = time.perf_counter()
            model.fit(pd.DataFrame(index=train_y.index), train_y)
            model.observe(cal_y)
            preds = model.predict(predict_input(model, test_y))
            wall_clock_s = time.perf_counter() - t0

            record_run(
                method=model.name,
                dataset="aemo",
                region=region,
                seed=None,
                config=config_of(model),
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=len(train_y),
                forecast=(test_y.to_numpy(), preds, test_y.index),
            )
            print(f"{region} {model.name}: wall_clock={wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()



