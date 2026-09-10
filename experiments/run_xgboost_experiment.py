"""Step 3 / F1 — fit the frozen XGBoost forecaster on AEMO and roll it through test.

Same frozen train/calibration/test split and blind `predict_input()` policy as
`run_aemo_baselines.py`'s xgboost leg (no target column, recursive multi-step
from the model's own forecasts). Kept as its own script so the single model can
be re-run / re-fitted in isolation without touching the reviewed baseline rows —
it writes rows under its own `split_id`.

Deterministic given the frozen split and the booster's fixed `random_state`, so
`seed=None` (no seed fabricated).
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_xgboost_experiment_v1"


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))
        model = XGBoostForecaster()
        t0 = time.perf_counter()
        model.fit(pd.DataFrame(index=train_y.index), train_y)
        model.observe(cal_y)
        preds = model.predict(pd.DataFrame(index=test_y.index))
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