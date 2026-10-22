"""NHITS frozen baseline on AEMO (F1), separate from run_aemo_baselines.py.

Fit on TRAIN once per config.SEEDS seed, observe the Calibration window, then
forecast TEST; weights never change. Built by run_aemo_adaptation_arms.make_model,
so it is the same model as arm A: 7 days of input (336) -> 1 day ahead (48),
500 steps. Its own split_id, because its protocol differs from the blind
xgboost / dhr_arima baselines.

Protocol, set by SPLIT_ID / BLIND below:
    BLOCK (default)  1-day blocks, each forecast from the last 7 days of real
                     demand; a block's actuals are revealed only after it is
                     scored, so nothing compounds and nothing leaks.
    BLIND            the model is fed its own forecasts, like xgboost /
                     dhr_arima; over years it diverges, so use it only over
                     short windows.
BLOCK is more forgiving than the blind xgboost / dhr_arima runs, so F1 is not
an apples-to-apples comparison. (The earlier 14-day -> 7-day setup is logged
under "aemo_nhits_block7d_v1"; those rows stay in runs.csv.)

Usage:
    python -m experiments.run.forecasting.run_aemo_nhits
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS, SEEDS
from experiments.run.adaptation.run_aemo_adaptation_arms import make_model
from experiments.run_harness import config_of, record_run

# Rolling-forecast protocol (see the module docstring); keep one pair.
# BLOCK (default):
SPLIT_ID = "aemo_nhits_block1d_v1"
BLIND = False
# BLIND (diverges over long windows):
# SPLIT_ID = "aemo_nhits_blind_v1"
# BLIND = True


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

        for seed in SEEDS:
            model = make_model("nhits", seed)
            t0 = time.perf_counter()
            model.fit(pd.DataFrame(index=train_y.index), train_y)
            model.observe(cal_y)

            if BLIND:
                predict_input = pd.DataFrame(index=test_y.index)
            else:
                predict_input = pd.DataFrame(
                    {"TOTALDEMAND": test_y.to_numpy()}, index=test_y.index
                )
            preds = model.predict(predict_input)
            wall_clock_s = time.perf_counter() - t0

            record_run(
                method=model.name,
                dataset="aemo",
                region=region,
                seed=model.random_seed,
                config=config_of(model),
                wall_clock_s=wall_clock_s,
                split_id=SPLIT_ID,
                train_samples=len(train_y),
                forecast=(test_y.to_numpy(), preds, test_y.index),
            )
            print(f"{region} nhits seed={seed}: wall_clock={wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()
