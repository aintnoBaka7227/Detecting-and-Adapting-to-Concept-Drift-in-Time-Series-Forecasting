"""Bingqi independent DHR-ARIMA implementation on AEMO.

Uses the same frozen AEMO train/calibration/test protocol as the shared
baseline runner so the result can be cross-checked fairly.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.bingqi.dhr_arima import BingqiDHRArima
from experiments.run_harness import config_of, record_run


SPLIT_ID = "aemo_frozen_v1"


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)

        train_y = demand_series(train)
        cal_y = demand_series(calibration)
        test_y = demand_series(test)

        # Fresh independent model for every region
        model = BingqiDHRArima()

        t0 = time.perf_counter()

        model.fit(
            pd.DataFrame(index=train_y.index),
            train_y,
        )

        # Calibration values are already known before the test period
        model.observe(cal_y)

        # Same blind forecasting condition as the shared DHR-ARIMA baseline
        predict_input = pd.DataFrame(index=test_y.index)
        preds = model.predict(predict_input)

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

        print(
            f"{region} {model.name}: "
            f"wall_clock={wall_clock_s:.1f}s"
        )


if __name__ == "__main__":
    main()