"""Seasonal-naive baseline on AEMO, run separately from run_aemo_baselines.py.

Uses the same frozen train/calibration/test split and walk-forward protocol
as the seasonal-naive entry in the combined baseline runner.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.seasonal_naive_aditya import SeasonalNaive
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_frozen_v1_aditya"


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(frame) for frame in (train, calibration, test))

        # The held-out 2019 validation window favored a guarded blend for
        # NSW1's rolling-error tail; SA1 keeps the day-only baseline.
        model = SeasonalNaive(
            day_weight=0.5 if region == "NSW1" else 1.0,
            max_blend_difference=705.2 if region == "NSW1" else None,
        )
        t0 = time.perf_counter()
        model.fit(pd.DataFrame(index=train_y.index), train_y)
        model.observe(cal_y)
        predict_input = pd.DataFrame(
            {"TOTALDEMAND": test_y.to_numpy()}, index=test_y.index
        )
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
        print(f"{region} {model.name}: wall_clock={wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()