"""Run the independent Aditya NHITS forecaster on the AEMO data.

This uses the same frozen train/calibration/test protocol as the existing
NHITS runner, but records a distinct split ID and method name. It also emits
residual-based detection rows so the shared Figure F2 AEMO detector pipeline can
consume the same model output without duplicating plotting logic.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS, SPLIT
from drift_lab.forecasting.nhits_forecaster_aditya import NHITSForecaster
from experiments import results_io
from experiments.run.detection.detection_artifacts import match_and_persist
from experiments.run_harness import config_of, record_run

# Each horizon-sized block is forecast from real context, then its actuals are
# revealed before the next block. The model weights remain frozen.
SPLIT_ID = "aemo_nhits_aditya_block7d_v1"
BLIND = False
TEST_START = pd.Timestamp(SPLIT["test"][0])


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame["TOTALDEMAND"].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name="TOTALDEMAND",
    )


def region_events(region: str) -> pd.DataFrame:
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    return events[events["region"].isin([region, "NEM"])].reset_index(drop=True)


def residual_detections(test_y: pd.Series, predictions: np.ndarray) -> list[pd.Timestamp]:
    residual = pd.Series(np.abs(np.asarray(test_y.to_numpy(), dtype=float) - np.asarray(predictions, dtype=float)), index=test_y.index)
    rolling = residual.rolling(window=48 * 7, min_periods=48).mean()
    threshold = max(float(rolling.quantile(0.90)), float(rolling.mean() + 1.5 * rolling.std(ddof=0)))
    flagged = rolling >= threshold
    return [ts for ts in residual.index[flagged & (residual.index >= TEST_START)]]


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(frame) for frame in (train, calibration, test))

        model = NHITSForecaster()
        t0 = time.perf_counter()
        model.fit(pd.DataFrame(index=train_y.index), train_y)
        model.observe(cal_y)

        if BLIND:
            predict_input = pd.DataFrame(index=test_y.index)
        else:
            predict_input = pd.DataFrame(
                {"TOTALDEMAND": test_y.to_numpy()}, index=test_y.index
            )
        predictions = model.predict(predict_input)
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
            forecast=(test_y.to_numpy(), predictions, test_y.index),
        )

        config = {**config_of(model), "detection_rule": "rolling_abs_error_7d_90pct_plus_1.5std"}
        config_hash = results_io.config_hash(config)
        events = region_events(region)
        test_detections = residual_detections(test_y, predictions)
        if test_detections:
            match_results, _ = match_and_persist(config_hash, region, test_detections, events)
            n_accepted = int((match_results["label"] != "Ignored").sum())
        else:
            n_accepted = 0

        record_run(
            method=model.name,
            dataset="aemo",
            region=region,
            seed=model.random_seed,
            config=config,
            wall_clock_s=wall_clock_s,
            split_id=SPLIT_ID,
            train_samples=len(train_y),
            detection=(test_detections, events, None),
            test_period_days=(test_y.index.max() - TEST_START) / pd.Timedelta(days=1),
        )
        print(f"{region} nhits_aditya: wall_clock={wall_clock_s:.1f}s, residual detections={len(test_detections)}, accepted={n_accepted}")


if __name__ == "__main__":
    main()
