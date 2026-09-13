"""NHITS + KSWIN + RetrainUsing3MonthWindows adaptation arm -- NSW1 only.

Smoke test, not a reviewed comparison. KSWIN watches
`daily_aggregate(test)` (raw half-hourly demand fires on ordinary
seasonality, and every firing here costs a full retrain -- see
DECISIONS.md). Detection runs once, fully upfront (not leakage: an online
detector's flag at day i only depends on data up to i).

Same BLOCK rolling protocol as `run_aemo_nhits.py`, except a block is cut
short exactly at a changepoint's day instead of always running the full
7 days -- the model retrains there (3-month window ending at that day),
then resumes a normal 7-day cadence from the next day.

`run_aemo_nhits_retrain3mo_kswin_sa.py` is SA1's twin (duplicated per this
project's `run_*.py` convention). `split_id="aemo_nhits_retrain3mo_kswin_v1"`
keeps these rows separate from `run_aemo_nhits.py`'s.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.adaptation.retrain_using_3_month_windows import (
    RetrainUsing3MonthWindows,
)
from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import daily_aggregate
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.forecasting.nhits_forecaster import NHITSForecaster
from experiments.run_harness import config_of, record_run

REGION = "NSW1"
SPLIT_ID = "aemo_nhits_retrain3mo_kswin_v1"
TARGET_COLUMN = "TOTALDEMAND"


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    )


def main() -> None:
    train, calibration, test = loader.load(REGION)
    train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))

    model = NHITSForecaster()
    adapter = RetrainUsing3MonthWindows(target_column=TARGET_COLUMN)

    t0 = time.perf_counter()
    model.fit(pd.DataFrame(index=train_y.index), train_y)
    model.observe(cal_y)

    # History fed to the adapter -- grows as test blocks are revealed.
    history = pd.concat([train_y, cal_y]).sort_index().to_frame()

    # Detect changepoints upfront on the daily-aggregated stream (see docstring).
    detector = KSWINDetector()
    daily_test = daily_aggregate(test, column=TARGET_COLUMN)
    flagged_days = detector.detect(daily_test.to_numpy())
    changepoint_days = sorted(daily_test.index[i] for i in flagged_days)
    print(
        f"{REGION}: kswin flagged {len(changepoint_days)} day(s) "
        f"on the daily-aggregated stream"
    )

    # Last half-hourly position of each flagged day -- a block ending here
    # gets cut short instead of running the full horizon.
    last_position_of_day = pd.Series(
        range(len(test_y)), index=test_y.index.normalize()
    ).groupby(level=0).max()
    cut_positions = sorted(
        int(last_position_of_day[day])
        for day in changepoint_days
        if day in last_position_of_day.index
    )
    cut_iter = iter(cut_positions)
    next_cut = next(cut_iter, None)

    horizon = model.horizon
    n = len(test_y.index)
    preds: list[pd.Series] = []
    pos = 0
    while pos < n:
        full_block_end = pos + horizon - 1
        cut_here = next_cut is not None and next_cut <= full_block_end
        end = next_cut if cut_here else min(full_block_end, n - 1)

        block_index = test_y.index[pos : end + 1]
        block_preds = model.predict(pd.DataFrame(index=block_index))
        preds.append(pd.Series(block_preds, index=block_index))

        block_actual = test_y.loc[block_index]
        model.observe(block_actual)
        history = pd.concat([history, block_actual.to_frame()]).sort_index()

        if cut_here:
            # history's last row is the changepoint day itself -- no gap to fill.
            model = adapter.adapt([len(history) - 1], model, history)
            next_cut = next(cut_iter, None)

        pos = end + 1

    wall_clock_s = time.perf_counter() - t0
    y_pred = pd.concat(preds).reindex(test_y.index).to_numpy()

    config = {
        **config_of(model),
        "adapter": adapter.name,
        **{f"adapter_{k}": v for k, v in config_of(adapter).items()},
    }
    record_run(
        method=model.name,
        dataset="aemo",
        region=REGION,
        seed=model.random_seed,
        config=config,
        wall_clock_s=wall_clock_s,
        split_id=SPLIT_ID,
        train_samples=len(train_y),
        n_retrains=adapter.retrain_count,
        forecast=(test_y.to_numpy(), y_pred, test_y.index),
        changepoints=cut_positions or None,
    )
    print(
        f"{REGION} nhits+{adapter.name}: {adapter.retrain_count} retrain(s), "
        f"wall_clock={wall_clock_s:.1f}s"
    )


if __name__ == "__main__":
    main()
