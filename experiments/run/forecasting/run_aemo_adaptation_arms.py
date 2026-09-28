"""Arm A (never retrain) and Arm B (retrain every 30 days) -- all three
frozen baseline models (seasonal-naive, XGBoost, DHR-ARIMA), on AEMO, both
regions. Step 5's four-arm comparison (TEAM_DECISIONS.md 5.1); Arms C/D are
separate runners, not here.

Generalising across models
---------------------------
The `Adapter` contract (`adapt(changepoints, model, data) -> model`) is
model-agnostic by design -- `NeverRetrain` and `ScheduledRetrain` never
needed changing to run against any of the three `Forecaster`
implementations. What *did* need care is how each model wants to be
called, since they aren't all stateless in the same way:

- `SeasonalNaive` and `XGBoostForecaster` both override `observe()` to
  extend an internal rolling history (see their own docstrings) --
  `XGBoostForecaster.predict()` gets no target column in `X` at all and
  relies entirely on that internal history / its own earlier forecasts
  to resolve lags; `SeasonalNaive.predict()` instead reads real lag
  values straight out of `X`'s target column when present (see
  `predict_input()` below, same split `run_aemo_baselines.py` already
  makes).
- `DHRArima` does not override `observe()` (it's a no-op) and its
  `predict()` always forecasts the next N steps from wherever it was
  last *fit*, not from wherever an earlier `predict()` call left off.

Arm A needs no block loop for any of the three models: fit once on
TRAIN, `observe()` calibration, then one `predict()` call over the whole
TEST stream -- exactly `run_aemo_baselines.py`'s frozen-baseline pattern.
`NeverRetrain.adapt()` is still called once afterwards purely so
`retrain_count == 0` is on record for the arm comparison.

Arm B walks forward in `BLOCK_DAYS`-day blocks: predict the next block,
call `model.observe(block_actual)` unconditionally (a no-op for
`DHRArima`, essential for `XGBoostForecaster` since it has no other way
to learn this block's actuals between refits, and redundant-but-harmless
for `SeasonalNaive` since it already receives those same values through
`predict_input`), then let `ScheduledRetrain` decide whether this block
is due for a refit on the full history so far.

Retraining boundary (TEAM_DECISIONS.md 5.2): for a block ending on day
`d`, the old model forecasts through `d 23:30`, the refit uses data
through day `d`, and the new model's next forecast starts `d + 1` --
which is exactly what falls out of looping over non-overlapping,
contiguous `BLOCK_DAYS`-day calendar blocks.

`wall_clock_s` here follows the existing repo convention (total time,
including the initial TRAIN fit -- same as `run_aemo_baselines.py` and
`run_aemo_nhits_retrain3mo_kswin_*.py`), NOT the retrain-only convention
TEAM_DECISIONS.md 5.4 proposes -- that item is listed there as a
recommendation still to be frozen, not yet adopted by any runner in this
repo. Switch this (and the other runners) together if/when the team
signs off on it, rather than having one runner alone measure it
differently.

`split_id="aemo_adaptation_arms_v1"` keeps these rows separate from
`run_aemo_baselines.py`'s `aemo_frozen_v1` rows -- Arm A here duplicates
those rows' forecast numbers under a comparable arm/adapter config so
each model's Arm A shows up next to its Arm B in the same split.

Runtime note: DHR-ARIMA is by far the slowest of the three here (a full
SARIMAX refit per block) -- expect its Arm B to take hours per region,
same as before. Seasonal-naive is near-instant (no real fitting) and
XGBoost is moderate. Running all three models x both arms x both regions
in one `python -m ...` call means DHR-ARIMA's cost dominates the total
wall-clock time even though the other two models finish quickly.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.adaptation.never_retrain import NeverRetrain
from drift_lab.adaptation.scheduled_retrain import ScheduledRetrain
from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.base import Forecaster
from drift_lab.forecasting.dhr_arima import DHRArima
from drift_lab.forecasting.seasonal_naive import SeasonalNaive
from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_adaptation_arms_v1"
TARGET_COLUMN = "TOTALDEMAND"
BLOCK_DAYS = 30
STEPS_PER_DAY = 48  # native AEMO half-hourly interval

# Fresh instance per (model, region, arm) -- fit() fully replaces prior
# state on all three (see forecasting/*.py), but a callable-per-run avoids
# any doubt about state bleeding across the many runs main() makes here,
# and matches this project's "one instance per model" convention from
# run_aemo_baselines.py without sharing that one instance six ways.
MODEL_FACTORIES: tuple[type[Forecaster], ...] = (
    SeasonalNaive,
    XGBoostForecaster,
    DHRArima,
)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    )


def predict_input(model: Forecaster, y: pd.Series) -> pd.DataFrame:
    """seasonal_naive needs the real lag values to walk forward with (it
    reads them straight out of X's target column -- see its docstring);
    xgboost / dhr_arima are frozen for predict() -- no target column, no
    feedback through X. Same split as run_aemo_baselines.py's helper."""
    if model.name == "seasonal_naive":
        return pd.DataFrame({TARGET_COLUMN: y.to_numpy()}, index=y.index)
    return pd.DataFrame(index=y.index)


def run_arm_a(
    model_factory: type[Forecaster],
    region: str,
    train_y: pd.Series,
    cal_y: pd.Series,
    test_y: pd.Series,
) -> None:
    """Never retrain: one fit on TRAIN, one predict through the whole TEST
    stream -- no block loop needed for any model, since nothing here ever
    retrains partway through."""
    adapter = NeverRetrain()
    model = model_factory()

    t0 = time.perf_counter()
    model.fit(pd.DataFrame(index=train_y.index), train_y)
    model.observe(cal_y)
    preds = model.predict(predict_input(model, test_y))
    wall_clock_s = time.perf_counter() - t0

    full_history = pd.concat([train_y, cal_y, test_y]).sort_index().to_frame()
    model = adapter.adapt([], model, full_history)  # records retrain_count == 0

    config = {
        **config_of(model),
        "adapter": adapter.name,
        **{f"adapter_{k}": v for k, v in config_of(adapter).items()},
    }
    record_run(
        method=f"{model.name}+{adapter.name}",
        dataset="aemo",
        region=region,
        seed=None,
        config=config,
        wall_clock_s=wall_clock_s,
        split_id=SPLIT_ID,
        train_samples=len(train_y),
        n_retrains=adapter.retrain_count,
        forecast=(test_y.to_numpy(), preds, test_y.index),
    )
    print(
        f"{region} {model.name} arm A ({adapter.name}): "
        f"{adapter.retrain_count} retrain(s), wall_clock={wall_clock_s:.1f}s"
    )


def run_arm_b(
    model_factory: type[Forecaster],
    region: str,
    train_y: pd.Series,
    cal_y: pd.Series,
    test_y: pd.Series,
) -> None:
    """Retrain every BLOCK_DAYS days, on all history seen so far."""
    adapter = ScheduledRetrain(interval_days=BLOCK_DAYS, target_column=TARGET_COLUMN)
    model = model_factory()

    t0 = time.perf_counter()
    model.fit(pd.DataFrame(index=train_y.index), train_y)
    model.observe(cal_y)

    # History fed to the adapter -- grows as test blocks are revealed.
    history = pd.concat([train_y, cal_y]).sort_index().to_frame()

    horizon = BLOCK_DAYS * STEPS_PER_DAY
    n = len(test_y)
    n_blocks = -(-n // horizon)  # ceil division, for progress display only
    preds: list[pd.Series] = []
    pos = 0
    block_num = 0
    while pos < n:
        block_num += 1
        end = min(pos + horizon - 1, n - 1)
        block_index = test_y.index[pos : end + 1]
        block_y = test_y.loc[block_index]

        block_t0 = time.perf_counter()
        block_preds = model.predict(predict_input(model, block_y))
        preds.append(pd.Series(block_preds, index=block_index))

        # Unconditional: a no-op for DHRArima, essential for
        # XGBoostForecaster (its only way to learn this block's actuals
        # between refits), redundant-but-harmless for SeasonalNaive
        # (already fed through predict_input above).
        model.observe(block_y)

        history = pd.concat([history, block_y.to_frame()]).sort_index()

        # Fixed schedule -- adapt() ignores changepoints entirely and
        # refits on the full history up to and including this block.
        retrains_before = adapter.retrain_count
        model = adapter.adapt([], model, history)
        retrained = adapter.retrain_count > retrains_before
        block_wall_s = time.perf_counter() - block_t0

        print(
            f"  {region} {model.name} arm B block {block_num}/{n_blocks} "
            f"({block_index[0].date()} to {block_index[-1].date()}): "
            f"{'retrained' if retrained else 'no retrain (schedule not due)'}, "
            f"block_wall={block_wall_s:.1f}s, "
            f"history_rows={len(history)}",
            flush=True,
        )

        pos = end + 1

    wall_clock_s = time.perf_counter() - t0
    y_pred = pd.concat(preds).reindex(test_y.index).to_numpy()

    config = {
        **config_of(model),
        "adapter": adapter.name,
        **{f"adapter_{k}": v for k, v in config_of(adapter).items()},
    }
    record_run(
        method=f"{model.name}+{adapter.name}",
        dataset="aemo",
        region=region,
        seed=None,
        config=config,
        wall_clock_s=wall_clock_s,
        split_id=SPLIT_ID,
        train_samples=len(train_y),
        n_retrains=adapter.retrain_count,
        forecast=(test_y.to_numpy(), y_pred, test_y.index),
    )
    print(
        f"{region} {model.name} arm B ({adapter.name}): "
        f"{adapter.retrain_count} retrain(s), wall_clock={wall_clock_s:.1f}s"
    )


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)
        train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))

        for model_factory in MODEL_FACTORIES:
            run_arm_a(model_factory, region, train_y, cal_y, test_y)
            run_arm_b(model_factory, region, train_y, cal_y, test_y)


if __name__ == "__main__":
    main()
