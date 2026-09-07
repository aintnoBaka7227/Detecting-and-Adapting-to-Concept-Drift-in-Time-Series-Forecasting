"""NHITS baseline on AEMO — run separately from run_aemo_baselines.py.

Same frozen train/calibration/test split as the other three baselines
(config.SPLIT), but kept under its OWN split_id so these rows never
silently mix into the reviewed baseline comparison in runs.csv — because
the evaluation protocol below is not the same as theirs yet.

NHITS training is stochastic, so `seed` is the model's real `random_seed`,
not the NaN used for the deterministic baselines.

Weights are trained once and never change either way.

Two rolling-forecast protocols, toggled by the BLIND flag below:

BLOCK (default) — frozen, causal, non-overlapping blocks. The model rolls
through the test stream one `horizon`-sized block at a time (7 days): each
block is a single forward pass from the last `input_size` real points; the
forecasts are scored; then the block's *actual* demand is revealed and
appended to history for the next block. The model never consumes its own
forecasts, so nothing compounds. Real values only ever become context for
a *later* block, never used to change one already issued — not leakage.

BLIND — no target column; `roll_forecast` feeds the model its own forecasts.
Matches xgboost/dhr_arima's `predict_input()`, but a direct multi-step
neural model fed its own compounding output for years runs away to ~1e37 MW
then NaN by ~2022 (which makes `record_run` raise). Useful only over a
shorter window or to demonstrate the divergence.

Caveat for F1: BLOCK is a more forgiving condition than the fully blind
xgboost/dhr_arima — not an apples-to-apples baseline until the others move
to the same policy.
"""

from __future__ import annotations

import time

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.config import REGIONS
from drift_lab.forecasting.nhits_forecaster import NHITSForecaster
from experiments.run_harness import config_of, record_run

# --- rolling-forecast protocol: swap by commenting the pair you don't want ----
#
# BLOCK (default): every horizon-sized block forecasts from real context; the
#   block's actuals are revealed only after it's scored. Frozen, causal,
#   bounded. Real values become context for a *later* block, never used to
#   change an issued one -> not leakage.
# SPLIT_ID = "aemo_nhits_block7d_v1"
# BLIND = False
#
# BLIND: no target column -> roll_forecast feeds the model its own forecasts.
#   Same condition as xgboost/dhr_arima's blind predict_input(). WARNING: a
#   direct multi-step model fed its own compounding output for years runs away
#   -> ~1e37 MW then NaN by ~2022, which makes record_run() raise on the NaN.
#   Use only over a shorter test window, or to demonstrate the divergence.
SPLIT_ID = "aemo_nhits_blind_v1"
BLIND = True
# -----------------------------------------------------------------------------


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
        print(f"{region} nhits: wall_clock={wall_clock_s:.1f}s")


if __name__ == "__main__":
    main()
