"""Step 5 / T3, T4, F3 -- the four adaptation arms on AEMO.

Arms (SHARED_DECISIONS.md 14, TEAM_DECISIONS.md 5.1), each an independent
run with its own model instance, adapter and retrain schedule:

    A  NeverRetrain              never retrain (control)
    B  RetrainEvery30Days        every 30 days, on the last 30 days (control)
    C  RetrainUsingFullHistory   on each accepted drift alarm, all history
    D  RetrainUsingRecentWindow  on each accepted drift alarm, the last
                                 30 / 60 / 120 / 180 days -- every window is
                                 run and logged; the official D is chosen
                                 later by lowest drift-regime MAE

Trigger: the accepted (post-refractory) change points written once by
run_adwin_changepoints.py -- detection is never re-run here.

Retraining boundary: for a trigger on day d the current model forecasts
through d 23:30, the adapter refits on data through d 23:30, and the new
model forecasts from d+1. Days are the calendar date of AEMO's timestamp.

Forecast protocol: each model's own F1 protocol, unchanged -- NHITS uses
the BLOCK protocol (real context between horizon-sized blocks) with the
shared settings (horizon 48, input 336, 500 steps); xgboost / dhr_arima
roll forward blind from their latest fit. A retrain is just `fit()`.

Regimes: documented-event regimes from `evaluation.assign_regime`, the
same labels for every arm. Cost fields per run: `n_retrains`,
`train_samples` (cumulative retrain samples) and `wall_clock_s`
(cumulative retrain fit time); the initial TRAIN fit is excluded.

Seeds: nhits and xgboost are stochastic, so `--seed` is required and must
come from the shared `config.SEEDS`; it is the model's `random_seed` /
`random_state`. dhr_arima is deterministic and takes no seed.

Logged under split_id "aemo_adapt_adwin_std_hh_v1", method "<model>/<arm>".

Usage:
    python -m experiments.run.adaptation.run_aemo_adaptation_arms --model nhits --region SA1 --seed 1
"""

from __future__ import annotations

import argparse
import logging
import time
import warnings
from collections.abc import Callable

import pandas as pd

from drift_lab.adaptation.never_retrain import NeverRetrain
from drift_lab.adaptation.retrain_accounting import RetrainAccounting
from drift_lab.adaptation.retrain_every_30_days import RetrainEvery30Days
from drift_lab.adaptation.retrain_using_full_history import RetrainUsingFullHistory
from drift_lab.adaptation.retrain_using_recent_window import (
    SWEEP_WINDOW_DAYS,
    RetrainUsingRecentWindow,
)
from drift_lab.aemo import loader
from drift_lab.config import DOCUMENTED_EVENTS_CSV, REGIONS, SEEDS
from drift_lab.evaluation import assign_regime, build_event_windows
from drift_lab.forecasting.base import Forecaster
from experiments.run.detection.run_adwin_changepoints import (
    DETECTOR as TRIGGER_DETECTOR,
)
from experiments.run.detection.run_adwin_changepoints import (
    changepoints_path,
    load_changepoints,
)
from experiments.run_harness import config_of, record_run

SPLIT_ID = "aemo_adapt_adwin_std_hh_v1"
TARGET_COLUMN = "TOTALDEMAND"
MODELS = ("nhits", "xgboost", "dhr_arima")
SEEDED_MODELS = ("nhits", "xgboost")

# Shared NHITS settings (SHARED_DECISIONS.md 14) -- constructor arguments.
NHITS_SHARED_SETTINGS = {"horizon": 48, "input_size": 336, "max_steps": 500}

ARM_A = "A_never_retrain"
ARM_B = "B_every_30_days"
ARM_C = "C_drift_full_history"


def arm_d(window_days: int) -> str:
    return f"D_drift_recent_{window_days}d"


ARMS = (ARM_A, ARM_B, ARM_C, *(arm_d(w) for w in SWEEP_WINDOW_DAYS))


def make_adapter(arm: str) -> RetrainAccounting:
    if arm == ARM_A:
        return NeverRetrain(target_column=TARGET_COLUMN)
    if arm == ARM_B:
        return RetrainEvery30Days(target_column=TARGET_COLUMN)
    if arm == ARM_C:
        return RetrainUsingFullHistory(target_column=TARGET_COLUMN)
    for window_days in SWEEP_WINDOW_DAYS:
        if arm == arm_d(window_days):
            return RetrainUsingRecentWindow(window_days, target_column=TARGET_COLUMN)
    raise ValueError(f"unknown arm {arm!r}")


def make_model(model_name: str, seed: int | None) -> Forecaster:
    if model_name == "nhits":
        from drift_lab.forecasting.nhits_forecaster import NHITSForecaster

        return NHITSForecaster(**NHITS_SHARED_SETTINGS, random_seed=seed)
    if model_name == "xgboost":
        from drift_lab.forecasting.xgboost_forecaster import XGBoostForecaster

        return XGBoostForecaster(random_state=seed)
    if model_name == "dhr_arima":
        from drift_lab.forecasting.dhr_arima import DHRArima

        return DHRArima()
    raise ValueError(f"unknown model {model_name!r}")


def predict_input(model_name: str, segment_y: pd.Series) -> pd.DataFrame:
    """Each model's F1 protocol: NHITS gets real context between blocks;
    xgboost / dhr_arima get no target column (blind roll-forward)."""
    if model_name == "nhits":
        return pd.DataFrame({TARGET_COLUMN: segment_y.to_numpy()}, index=segment_y.index)
    return pd.DataFrame(index=segment_y.index)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    )


def regime_labels(test_index: pd.DatetimeIndex, region: str) -> pd.Series:
    """Documented-event regime for every TEST timestamp (shared by all arms)."""
    events = pd.read_csv(DOCUMENTED_EVENTS_CSV, parse_dates=["start_date", "end_date"])
    labelled = assign_regime(test_index, build_event_windows(events, region))
    return labelled.set_index("timestamp")["regime"].reindex(test_index)


def run_arm(
    model: Forecaster,
    adapter: RetrainAccounting,
    trigger_days: list[pd.Timestamp],
    history_y: pd.Series,
    test_y: pd.Series,
    make_input: Callable[[pd.Series], pd.DataFrame],
    on_retrain: Callable[[pd.Timestamp, RetrainAccounting], None] | None = None,
) -> pd.Series:
    """Roll a fitted `model` through `test_y`, handing it to `adapter` at the
    end of each trigger day d: the segment through d 23:30 is forecast by
    the current model, `adapter.adapt` sees every observation through
    d 23:30, and the returned model forecasts from d+1 00:00."""
    test_index = pd.DatetimeIndex(test_y.index)
    last_day = test_index[-1].normalize()
    boundaries = [day for day in sorted(set(trigger_days)) if day < last_day]

    forecasts: list[pd.Series] = []
    start = 0
    for day in boundaries:
        end = int(test_index.searchsorted(day + pd.Timedelta(days=1), side="left"))
        if end <= start:
            continue
        segment = test_y.iloc[start:end]
        forecasts.append(pd.Series(model.predict(make_input(segment)), index=segment.index))

        observed = pd.concat([history_y, test_y.iloc[:end]]).to_frame(TARGET_COLUMN)
        model = adapter.adapt([len(observed) - 1], model, observed)
        if on_retrain is not None:
            on_retrain(day, adapter)
        start = end

    if start < len(test_y):
        segment = test_y.iloc[start:]
        forecasts.append(pd.Series(model.predict(make_input(segment)), index=segment.index))

    return pd.concat(forecasts).reindex(test_index)


def trigger_days_for(
    arm: str, adapter: RetrainAccounting, alarm_days: list[pd.Timestamp], test_y: pd.Series
) -> list[pd.Timestamp]:
    if arm == ARM_A:
        return []
    if arm == ARM_B:
        return adapter.schedule(test_y.index[0], test_y.index[-1])
    return alarm_days


def run_one(model_name: str, region: str, seed: int | None, arms: list[str]) -> None:
    train, calibration, test = loader.load(region)
    train_y, cal_y, test_y = (demand_series(f) for f in (train, calibration, test))
    history_y = pd.concat([train_y, cal_y]).sort_index()

    changepoints = load_changepoints(region)
    alarm_days = sorted(set(pd.DatetimeIndex(changepoints["day"])))
    labels = regime_labels(pd.DatetimeIndex(test_y.index), region)
    print(
        f"[{model_name} {region} seed={seed}] {len(alarm_days)} accepted {TRIGGER_DETECTOR} "
        f"change-point days; regimes {labels.value_counts().to_dict()}",
        flush=True,
    )

    for arm in arms:
        adapter = make_adapter(arm)
        days = trigger_days_for(arm, adapter, alarm_days, test_y)
        n_days = len(days)

        t0 = time.perf_counter()
        model = make_model(model_name, seed)
        model.fit(pd.DataFrame(index=train_y.index), train_y)
        model.observe(cal_y)

        def log_retrain(day, adapter_, arm_=arm, t0_=t0, n_days_=n_days):
            print(
                f"  {arm_}: retrain {adapter_.retrain_count}/{n_days_} at {day.date()} "
                f"(elapsed {time.perf_counter() - t0_:.0f}s)",
                flush=True,
            )

        y_pred = run_arm(
            model,
            adapter,
            days,
            history_y,
            test_y,
            lambda segment: predict_input(model_name, segment),
            on_retrain=log_retrain,
        )

        config = {
            **config_of(model),
            "model": model_name,
            "arm": arm,
            "adapter": adapter.name,
            **{f"adapter_{k}": v for k, v in config_of(adapter).items()},
            "retraining_boundary": "end_of_trigger_day",
            "trigger": (
                f"{TRIGGER_DETECTOR}_accepted:{changepoints_path(region).name}"
                if arm not in (ARM_A, ARM_B)
                else "none"
            ),
        }
        record_run(
            method=f"{model_name}/{arm}",
            dataset="aemo",
            region=region,
            seed=seed,
            config=config,
            wall_clock_s=adapter.fit_seconds,
            split_id=SPLIT_ID,
            train_samples=adapter.train_samples,
            n_retrains=adapter.retrain_count,
            forecast=(test_y.to_numpy(), y_pred.to_numpy(), test_y.index),
            regime_labels=labels.to_numpy(),
            retrain_timestamps=adapter.retrain_timestamps,
        )
        print(
            f"[{model_name} {region} seed={seed}] {arm}: {adapter.retrain_count} retrains, "
            f"retrain fit {adapter.fit_seconds:.0f}s, total {time.perf_counter() - t0:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the four adaptation arms on AEMO.")
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--region", choices=REGIONS, required=True)
    parser.add_argument(
        "--seed", type=int, choices=SEEDS, default=None, help="model seed (nhits / xgboost)"
    )
    parser.add_argument("--arms", default=",".join(ARMS), help=f"comma list from {ARMS}")
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        parser.error(f"unknown arms {unknown}")

    if args.model in SEEDED_MODELS:
        if args.seed is None:
            parser.error(f"--seed is required for {args.model}")
    elif args.seed is not None:
        parser.error(f"{args.model} is deterministic; --seed applies to {SEEDED_MODELS} only")

    if args.model == "nhits":
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        logging.getLogger("lightning").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        if args.torch_threads:
            import torch

            torch.set_num_threads(args.torch_threads)

    run_one(args.model, args.region, args.seed, arms)


if __name__ == "__main__":
    main()
