"""Check that the TRAIN-fitted seasonal profile generalises to Calibration
and TEST, not only TRAIN.

The earlier check_seasonal_residuals.py only plots/reports TRAIN, which is
the same data the profile was fitted on -- a fitted profile naturally
performs best there. This script applies the same frozen (TRAIN-fitted)
profile to every split and compares raw versus residual seasonality for
each one, using:

    monthly-mean amplitude, before vs after
    weekday-mean spread, before vs after
    half-hour-mean spread, before vs after (half-hourly pipeline only)
    autocorrelation at each pipeline's seasonal lags, before vs after

Seasonal lags:
    daily:       7, 365
    half-hourly: 48, 336, 17520

The required pattern is strong raw autocorrelation collapsing to
substantially weaker residual autocorrelation -- not residual
autocorrelation forced to exactly zero.

This script does not call a detector and does not call record_run().

Also saves the printed summary as a table image (plus CSV twin), split
into a daily table and a half-hourly table -- combining them into one
would leave every daily row's half-hour/lag-48/336/17520 columns (and
every half-hourly row's lag-7/365 columns) empty, since those lags only
apply to one pipeline.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import (
    aggregate_daily_demand,
    apply_seasonal_profile,
    fit_seasonal_profile,
)
from drift_lab.config import REGIONS
from experiments.produce.table_image import save_table_image
from experiments.results_io import TABLES_DIR

TARGET_COLUMN = "TOTALDEMAND"

DAILY_LAGS = (7, 365)
HALF_HOURLY_LAGS = (48, 336, 17520)


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    ).sort_index()


def monthly_amplitude(series: pd.Series) -> float:
    monthly_means = series.groupby(series.index.month).mean()
    return float(monthly_means.max() - monthly_means.min())


def weekday_spread(series: pd.Series) -> float:
    weekday_means = series.groupby(series.index.dayofweek).mean()
    return float(weekday_means.max() - weekday_means.min())


def half_hour_spread(series: pd.Series) -> float:
    slot = series.index.hour * 2 + series.index.minute // 30
    slot_means = series.groupby(slot).mean()
    return float(slot_means.max() - slot_means.min())


def autocorrelations(series: pd.Series, lags: tuple[int, ...]) -> dict[int, float]:
    return {lag: series.autocorr(lag=lag) for lag in lags}


def compare(raw: pd.Series, residual: pd.Series, pipeline: str) -> dict:
    lags = DAILY_LAGS if pipeline == "daily" else HALF_HOURLY_LAGS

    result = {
        "monthly_amplitude_before": monthly_amplitude(raw),
        "monthly_amplitude_after": monthly_amplitude(residual),
        "weekday_spread_before": weekday_spread(raw),
        "weekday_spread_after": weekday_spread(residual),
    }

    if pipeline == "half_hourly":
        result["half_hour_spread_before"] = half_hour_spread(raw)
        result["half_hour_spread_after"] = half_hour_spread(residual)

    raw_autocorr = autocorrelations(raw, lags)
    residual_autocorr = autocorrelations(residual, lags)
    for lag in lags:
        result[f"autocorr_lag{lag}_before"] = raw_autocorr[lag]
        result[f"autocorr_lag{lag}_after"] = residual_autocorr[lag]

    return result


def main() -> None:
    rows = []

    for region in REGIONS:
        train, calibration, test = loader.load(region)

        train_s = demand_series(train)
        calibration_s = demand_series(calibration)
        test_s = demand_series(test)

        train_daily = aggregate_daily_demand(train_s)
        calibration_daily = aggregate_daily_demand(calibration_s)
        test_daily = aggregate_daily_demand(test_s)

        # Profiles are fit on TRAIN only, then reused unchanged below.
        daily_profile = fit_seasonal_profile(train_daily, frequency="daily")
        half_hourly_profile = fit_seasonal_profile(train_s, frequency="30min")

        pipelines = {
            "daily": (
                daily_profile,
                {"train": train_daily, "calibration": calibration_daily, "test": test_daily},
            ),
            "half_hourly": (
                half_hourly_profile,
                {"train": train_s, "calibration": calibration_s, "test": test_s},
            ),
        }

        for pipeline, (profile, splits) in pipelines.items():
            for split_name, raw in splits.items():
                residual = apply_seasonal_profile(raw, profile)
                metrics = compare(raw, residual, pipeline)
                rows.append(
                    {
                        "region": region,
                        "pipeline": pipeline,
                        "split": split_name,
                        **metrics,
                    }
                )

    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(summary.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    for pipeline, lags in (("daily", DAILY_LAGS), ("half_hourly", HALF_HOURLY_LAGS)):
        columns = ["region", "split", "monthly_amplitude_before", "monthly_amplitude_after"]
        columns += ["weekday_spread_before", "weekday_spread_after"]
        if pipeline == "half_hourly":
            columns += ["half_hour_spread_before", "half_hour_spread_after"]
        for lag in lags:
            columns += [f"autocorr_lag{lag}_before", f"autocorr_lag{lag}_after"]

        table = summary.loc[summary["pipeline"] == pipeline, columns].round(3).reset_index(drop=True)

        out_csv = TABLES_DIR / f"qc_residual_generalisation_{pipeline}.csv"
        out_png = TABLES_DIR / f"qc_residual_generalisation_{pipeline}.png"
        table.to_csv(out_csv, index=False)
        save_table_image(
            table,
            out_png,
            title=f"Residual generalisation check — {pipeline}",
            subtitle=(
                "TRAIN-fitted seasonal profile applied unchanged to TRAIN/Calibration/TEST; "
                "before vs after removing it. Required pattern: substantially weaker, not exactly zero."
            ),
        )
        print(f"\nwrote {out_csv}")
        print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
