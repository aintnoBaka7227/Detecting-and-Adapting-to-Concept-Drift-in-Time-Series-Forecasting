"""Check that the TRAIN-fitted seasonal profile generalises beyond TRAIN.

Applies the frozen half-hourly profile to TRAIN, Calibration and TEST and
compares raw vs residual monthly amplitude, weekday spread, half-hour spread
and autocorrelation at lags 48, 336 and 17520. Required pattern: strong raw
autocorrelation collapsing to much weaker (not zero) residual
autocorrelation. No detector, no runs.csv; writes a table image and CSV. The
daily pipeline was retired with the standard-daily T2.
"""

from __future__ import annotations

import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.deseasonalise import apply_seasonal_profile, fit_seasonal_profile
from drift_lab.config import REGIONS
from experiments.produce.shared.table_image import save_table_image
from experiments.results_io import TABLES_DIR

TARGET_COLUMN = "TOTALDEMAND"

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


def compare(raw: pd.Series, residual: pd.Series) -> dict:
    result = {
        "monthly_amplitude_before": monthly_amplitude(raw),
        "monthly_amplitude_after": monthly_amplitude(residual),
        "weekday_spread_before": weekday_spread(raw),
        "weekday_spread_after": weekday_spread(residual),
        "half_hour_spread_before": half_hour_spread(raw),
        "half_hour_spread_after": half_hour_spread(residual),
    }

    raw_autocorr = autocorrelations(raw, HALF_HOURLY_LAGS)
    residual_autocorr = autocorrelations(residual, HALF_HOURLY_LAGS)
    for lag in HALF_HOURLY_LAGS:
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

        # The profile is fit on TRAIN only, then reused unchanged below.
        profile = fit_seasonal_profile(train_s, frequency="30min")

        for split_name, raw in {"train": train_s, "calibration": calibration_s, "test": test_s}.items():
            residual = apply_seasonal_profile(raw, profile)
            rows.append({"region": region, "split": split_name, **compare(raw, residual)})

    summary = pd.DataFrame(rows)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print(summary.to_string(index=False))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table = summary.round(3)
    out_csv = TABLES_DIR / "qc_residual_generalisation_half_hourly.csv"
    out_png = TABLES_DIR / "qc_residual_generalisation_half_hourly.png"
    table.to_csv(out_csv, index=False)
    save_table_image(
        table,
        out_png,
        title="Residual generalisation check — half_hourly",
        subtitle=(
            "TRAIN-fitted seasonal profile applied unchanged to TRAIN/Calibration/TEST; "
            "before vs after removing it. Required pattern: substantially weaker, not exactly zero."
        ),
    )
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
