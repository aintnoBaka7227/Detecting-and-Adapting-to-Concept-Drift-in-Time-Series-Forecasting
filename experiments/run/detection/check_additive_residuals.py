"""QC gate for the "additive" AEMO input stream -- second, independent
deseasonalisation approach, kept fully separate from
check_seasonal_residuals.py.

Step 1: produce half-hourly residuals for every split (TRAIN,
Calibration, TEST) x region, using
drift_lab.aemo.additive_deseasonalise.remove_daily_weekly_profile(),
fit on TRAIN only and applied unchanged to all three splits (see
additive_stream_common.py).

Step 2: plot TRAIN demand, expected seasonal demand, residual demand,
and residual broken down by day-of-year / weekday / half-hour, and print
the associated summary numbers, so deseasonalisation quality can be
checked before any detector runs on the residuals.

Step 3: standardise using the TRAIN residual's own mean and standard
deviation (ddof=0), applied unchanged to Calibration and TEST.

half-hourly only -- see additive_stream_common.py for why there is no
daily counterpart. This script does not call a detector and does not
call record_run().

Output filenames are prefixed "qc_additive_" so they never collide with
check_seasonal_residuals.py's "qc_seasonal_"/"qc_standardised_" files.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from drift_lab.aemo import loader
from drift_lab.aemo.additive_deseasonalise import remove_daily_weekly_profile
from drift_lab.config import REGIONS
from experiments.results_io import FIGURES_DIR

TARGET_COLUMN = "TOTALDEMAND"


def demand_series(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        frame[TARGET_COLUMN].to_numpy(dtype=float),
        index=pd.DatetimeIndex(frame["SETTLEMENTDATE"]),
        name=TARGET_COLUMN,
    ).sort_index()


def plot_and_report(region: str, train: pd.Series, train_residual: pd.Series) -> None:
    """Step 2 checks for one region's TRAIN series."""

    expected = train - train_residual.to_numpy()

    fig, axes = plt.subplots(5, 1, figsize=(12, 3.2 * 5))

    axes[0].plot(train.index, train.to_numpy(), label="original TRAIN demand", linewidth=0.8)
    axes[0].plot(expected.index, expected.to_numpy(), label="expected additive seasonal demand", linewidth=0.8)
    axes[0].set_title(f"{region} additive half_hourly: original vs expected seasonal demand")
    axes[0].legend()

    axes[1].plot(train_residual.index, train_residual.to_numpy(), linewidth=0.6, color="tab:green")
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].set_title(f"{region} additive half_hourly: residual demand")

    by_day_of_year = (
        pd.Series(train_residual.to_numpy(), index=train_residual.index)
        .groupby(train_residual.index.dayofyear)
        .mean()
    )
    axes[2].plot(by_day_of_year.index, by_day_of_year.to_numpy(), linewidth=0.8, color="tab:orange")
    axes[2].axhline(0.0, color="black", linewidth=0.6)
    axes[2].set_title(f"{region} additive half_hourly: mean residual by day of year")
    axes[2].set_xlabel("day of year")

    by_weekday = train_residual.groupby(train_residual.index.dayofweek).mean()
    axes[3].bar(by_weekday.index, by_weekday.to_numpy(), color="tab:purple")
    axes[3].axhline(0.0, color="black", linewidth=0.6)
    axes[3].set_title(f"{region} additive half_hourly: mean residual by weekday")
    axes[3].set_xlabel("weekday (0=Mon .. 6=Sun)")

    by_half_hour = train_residual.groupby(
        train_residual.index.hour * 2 + train_residual.index.minute // 30
    ).mean()
    axes[4].plot(by_half_hour.index, by_half_hour.to_numpy(), linewidth=0.8, color="tab:red")
    axes[4].axhline(0.0, color="black", linewidth=0.6)
    axes[4].set_title(f"{region} additive half_hourly: mean residual by half-hour slot")
    axes[4].set_xlabel("half-hour slot (0-47)")

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"qc_additive_residual_{region}_half_hourly.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    seasonal_amplitude_before = float(
        train.groupby(train.index.month).mean().max() - train.groupby(train.index.month).mean().min()
    )
    seasonal_amplitude_after = float(
        train_residual.groupby(train_residual.index.month).mean().max()
        - train_residual.groupby(train_residual.index.month).mean().min()
    )
    weekday_spread = float(by_weekday.max() - by_weekday.min())
    half_hour_spread = float(by_half_hour.max() - by_half_hour.min())
    residual_mean = float(train_residual.mean())
    residual_std = float(train_residual.std())

    print(f"\n{region} additive half_hourly:")
    print(f"  figure saved to {out_path}")
    print(f"  monthly-mean amplitude: original={seasonal_amplitude_before:.1f}  residual={seasonal_amplitude_after:.1f}")
    print(f"  weekday-mean spread (residual): {weekday_spread:.2f}")
    print(f"  half-hour-mean spread (residual): {half_hour_spread:.2f}")
    print(f"  residual mean={residual_mean:.4f}  std={residual_std:.2f}")


def standardise(
    train_residual: pd.Series,
    calibration_residual: pd.Series,
    test_residual: pd.Series,
    label: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Step 3: z-score using TRAIN residual mean/std only."""

    residual_mean = train_residual.mean()
    residual_std = train_residual.std(ddof=0)

    train_z = (train_residual - residual_mean) / residual_std
    calibration_z = (calibration_residual - residual_mean) / residual_std
    test_z = (test_residual - residual_mean) / residual_std

    print(f"\n{label} standardisation:")
    print(f"  residual_mean={residual_mean:.4f}  residual_std={residual_std:.4f}")
    print(f"  train_z:       mean={train_z.mean():.4f}  std={train_z.std(ddof=0):.4f}")
    print(f"  calibration_z: mean={calibration_z.mean():.4f}  std={calibration_z.std(ddof=0):.4f}")
    print(f"  test_z:        mean={test_z.mean():.4f}  std={test_z.std(ddof=0):.4f}")

    return train_z, calibration_z, test_z


def plot_standardised(
    region: str,
    train_demand: pd.Series,
    calibration_demand: pd.Series,
    test_demand: pd.Series,
    train_z: pd.Series,
    calibration_z: pd.Series,
    test_z: pd.Series,
) -> None:
    """Plot the pre-deseasonalisation half-hourly demand (top) against the
    Step 3 standardised residual (bottom) across all three splits, sharing
    a time axis. half-hourly only -- no daily variant exists here."""

    fig, (ax_demand, ax_z) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    ax_demand.plot(train_demand.index, train_demand.to_numpy(), linewidth=0.6, label="train", color="tab:blue")
    ax_demand.plot(calibration_demand.index, calibration_demand.to_numpy(), linewidth=0.6, label="calibration", color="tab:orange")
    ax_demand.plot(test_demand.index, test_demand.to_numpy(), linewidth=0.6, label="test", color="tab:green")
    ax_demand.set_title(f"{region} additive half_hourly: demand before deseasonalisation (half-hourly)")
    ax_demand.set_ylabel("demand")
    ax_demand.legend()

    ax_z.plot(train_z.index, train_z.to_numpy(), linewidth=0.6, label="train_z", color="tab:blue")
    ax_z.plot(calibration_z.index, calibration_z.to_numpy(), linewidth=0.6, label="calibration_z", color="tab:orange")
    ax_z.plot(test_z.index, test_z.to_numpy(), linewidth=0.6, label="test_z", color="tab:green")

    for level in (-3, -2, -1, 0, 1, 2, 3):
        ax_z.axhline(level, color="black", linewidth=0.5 if level == 0 else 0.3, linestyle="-" if level == 0 else "--")

    ax_z.set_title(f"{region} additive half_hourly: standardised residual (z-score, TRAIN mean/std)")
    ax_z.set_ylabel("z")
    ax_z.legend()
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / f"qc_additive_standardised_residual_{region}_half_hourly.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"  figure saved to {out_path}")


def main() -> None:
    for region in REGIONS:
        train, calibration, test = loader.load(region)

        train_s = demand_series(train)
        calibration_s = demand_series(calibration)
        test_s = demand_series(test)

        train_residual = remove_daily_weekly_profile(train_s, reference=train_s)
        calibration_residual = remove_daily_weekly_profile(calibration_s, reference=train_s)
        test_residual = remove_daily_weekly_profile(test_s, reference=train_s)

        # Step 2: plot + report TRAIN checks.
        plot_and_report(region, train_s, train_residual)

        # Step 3: standardise using TRAIN residual statistics only.
        train_z, calibration_z, test_z = standardise(
            train_residual,
            calibration_residual,
            test_residual,
            f"{region} additive half_hourly",
        )

        plot_standardised(region, train_s, calibration_s, test_s, train_z, calibration_z, test_z)


if __name__ == "__main__":
    main()
