"""
Dynamic Harmonic Regression + ARIMA errors — synthetic drift version
=========================================================================

Takes a single synthetic drift CSV (columns: t, y, is_changepoint — the
output of make_series() / save_series_to_csv()), splits it into train
and test using scikit-learn, fits a frozen DHR+ARIMA baseline on the
training portion, forecasts across the test portion, and outputs a
rolling MAE curve — the synthetic-data sanity check that should precede
trusting this model on real AEMO data.

IMPORTANT — read before changing the split fraction
-----------------------------------------------------
This is NOT a random-shuffle ML split. Two things make it different from
a typical scikit-learn train/test split:

1. Order matters. Splitting a time series must never shuffle rows —
   `train_test_split(..., shuffle=False)` is used specifically to keep
   the last N% as a contiguous block after the first (1-N)%, not a
   random sample. Shuffling would let the model "see the future" during
   training and would make training data appear interleaved with drift
   in a way that never happens in reality.

2. The split boundary should not land in the middle of the pre-drift
   period by accident, and for this project's purpose (proving a frozen
   model degrades under drift), the split boundary SHOULD align with
   the true changepoint — training on drifted data would defeat the
   entire point of the test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------------
# Fourier / model fitting
# ---------------------------------------------------------------------------

def build_fourier_terms(n: int, periods: list[int], n_harmonics: list[int]) -> pd.DataFrame:
    """Build sine/cosine Fourier terms for one or more seasonal periods."""
    t = np.arange(n)
    terms = {}
    for period, K in zip(periods, n_harmonics):
        for k in range(1, K + 1):
            terms[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            terms[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms)


def fit_dhr_arima(
    y_train: np.ndarray,
    periods: list[int] = [48],
    n_harmonics: list[int] = [3],
    arima_order: tuple[int, int, int] = (2, 0, 2),
):
    """
    Fit the frozen DHR+ARIMA baseline on the training window only.

    Parameters for the SYNTHETIC generator (make_series):
        periods = [48]
            The synthetic generator only injects ONE seasonal cycle
            (period = 48, a "daily" sine wave — see generator.py's
            `period = 48` constant). Unlike the real AEMO data, there is
            no second (weekly) cycle actually present in this signal, so
            passing periods=[48, 336] here would just fit the 336-period
            Fourier terms to noise. Use [48] only.
        n_harmonics = [3]
            The synthetic signal is a single clean sine wave (plus
            noise) — it doesn't need many harmonics to represent
            faithfully. 3 is more than enough; going higher mostly just
            risks fitting noise.
        arima_order = (2, 0, 2)
            Same reasoning as the AEMO version — a reasonable starting
            point for whatever autocorrelation is left after removing
            the seasonal shape.
    """
    y_train = np.asarray(y_train)
    n = len(y_train)
    exog_train = build_fourier_terms(n, periods, n_harmonics)

    model = SARIMAX(
        y_train,
        exog=exog_train,
        order=arima_order,
        trend="c",
        seasonal_order=(0, 0, 0, 0),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, method="bfgs", maxiter=300)
    return fitted


def forecast_dhr_arima(
    fitted,
    n_train: int,
    horizon: int,
    periods: list[int] = [48],
    n_harmonics: list[int] = [3],
) -> np.ndarray:
    """Forecast forward from the frozen model. Never re-fits."""
    t_future = np.arange(n_train, n_train + horizon)
    terms = {}
    for period, K in zip(periods, n_harmonics):
        for k in range(1, K + 1):
            terms[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t_future / period)
            terms[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t_future / period)
    exog_future = pd.DataFrame(terms)

    result = fitted.get_forecast(steps=horizon, exog=exog_future)
    return np.asarray(result.predicted_mean)


# ---------------------------------------------------------------------------
# Loading + splitting the single synthetic CSV
# ---------------------------------------------------------------------------

def load_synthetic_csv(path: str) -> pd.DataFrame:
    """Load a make_series()-generated CSV (t, y, is_changepoint)."""
    df = pd.read_csv(path)
    df = df.sort_values("t").reset_index(drop=True)
    return df


def split_synthetic_series(
    df: pd.DataFrame,
    test_size: float = 0.5,
    calibration_size: float = 0.0,
):
    """
    Split the synthetic series into train / test (/ calibration) using
    scikit-learn's train_test_split, with shuffle=False (mandatory for
    time series — see module docstring).

    Also checks the split against the file's true changepoint(s) and
    warns loudly if the chosen split would let drifted data leak into
    the training set.

    Parameters
    ----------
    test_size : fraction of rows to hold out as the final test set.
    calibration_size : fraction of rows to hold out as a calibration
        set, taken from what would otherwise be training data (i.e.
        calibration sits BETWEEN train and test in time, not after
        test). Set to 0.0 to skip calibration entirely (see
        justification below).

    Returns
    -------
    dict with 'train', 'test', and optionally 'calibration' — each a
    DataFrame with columns t, y, is_changepoint, in time order.
    """
    changepoints = df.loc[df["is_changepoint"] == 1, "t"].tolist()

    # Step 1: carve off the test set from the end — shuffle=False keeps
    # it as the LAST test_size fraction of rows, not a random sample.
    train_and_cal, test = train_test_split(df, test_size=test_size, shuffle=False)

    result = {}
    if calibration_size > 0:
        # Step 2: carve calibration off the end of what remains (so the
        # final order is train -> calibration -> test, matching the
        # AEMO three-way split's structure).
        cal_fraction_of_remaining = calibration_size / (1 - test_size)
        train, calibration = train_test_split(
            train_and_cal, test_size=cal_fraction_of_remaining, shuffle=False
        )
        result["train"] = train.reset_index(drop=True)
        result["calibration"] = calibration.reset_index(drop=True)
    else:
        result["train"] = train_and_cal.reset_index(drop=True)

    result["test"] = test.reset_index(drop=True)

    # Contamination check: warn if any changepoint falls inside the
    # training portion, since that defeats the "frozen pre-drift model"
    # premise this whole exercise depends on.
    train_end_t = result["train"]["t"].max()
    contaminating_cps = [cp for cp in changepoints if cp <= train_end_t]
    if contaminating_cps:
        print(
            f"WARNING: changepoint(s) {contaminating_cps} fall within the "
            f"training set (t=0 to t={train_end_t}). The model will be "
            f"trained on already-drifted data, which defeats the purpose "
            f"of testing whether a FROZEN pre-drift model degrades after "
            f"drift. Consider reducing test_size / calibration_size, or "
            f"splitting at the changepoint directly instead of by fraction."
        )
    else:
        print(f"OK: training set (t=0 to t={train_end_t}) ends before "
              f"changepoint(s) {changepoints} — no contamination.")

    return result


# ---------------------------------------------------------------------------
# End-to-end run
# ---------------------------------------------------------------------------

def run_dhr_arima_on_synthetic(
    csv_path: str,
    test_size: float = 0.5,
    calibration_size: float = 0.0,
    periods: list[int] = [48],
    n_harmonics: list[int] = [3],
    arima_order: tuple[int, int, int] = (2, 0, 2),
    rolling_window: int = 240,
) -> pd.Series:
    """
    End-to-end: load the synthetic CSV, split it, fit the frozen model
    on train, forecast across test, return a rolling MAE curve indexed
    by t.

    rolling_window : int
        Row-based rolling window (no real dates here, just row count).
        240 = 5 "days" at period=48, a reasonable default; adjust as
        needed.
    """
    df = load_synthetic_csv(csv_path)
    splits = split_synthetic_series(df, test_size=test_size, calibration_size=calibration_size)

    y_train = splits["train"]["y"].to_numpy()
    y_test = splits["test"]["y"].to_numpy()
    n_train = len(y_train)
    horizon = len(y_test)

    print(f"Fitting DHR+ARIMA on {n_train} training rows (t=0 to t={n_train - 1})...")
    fitted = fit_dhr_arima(y_train, periods=periods, n_harmonics=n_harmonics, arima_order=arima_order)

    print(f"Forecasting {horizon} steps ahead...")
    preds = forecast_dhr_arima(fitted, n_train=n_train, horizon=horizon, periods=periods, n_harmonics=n_harmonics)

    abs_error = pd.Series(np.abs(y_test - preds), index=splits["test"]["t"].to_numpy())
    mae_curve = abs_error.rolling(rolling_window).mean()
    return mae_curve


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # ---- Split decision: 80/20 train/test, NO calibration set. ----
    # Justification: this script covers Step 3 (frozen baseline +
    # degradation curve) only. A calibration set is specifically needed
    # for Step 6 (conformal prediction / uncertainty intervals), which
    # is a separate downstream task with its own script and its own
    # held-out data — introducing a calibration split here would just
    # shrink the test stream without serving this script's purpose.
    # If the team later wants one shared 3-way split reused everywhere
    # (mirroring the AEMO train/calibration/test convention) that's a
    # reasonable alternative — just make it an explicit team decision,
    # not a silent default, since it changes what "test" means in every
    # downstream table.

    #sudden drift synthetic series (n=20000, seed=1)
    mae_curve_sudden = run_dhr_arima_on_synthetic(
        csv_path="series_sudden_n20000_seed1.csv",
        test_size=0.5,
        calibration_size=0.0,
        periods=[48],
        n_harmonics=[3],
        arima_order=(2, 0, 2),
        rolling_window=240,
    )

    mae_curve_sudden.to_csv("synthetic_sudden_rolling_mae.csv", header=["mae"])
    print("Saved rolling MAE series to synthetic_sudden_rolling_mae.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mae_curve_sudden.index, mae_curve_sudden.values, linewidth=1, color="#2c6e8f")
    ax.set_ylabel("Rolling MAE")
    ax.set_xlabel("t")
    ax.set_title("DHR + ARIMA errors — frozen baseline on synthetic drift, rolling MAE")
    fig.tight_layout()
    fig.savefig("synthetic_sudden_rolling_mae.png", dpi=130)
    print("Saved plot to synthetic_sudden_rolling_mae.png")


    #gradual drift synthetic series (n=20000, seed=1)
    mae_curve_gradual = run_dhr_arima_on_synthetic(
        csv_path="series_gradual_n20000_seed1.csv",
        test_size=0.5,
        calibration_size=0.0,
        periods=[48],
        n_harmonics=[3],
        arima_order=(2, 0, 2),
        rolling_window=240,
    )

    mae_curve_gradual.to_csv("synthetic_gradual_rolling_mae.csv", header=["mae"])
    print("Saved rolling MAE series to synthetic_gradual_rolling_mae.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mae_curve_gradual.index, mae_curve_gradual.values, linewidth=1, color="#8f2c2c")
    ax.set_ylabel("Rolling MAE")
    ax.set_xlabel("t")
    ax.set_title("DHR + ARIMA errors — frozen baseline on synthetic drift, rolling MAE")
    fig.tight_layout()
    fig.savefig("synthetic_gradual_rolling_mae.png", dpi=130)
    print("Saved plot to synthetic_gradual_rolling_mae.png")

    #recurring drift synthetic series (n=20000, seed=1)
    mae_curve_recurring = run_dhr_arima_on_synthetic(
        csv_path="series_recurring_n20000_seed1.csv",
        test_size=0.5,
        calibration_size=0.0,
        periods=[48],
        n_harmonics=[3],
        arima_order=(2, 0, 2),
        rolling_window=240,
    )

    mae_curve_recurring.to_csv("synthetic_recurring_rolling_mae.csv", header=["mae"])
    print("Saved rolling MAE series to synthetic_recurring_rolling_mae.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mae_curve_recurring.index, mae_curve_recurring.values, linewidth=1, color="#458f2c")
    ax.set_ylabel("Rolling MAE")
    ax.set_xlabel("t")
    ax.set_title("DHR + ARIMA errors — frozen baseline on synthetic drift, rolling MAE")
    fig.tight_layout()
    fig.savefig("synthetic_recurring_rolling_mae.png", dpi=130)
    print("Saved plot to synthetic_recurring_rolling_mae.png")


