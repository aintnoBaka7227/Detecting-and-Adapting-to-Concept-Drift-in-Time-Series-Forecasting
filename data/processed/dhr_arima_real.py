"""
Dynamic Harmonic Regression + ARIMA errors — AEMO version
============================================================

Takes SA1_train.csv / SA1_test.csv (or NSW1 equivalents), fits a frozen
DHR+ARIMA baseline on the training window, forecasts across the entire
test stream, and outputs a rolling MAE curve indexed by SETTLEMENTDATE —
the same "detector statistic over time" shape used for the degradation
curve (F1) / detection-style panels elsewhere in the project.

Expected CSV columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP
    y_train = SA1_train.csv['TOTALDEMAND']   (35,040 rows = 730 days
              = exactly 2018-01-01 to 2019-12-31, half-hourly)
    y_test  = SA1_test.csv['TOTALDEMAND']    (67,250 rows ≈ the full
              test stream, 2020-03-01 onward)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------------
# Fourier / model fitting — unchanged in mechanics from the synthetic version
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
    periods: list[int] = [48, 336],
    n_harmonics: list[int] = [4, 4],
    arima_order: tuple[int, int, int] = (2, 0, 2),
):
    """
    Fit the frozen DHR+ARIMA baseline on the training window only.

    Parameters for AEMO data (SA1/NSW1, half-hourly):
        periods      = [48, 336]
            48  = one day (24h * 2 half-hour intervals/hour)
            336 = one week (7 days * 48 intervals/day)
            These are fixed by the data's actual sampling rate and the
            two real seasonal cycles your team identified — not a tuning
            choice.
        n_harmonics  = [4, 4]
            4 harmonic pairs per period is a reasonable starting point
            given 35,040 training rows = 730 daily cycles and ~104
            weekly cycles — plenty of repetitions to reliably estimate
            more harmonics than the earlier short-series synthetic test
            could support. Increase if the seasonal shape still looks
            underfit (e.g. a sharp morning + evening double-peak isn't
            being captured); watch AIC/BIC if you raise it much further.
        arima_order  = (2, 0, 2)
            A reasonable starting point for the residual structure.
            d=0 because DHR already removes trend/seasonality, so the
            leftover should already look roughly stationary. Tune p/q
            via AIC/BIC or residual ACF/PACF plots if needed — this is
            not something to guess once and leave unquestioned.
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
    periods: list[int] = [48, 336],
    n_harmonics: list[int] = [4, 4],
) -> np.ndarray:
    """
    Forecast forward from the frozen model. Never re-fits.

    Parameters for AEMO data:
        n_train  = 35040
            Length of y_train — needed so the Fourier terms continue
            their phase correctly from where training left off, instead
            of restarting at t=0 (which would misalign daily/weekly
            phase against the test period).
        horizon  = 67250
            Length of y_test — forecast one point for every row in the
            test stream, in one pass (no re-fitting along the way).
        periods, n_harmonics
            Must exactly match what was passed to fit_dhr_arima(),
            otherwise the Fourier terms won't line up with the fitted
            regression weights.
    """
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
# AEMO-specific: load CSVs, run the frozen model, produce a rolling MAE curve
# ---------------------------------------------------------------------------

def load_aemo_csv(path: str) -> pd.DataFrame:
    """
    Load an AEMO-format CSV (REGION, SETTLEMENTDATE, TOTALDEMAND, RRP),
    parse timestamps, and sort by time.
    """
    df = pd.read_csv(path, parse_dates=["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    return df


def rolling_mae_series(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: str = "7D",
) -> pd.Series:
    """
    Compute a timestamped rolling MAE curve — the "detector statistic
    over time" shape used for degradation-curve style figures.

    Parameters
    ----------
    dates : the SETTLEMENTDATE values matching y_true/y_pred, in order.
    y_true, y_pred : actual vs forecast TOTALDEMAND.
    window : rolling window size. Default "7D" (7-day rolling MAE),
        matching Step 3's stated window ("Roll forward through the test
        stream computing MAE in a 7-day rolling window").

    Returns
    -------
    pd.Series, indexed by SETTLEMENTDATE, one rolling-MAE value per
    original row (NaN for the initial partial window at the start).
    """
    abs_error = pd.Series(np.abs(y_true - y_pred), index=pd.DatetimeIndex(dates))
    rolling_mae = abs_error.rolling(window).mean()
    return rolling_mae


def run_dhr_arima_on_aemo(
    train_csv: str,
    test_csv: str,
    periods: list[int] = [48, 336],
    n_harmonics: list[int] = [4, 4],
    arima_order: tuple[int, int, int] = (2, 0, 2),
    rolling_window: str = "7D",
) -> pd.Series:
    """
    End-to-end: load train/test CSVs, fit the frozen model on train,
    forecast across the whole test stream, return a rolling MAE curve.

    This is the "n_train / horizon / periods / n_harmonics" block from
    forecast_dhr_arima(), wired up automatically from the CSVs' actual
    lengths — you don't set n_train/horizon by hand, they're read off
    the data:
        n_train = len(y_train)   -> 35040 for the SA1 split described
        horizon = len(y_test)    -> 67250 for the SA1 split described
    """
    train_df = load_aemo_csv(train_csv)
    test_df = load_aemo_csv(test_csv)

    y_train = train_df["TOTALDEMAND"].to_numpy()
    y_test = test_df["TOTALDEMAND"].to_numpy()
    n_train = len(y_train)
    horizon = len(y_test)

    print(f"Fitting DHR+ARIMA on {n_train} training rows "
          f"({train_df['SETTLEMENTDATE'].iloc[0]} to {train_df['SETTLEMENTDATE'].iloc[-1]})...")
    fitted = fit_dhr_arima(y_train, periods=periods, n_harmonics=n_harmonics, arima_order=arima_order)

    print(f"Forecasting {horizon} steps ahead "
          f"({test_df['SETTLEMENTDATE'].iloc[0]} to {test_df['SETTLEMENTDATE'].iloc[-1]})...")
    preds = forecast_dhr_arima(fitted, n_train=n_train, horizon=horizon, periods=periods, n_harmonics=n_harmonics)

    mae_curve = rolling_mae_series(test_df["SETTLEMENTDATE"], y_test, preds, window=rolling_window)
    return mae_curve


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mae_curve_sa = run_dhr_arima_on_aemo(
        train_csv="SA1_train.csv",
        test_csv="SA1_test.csv",
        periods=[48, 336],
        n_harmonics=[4, 4],
        arima_order=(2, 0, 2),
        rolling_window="7D",
    )

    mae_curve_nsw = run_dhr_arima_on_aemo(
        train_csv="NSW1_train.csv",
        test_csv="NSW1_test.csv",
        periods=[48, 336],
        n_harmonics=[4, 4],
        arima_order=(2, 0, 2),
        rolling_window="7D",
    )


    mae_curve_sa.to_csv("SA1_rolling_mae.csv", header=["mae"])
    mae_curve_nsw.to_csv("NSW1_rolling_mae.csv", header=["mae"])

    print("Saved rolling MAE series to SA1_rolling_mae.csv")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mae_curve_sa.index, mae_curve_sa.values, linewidth=1, color="#2c6e8f", label="SA1 DHR+ARIMA")
    ax.plot(mae_curve_nsw.index, mae_curve_nsw.values, linewidth=1,color="#d9534f", label="NSW1 DHR+ARIMA")
    ax.legend()
    ax.set_ylabel("7-day rolling MAE (MW)")
    ax.set_xlabel("SETTLEMENTDATE")
    ax.set_title("DHR + ARIMA Errors. Frozen baseline, rolling MAE over test stream")
    fig.tight_layout()
    fig.savefig("AEMO_rolling_mae.png", dpi=130)
    print("Saved plot to AEMO_rolling_mae.png")
