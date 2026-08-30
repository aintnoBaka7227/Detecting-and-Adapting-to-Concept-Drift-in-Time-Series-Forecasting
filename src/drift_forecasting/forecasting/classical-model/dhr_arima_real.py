

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX



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

    t_future = np.arange(n_train, n_train + horizon)
    terms = {}
    for period, K in zip(periods, n_harmonics):
        for k in range(1, K + 1):
            terms[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t_future / period)
            terms[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t_future / period)
    exog_future = pd.DataFrame(terms)

    result = fitted.get_forecast(steps=horizon, exog=exog_future)
    return np.asarray(result.predicted_mean)




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
