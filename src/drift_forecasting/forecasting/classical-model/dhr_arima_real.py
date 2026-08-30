from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX


def build_fourier_terms(t, periods: list[int], n_harmonics: list[int]) -> pd.DataFrame:
    t = np.arange(t) if isinstance(t, int) else np.asarray(t)
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
    t_train=None,
):
    y_train = np.asarray(y_train)
    n = len(y_train)
    if t_train is None:
        t_train = n  # backward-compatible default: assume no internal gaps
    exog_train = build_fourier_terms(t_train, periods, n_harmonics)

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
    t_future: np.ndarray,
    periods: list[int] = [48, 336],
    n_harmonics: list[int] = [4, 4],
) -> np.ndarray:
    t_future = np.asarray(t_future)
    horizon = len(t_future)
    exog_future = build_fourier_terms(t_future, periods, n_harmonics)
    result = fitted.get_forecast(steps=horizon, exog=exog_future)
    return np.asarray(result.predicted_mean)


def load_aemo_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["SETTLEMENTDATE"])
    df = df.sort_values("SETTLEMENTDATE").reset_index(drop=True)
    return df


def elapsed_steps(df: pd.DataFrame, epoch: pd.Timestamp, step_minutes: int = 30) -> np.ndarray:
    # Real elapsed half-hour steps from a shared epoch, NOT a naive row
    # count. This is what keeps the daily/weekly Fourier phase correct
    # across any gap between splits (e.g. the calibration period sitting
    # between train and test) instead of silently assuming test starts
    # the instant train ends.
    delta = (df["SETTLEMENTDATE"] - epoch) / pd.Timedelta(minutes=step_minutes)
    return delta.round().astype(int).to_numpy()


def rolling_mae_series(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: str = "7D",
    step_minutes: int = 30,
) -> pd.Series:
    abs_error = pd.Series(np.abs(y_true - y_pred), index=pd.DatetimeIndex(dates))
    # Require a completely full window before reporting a value, so the first
    # (window - 1) test intervals come out as NaN instead of partial-window
    # means. This matches the seasonal-naive baseline
    # (min_periods == ROLLING_WINDOW); for regular half-hourly data a 7-day
    # window is 336 intervals, so the first 335 test rows are NaN.
    min_periods = int(pd.Timedelta(window) / pd.Timedelta(minutes=step_minutes))
    rolling_mae = abs_error.rolling(window, min_periods=min_periods).mean()
    return rolling_mae


def run_dhr_arima_on_aemo(
    train_csv: str,
    test_csv: str,
    calibration_csv: str | None = None,
    periods: list[int] = [48, 336],
    n_harmonics: list[int] = [4, 4],
    arima_order: tuple[int, int, int] = (2, 0, 2),
    rolling_window: str = "7D",
) -> pd.Series:
    train_df = load_aemo_csv(train_csv)
    test_df = load_aemo_csv(test_csv)

    epoch = train_df["SETTLEMENTDATE"].iloc[0]
    t_train = elapsed_steps(train_df, epoch)
    t_test = elapsed_steps(test_df, epoch)

    if calibration_csv is not None:
        calibration_df = load_aemo_csv(calibration_csv)
        t_cal = elapsed_steps(calibration_df, epoch)
        gap_steps = t_test[0] - (t_train[-1] + 1)
        print(f"Calibration file spans {len(calibration_df)} rows "
              f"({calibration_df['SETTLEMENTDATE'].iloc[0]} to {calibration_df['SETTLEMENTDATE'].iloc[-1]}); "
              f"gap between train end and test start = {gap_steps} steps.")

    y_train = train_df["TOTALDEMAND"].to_numpy()
    y_test = test_df["TOTALDEMAND"].to_numpy()

    print(f"Fitting DHR+ARIMA on {len(y_train)} training rows "
          f"({train_df['SETTLEMENTDATE'].iloc[0]} to {train_df['SETTLEMENTDATE'].iloc[-1]})...")
    fitted = fit_dhr_arima(y_train, periods=periods, n_harmonics=n_harmonics, arima_order=arima_order, t_train=t_train)

    print(f"Forecasting {len(y_test)} steps ahead "
          f"({test_df['SETTLEMENTDATE'].iloc[0]} to {test_df['SETTLEMENTDATE'].iloc[-1]})...")
    preds = forecast_dhr_arima(fitted, t_future=t_test, periods=periods, n_harmonics=n_harmonics)

    mae_curve = rolling_mae_series(test_df["SETTLEMENTDATE"], y_test, preds, window=rolling_window)
    return mae_curve


def format_month_axis(ax, year: int) -> None:
    ax.set_xlim(pd.Timestamp(year=year, month=1, day=1), pd.Timestamp(year=year + 1, month=1, day=1))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", labelrotation=45, labelsize=9)
    ax.grid(alpha=0.25)


def detect_mae_spikes(mae_series: pd.Series, region: str, threshold_std: float = 2.0,
                       min_gap_hours: float = 24.0) -> pd.DataFrame:
    threshold = mae_series.mean() + threshold_std * mae_series.std()
    flagged = mae_series >= threshold
    groups = (flagged != flagged.shift()).cumsum()

    raw_events = []
    for _, grp in mae_series[flagged].groupby(groups[flagged]):
        raw_events.append({
            "start": grp.index.min(),
            "end": grp.index.max(),
            "peak_mae": grp.max(),
            "peak_time": grp.idxmax(),
        })
    raw_events.sort(key=lambda e: e["start"])

    merged_events = []
    for ev in raw_events:
        if merged_events and ev["start"] - merged_events[-1]["end"] <= pd.Timedelta(hours=min_gap_hours):
            merged_events[-1]["end"] = max(merged_events[-1]["end"], ev["end"])
            if ev["peak_mae"] > merged_events[-1]["peak_mae"]:
                merged_events[-1]["peak_mae"] = ev["peak_mae"]
                merged_events[-1]["peak_time"] = ev["peak_time"]
        else:
            merged_events.append(dict(ev))

    rows = []
    for i, ev in enumerate(merged_events, start=1):
        start_date = ev["start"]
        end_date = ev["end"]
        rows.append({
            "event_id": f"EVT-{region}-{i:03d}",
            "event_name": f"{region} MAE spike - peak {ev['peak_mae']:.0f} MW on {ev['peak_time'].date()}",
            "start_date": start_date,
            "end_date": end_date,
            "date_precision": "day" if start_date.date() == end_date.date() else "date_range",
            "region": region,
            "event_type": "mae_spike",
            "description": f"Rolling MAE exceeded mean + {threshold_std:.1f} std ({threshold:.0f} MW) "
                            f"between {start_date} and {end_date}, peaking at {ev['peak_mae']:.0f} MW on {ev['peak_time']}.",
            "peak_mae_mw": round(ev["peak_mae"], 1),
            "source_title": "Auto-detected from rolling MAE",
        })
    return pd.DataFrame(rows)


def plot_year_with_events(series: pd.Series, events: pd.DataFrame, region: str, year: int,
                           label: str, color: str, out_prefix: str = "AEMO_rolling_mae_events") -> None:
    year_series = series[series.index.year == year]
    if year_series.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(year_series.index, year_series.values, linewidth=1.2, color=color, label=label)

    region_events = events[events["region"] == region]
    year_events = region_events[
        (region_events["start_date"] <= pd.Timestamp(year=year, month=12, day=31))
        & (region_events["end_date"] >= pd.Timestamp(year=year, month=1, day=1))
    ].sort_values("start_date").reset_index(drop=True)

    ax.set_title(f"DHR + ARIMA Errors — {region} {year}", loc="left")
    ax.set_xlabel("Date")
    ax.set_ylabel("7-day rolling MAE (MW)")
    format_month_axis(ax, year)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, borderaxespad=0.0)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    for i, ev in year_events.iterrows():
        if ev["date_precision"] in ("date_range", "month"):
            ax.axvspan(ev["start_date"], ev["end_date"], color="red", alpha=0.08)
        ax.axvline(ev["start_date"], color="red", linestyle="--", linewidth=0.7)
        y_text = ymax - (i % 4) * (yrange * 0.045) - yrange * 0.02
        ax.text(ev["start_date"], y_text, ev["event_name"], rotation=0, fontsize=6, color="red", va="top", ha="left")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}_{region}_{year}.png", dpi=130)
    print(f"Saved plot to {out_prefix}_{region}_{year}.png")


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    mae_curve_sa = run_dhr_arima_on_aemo(
        train_csv="SA1_train.csv",
        test_csv="SA1_test.csv",
        calibration_csv="SA1_calibration.csv",
        periods=[48, 336],
        n_harmonics=[4, 4],
        arima_order=(2, 0, 2),
        rolling_window="7D",
    )

    mae_curve_nsw = run_dhr_arima_on_aemo(
        train_csv="NSW1_train.csv",
        test_csv="NSW1_test.csv",
        calibration_csv="NSW1_calibration.csv",
        periods=[48, 336],
        n_harmonics=[4, 4],
        arima_order=(2, 0, 2),
        rolling_window="7D",
    )

    mae_curve_sa.to_csv("SA1_rolling_mae.csv", header=["mae"])
    mae_curve_nsw.to_csv("NSW1_rolling_mae.csv", header=["mae"])
    print("Saved rolling MAE series to SA1_rolling_mae.csv / NSW1_rolling_mae.csv")

    # --- Full-range plot (as before) ---
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mae_curve_sa.index, mae_curve_sa.values, linewidth=1, color="#2c6e8f", label="SA1 DHR+ARIMA")
    ax.plot(mae_curve_nsw.index, mae_curve_nsw.values, linewidth=1, color="#d9534f", label="NSW1 DHR+ARIMA")
    ax.legend()
    ax.set_ylabel("7-day rolling MAE (MW)")
    ax.set_xlabel("SETTLEMENTDATE")
    ax.set_title("DHR + ARIMA Errors. Frozen baseline, rolling MAE over test stream")
    fig.tight_layout()
    fig.savefig("AEMO_rolling_mae.png", dpi=130)
    print("Saved plot to AEMO_rolling_mae.png")

    # --- Yearly plots, one PNG per calendar year present in the test period ---
    test_years = sorted({
        ts.year
        for series in (mae_curve_sa, mae_curve_nsw)
        for ts in series.index
    })

    for year in test_years:
        fig, ax = plt.subplots(figsize=(14, 5))
        for series, label, color in [
            (mae_curve_sa, "SA1 DHR+ARIMA", "#2c6e8f"),
            (mae_curve_nsw, "NSW1 DHR+ARIMA", "#d9534f"),
        ]:
            year_series = series[series.index.year == year]
            ax.plot(year_series.index, year_series.values, linewidth=1.2, color=color, label=label)

        ax.set_title(f"DHR + ARIMA Errors — {year}", loc="left")
        ax.set_xlabel("Date")
        ax.set_ylabel("7-day rolling MAE (MW)")
        format_month_axis(ax, year)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2, frameon=False, borderaxespad=0.0)
        fig.tight_layout()
        fig.savefig(f"AEMO_rolling_mae_{year}.png", dpi=130)
        print(f"Saved plot to AEMO_rolling_mae_{year}.png")

    # --- Detect MAE spikes, build events.csv, and save event-annotated yearly plots ---
    events_sa = detect_mae_spikes(mae_curve_sa, region="SA1")
    events_nsw = detect_mae_spikes(mae_curve_nsw, region="NSW1")
    events_df = pd.concat([events_sa, events_nsw], ignore_index=True)
    events_df.to_csv("events.csv", index=False)
    print(f"Saved {len(events_df)} detected events to events.csv")

    for region, series, label, color in [
        ("SA1", mae_curve_sa, "SA1 DHR+ARIMA", "#2c6e8f"),
        ("NSW1", mae_curve_nsw, "NSW1 DHR+ARIMA", "#d9534f"),
    ]:
        for year in test_years:
            plot_year_with_events(series, events_df, region, year, label, color)
