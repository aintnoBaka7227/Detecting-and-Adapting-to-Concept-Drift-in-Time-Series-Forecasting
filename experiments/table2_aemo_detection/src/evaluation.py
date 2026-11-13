"""Shared evaluation metrics: every project metric is computed here.

Sections, in file order:
    forecast      calculate_mae, calculate_rolling_mae (default 336 = 7 days)
    synthetic     evaluate_detections: delay, false alarms per 10,000 and
                  missed detections against known changepoints
    AEMO events   build_event_windows, regime assignment, match_unmatch,
                  calculate_event_metrics
    adaptation    MAE by regime, adaptation gain, pinball loss
    uncertainty   coverage, width, rolling and worst-window coverage (T5)
    escalation    review rate, retained coverage, threshold sweep (T6, F5)

Synthetic matching: a detection matches a changepoint when it falls in
[cp, cp + tolerance] (default 336); a gradual drift is one event
[drift_start, drift_end + grace], a recurring drift two events, and in the
no-drift control every detection is a false alarm. Delay is measured from
the changepoint (drift_start for gradual drift).

AEMO matching: documented events are context, not ground truth, so an
unmatched detection is "Unmatch", not a false alarm. Detections are first
debounced (REFRACTORY_PERIOD), then matched one-to-one, Tier 1 before
Tier 2. Precision is matched / accepted detections; precision_t1 and
precision_t2 use the same pooled denominator.

Metrics are per run; aggregation across seeds happens in the producers.
"""

import math

import numpy as np
import pandas as pd

# Forecast evaluation metrics


def calculate_mae(y_true, y_pred):
    """
    Calculate Mean Absolute Error (MAE) using the shared
    project evaluation logic.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Forecast values.

    Returns
    -------
    float
        Mean absolute error.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("y_true and y_pred must not contain missing values.")

    return float((y_true - y_pred).abs().mean())


def calculate_rolling_mae(y_true, y_pred, window=48 * 7):
    """
    Calculate rolling Mean Absolute Error.

    Default:
        48 half-hour intervals × 7 days = 336 observations.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    y_pred : array-like
        Forecast values.

    window : int
        Number of observations in the rolling window.

    Returns
    -------
    pd.Series
        Rolling MAE.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_pred = pd.Series(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("y_true and y_pred must not contain missing values.")

    absolute_error = (y_true - y_pred).abs()

    return absolute_error.rolling(window=window).mean()


# Detection evaluation metrics 

def _validate_changepoints(changepoints, n_observations, name):
    """
    Validate changepoint indices and return them as sorted unique integers.
    """
    if n_observations <= 0:
        raise ValueError("n_observations must be greater than 0.")

    values = []

    for cp in changepoints:
        if isinstance(cp, bool):
            raise ValueError(f"{name} must contain integer indices.")  # noqa: TRY004

        cp_int = int(cp)

        if cp_int != cp:
            raise ValueError(f"{name} must contain integer indices.")

        if not 0 <= cp_int < n_observations:
            raise ValueError(
                f"{name} contains index {cp_int}, which is outside "
                f"0..{n_observations - 1}."
            )

        values.append(cp_int)

    return sorted(set(values))


def _match_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Match detected changepoints to synthetic ground truth.

    none: no true events, every detection is a false alarm. sudden/recurring:
    each changepoint takes the first unused detection in [cp, cp + tolerance].
    gradual: the one transition [drift_start, drift_end] takes the first unused
    detection in [drift_start, drift_end + gradual_grace]; delay is measured
    from drift_start. A detection matches at most one event.

    Parameters
    ----------
    detected_changepoints, true_changepoints : array-like
        Indices in the same detector-input coordinate system.
    n_observations : int
        Length of the detector input stream.
    drift_type : str
        "none", "sudden", "gradual" or "recurring".
    tolerance : int
        Maximum post-changepoint delay for point events (default 336, 7 days).
    gradual_grace : int, optional
        Allowance after a gradual window; defaults to `tolerance`.

    Returns
    -------
    dict
        matched_pairs, delays, false_alarm_indices, missed_true_events,
        n_true_events.
    """
    drift_type = str(drift_type).lower()

    if drift_type not in {
        "none",
        "sudden",
        "gradual",
        "recurring",
    }:
        raise ValueError(
            "drift_type must be 'none', 'sudden', 'gradual', or 'recurring'."
        )

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")

    if gradual_grace is None:
        gradual_grace = tolerance

    if gradual_grace < 0:
        raise ValueError("gradual_grace must be non-negative.")

    detected = _validate_changepoints(
        detected_changepoints,
        n_observations,
        "detected_changepoints",
    )

    truth = _validate_changepoints(
        true_changepoints,
        n_observations,
        "true_changepoints",
    )

    if drift_type == "none":
        if truth:
            raise ValueError("No-drift evaluation requires true_changepoints=[]")

        return {
            "matched_pairs": [],
            "delays": [],
            "false_alarm_indices": detected,
            "missed_true_events": 0,
            "n_true_events": 0,
        }

    if drift_type == "sudden" and len(truth) != 1:
        raise ValueError("Sudden drift requires exactly one true changepoint.")

    if drift_type == "gradual" and len(truth) != 2:
        raise ValueError("Gradual drift requires [drift_start, drift_end].")

    if drift_type == "recurring" and len(truth) != 2:
        raise ValueError("Recurring drift requires exactly two true changepoints.")

    unused = set(detected)
    matched_pairs = []
    delays = []
    missed = 0

    if drift_type == "gradual":
        drift_start, drift_end = truth
        match_end = min(
            n_observations - 1,
            drift_end + gradual_grace,
        )

        candidates = [
            cp for cp in detected if cp in unused and drift_start <= cp <= match_end
        ]

        if candidates:
            detected_cp = candidates[0]
            unused.remove(detected_cp)

            matched_pairs.append(
                {
                    "true_start": drift_start,
                    "true_end": drift_end,
                    "detected": detected_cp,
                }
            )
            delays.append(detected_cp - drift_start)
        else:
            missed = 1

        n_true_events = 1

    else:
        # sudden and recurring are point-event matching cases
        for true_cp in truth:
            match_end = min(
                n_observations - 1,
                true_cp + tolerance,
            )

            candidates = [
                cp for cp in detected if cp in unused and true_cp <= cp <= match_end
            ]

            if candidates:
                detected_cp = candidates[0]
                unused.remove(detected_cp)

                matched_pairs.append(
                    {
                        "true": true_cp,
                        "detected": detected_cp,
                    }
                )
                delays.append(detected_cp - true_cp)
            else:
                missed += 1

        n_true_events = len(truth)

    false_alarm_indices = [cp for cp in detected if cp in unused]

    return {
        "matched_pairs": matched_pairs,
        "delays": delays,
        "false_alarm_indices": false_alarm_indices,
        "missed_true_events": missed,
        "n_true_events": n_true_events,
    }


def calculate_detection_delay(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Calculate mean detection delay in observations.

    Delay is calculated only for successfully matched true drift events.
    Returns NaN when no drift event is matched, including the no-drift
    control.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    if not result["delays"]:
        return float("nan")

    return float(
        pd.Series(
            result["delays"],
            dtype=float,
        ).mean()
    )


def calculate_false_alarms_per_10000(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Calculate false alarms normalised per 10,000 observations.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    n_false_alarms = len(result["false_alarm_indices"])

    return float(n_false_alarms / n_observations * 10000)


def calculate_missed_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Calculate the number of true drift events that were not detected.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    return int(result["missed_true_events"])


def evaluate_detections(
    detected_changepoints,
    true_changepoints,
    n_observations,
    drift_type,
    tolerance=48 * 7,
    gradual_grace=None,
):
    """
    Evaluate detector output against synthetic ground truth.

    Returns the shared detection metrics plus matching details
    used for diagnostics and plotting.

    Notes
    -----
    Ground-truth indices and detected indices must already use the same
    coordinate system. If synthetic data is split before detection,
    convert full-series ground-truth indices to detector-stream-relative
    indices before calling this function.
    """
    result = _match_detections(
        detected_changepoints=detected_changepoints,
        true_changepoints=true_changepoints,
        n_observations=n_observations,
        drift_type=drift_type,
        tolerance=tolerance,
        gradual_grace=gradual_grace,
    )

    if result["delays"]:
        detection_delay = float(
            pd.Series(
                result["delays"],
                dtype=float,
            ).mean()
        )
    else:
        detection_delay = float("nan")

    false_alarms_per_10000 = float(
        len(result["false_alarm_indices"]) / n_observations * 10000
    )

    return {
        "detection_delay": detection_delay,
        "false_alarms_per_10000": false_alarms_per_10000,
        "missed_detections": int(result["missed_true_events"]),
        "matched_pairs": result["matched_pairs"],
        "false_alarm_indices": result["false_alarm_indices"],
        "delays": result["delays"],
    }

REGIME_PRE_DRIFT_DAYS = 7
REGIME_POST_DRIFT_DAYS = 7

# Global regime only: added to the end of each event's drift window, so
# the week after an event still counts as drift before post_drift begins.
DRIFT_END_BUFFER = pd.Timedelta(days=7)

POINT_WINDOW = pd.Timedelta(days=7)
INTERVAL_GRACE = pd.Timedelta(days=7)

# Detections within this time after a kept detection are "Ignored" and never
# matched: match_unmatch debounces the raw stream before matching.
REFRACTORY_PERIOD = pd.Timedelta(days=14)

_REQUIRED_EVENT_COLUMNS = ("event_id", "start_date", "end_date", "date_precision", "region")


def build_event_windows(
    events,
    region,
    pre_drift_days=REGIME_PRE_DRIFT_DAYS,
    post_drift_days=REGIME_POST_DRIFT_DAYS,
):
    """
    Build pre-drift, drift, and post-drift windows for documented events.

    Parameters
    ----------
    events : pandas.DataFrame
        Must contain columns: event_id, start_date, end_date,
        date_precision, region.
    region : str
        Filter events to this region before building windows.
    pre_drift_days : int
        Days before event start to begin pre-drift window.
    post_drift_days : int
        Days after event end to extend post-drift window.

    Returns
    -------
    pandas.DataFrame
        One row per event with columns: event_id, start_date, end_date,
        drift_start, drift_end, pre_drift_start, post_drift_end,
        date_precision.
    """
    df = events[events["region"].isin(["NEM", region])].copy()
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    df = df.sort_values("start_date").reset_index(drop=True)

    df["drift_start"] = df["start_date"]
    df["drift_end"] = df["end_date"] + pd.Timedelta(days=1)
    df["pre_drift_start"] = df["start_date"] - pd.Timedelta(days=pre_drift_days)
    df["post_drift_end"] = (df["drift_end"] + pd.Timedelta(days=post_drift_days))

    return df[
        [
            "event_id",
            "start_date",
            "end_date",
            "drift_start",
            "drift_end",
            "pre_drift_start",
            "post_drift_end",
            "date_precision",
        ]
    ]


# assign_detection_with_regime (retired 2026-10-15; no table used it) is in git history.


def assign_timestamp_global_regime(
    timestamps,
    event_windows,
    drift_end_buffer=DRIFT_END_BUFFER,
):
    """
    Assign one global regime and event to every test timestamp (official rule).

    pre_drift: before the first test event. drift: inside an event's window
    [drift_start, drift_end + drift_end_buffer). post_drift: after that window
    until the next event. Overlaps: drift goes to the earliest event (by
    drift_start, event_id), post_drift to the most recently started one.

    Parameters
    ----------
    timestamps : array-like of datetime-like
        Test-split timestamps.
    event_windows : pandas.DataFrame
        Output of build_event_windows().
    drift_end_buffer : pandas.Timedelta
        Added to the end of every drift window (default 7 days).

    Returns
    -------
    pandas.DataFrame
        One row per unique timestamp, sorted: timestamp, regime, event_id.
    """
    required_columns = {"event_id", "drift_start", "drift_end"}

    missing_columns = required_columns - set(event_windows.columns)
    if missing_columns:
        raise ValueError(
            "event_windows is missing columns: "
            f"{sorted(missing_columns)}"
        )

    timestamps = (
        pd.DatetimeIndex(
            pd.to_datetime(list(timestamps))
        )
        .unique()
        .sort_values()
    )

    windows = event_windows.copy()

    for column in ("drift_start", "drift_end"):
        windows[column] = pd.to_datetime(windows[column])

    windows["drift_end"] = windows["drift_end"] + pd.Timedelta(drift_end_buffer)

    windows = (
        windows
        .sort_values(["drift_start", "event_id"])
        .reset_index(drop=True)
    )

    if len(timestamps) == 0:
        return pd.DataFrame(columns=["timestamp", "regime", "event_id"])

    # Global rule for the test split: only events whose drift window
    # overlaps the supplied timestamps count. Events wholly before or
    # after the split are ignored.
    windows = windows[
        (windows["drift_end"] > timestamps[0])
        & (windows["drift_start"] <= timestamps[-1])
    ]

    if windows.empty:
        raise ValueError(
            "no event's drift window overlaps the timestamps; "
            "every timestamp needs a regime and an event."
        )

    # pre_drift: everything before the first event in the split.
    regime = np.full(len(timestamps), "pre_drift", dtype=object)
    event_id = np.full(len(timestamps), windows.iloc[0]["event_id"], dtype=object)

    # post_drift: in start order, so the most recently started event
    # whose drift window has ended is written last.
    for window in windows.itertuples():
        ended = timestamps >= window.drift_end
        regime[ended] = "post_drift"
        event_id[ended] = window.event_id

    # drift outranks post_drift; in reverse start order, so the earliest
    # event is written last.
    for window in windows.iloc[::-1].itertuples():
        inside = (
            (timestamps >= window.drift_start)
            & (timestamps < window.drift_end)
        )
        regime[inside] = "drift"
        event_id[inside] = window.event_id

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "regime": regime,
            "event_id": event_id,
        },
        columns=["timestamp", "regime", "event_id"],
    )


def assign_event_local_regime(timestamps, event_windows):
    """
    Assign an event-local regime to test timestamps near an event (comparison rule).

    Uses build_event_windows' three windows, no drift-end buffer:
    pre_drift [pre_drift_start, drift_start), drift [drift_start, drift_end),
    post_drift [drift_end, post_drift_end). Timestamps outside every window are
    left out. Overlaps follow the global rule: drift > pre_drift > post_drift;
    earliest event for drift/pre_drift, most recent for post_drift.

    Parameters
    ----------
    timestamps : array-like of datetime-like
        Test-split timestamps.
    event_windows : pandas.DataFrame
        Output of build_event_windows().

    Returns
    -------
    pandas.DataFrame
        One row per labelled timestamp, sorted: timestamp, regime, event_id.
    """
    required_columns = {
        "event_id",
        "pre_drift_start",
        "drift_start",
        "drift_end",
        "post_drift_end",
    }

    missing_columns = required_columns - set(event_windows.columns)
    if missing_columns:
        raise ValueError(
            "event_windows is missing columns: "
            f"{sorted(missing_columns)}"
        )

    timestamps = (
        pd.DatetimeIndex(
            pd.to_datetime(list(timestamps))
        )
        .unique()
        .sort_values()
    )

    windows = event_windows.copy()

    for column in ("pre_drift_start", "drift_start", "drift_end", "post_drift_end"):
        windows[column] = pd.to_datetime(windows[column])

    windows = (
        windows
        .sort_values(["drift_start", "event_id"])
        .reset_index(drop=True)
    )

    if len(timestamps) > 0:
        # Only events in the test split count, as in the global rule.
        windows = windows[
            (windows["drift_end"] > timestamps[0])
            & (windows["drift_start"] <= timestamps[-1])
        ]

    regime = np.full(len(timestamps), None, dtype=object)
    event_id = np.full(len(timestamps), None, dtype=object)

    def label(name, start_column, end_column, frame):
        for window in frame.itertuples():
            inside = (
                (timestamps >= getattr(window, start_column))
                & (timestamps < getattr(window, end_column))
            )
            regime[inside] = name
            event_id[inside] = window.event_id

    # Lowest priority first, so higher priorities overwrite. Within a
    # regime the winning event is written last.
    label("post_drift", "drift_end", "post_drift_end", windows)
    label("pre_drift", "pre_drift_start", "drift_start", windows.iloc[::-1])
    label("drift", "drift_start", "drift_end", windows.iloc[::-1])

    labelled = pd.notna(regime)

    return pd.DataFrame(
        {
            "timestamp": timestamps[labelled],
            "regime": regime[labelled],
            "event_id": event_id[labelled],
        },
        columns=["timestamp", "regime", "event_id"],
    )


def calculate_match_window(
    start_date,
    end_date,
    date_precision,
    point_window=POINT_WINDOW,
    interval_grace=INTERVAL_GRACE,
):
    """
    Match window of one documented event: [start, end), left inclusive,
    right exclusive.

    The event's end date counts as a whole day, so an event from s to e
    covers [s 00:00, (e+1) 00:00); the tolerance is added after that.

    - Single-date event (date_precision == day):
      [start, start + 1 day + point_window).
    - Long event (date_precision != day):
      [start, end + 1 day + interval_grace).

    Shared by `match_unmatch` and the T2 chance baseline, so both use the
    same window.

    Returns
    -------
    tuple of pandas.Timestamp
        (window_start, window_end), window_end exclusive.
    """
    start = pd.Timestamp(start_date)

    if str(date_precision).lower() == "day":
        end = start + pd.Timedelta(days=1) + point_window
    else:
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1) + interval_grace

    return start, end


def match_unmatch(
    detected_timestamps,
    events,
    region,
    point_window=POINT_WINDOW,
    interval_grace=INTERVAL_GRACE,
    refractory_period=REFRACTORY_PERIOD,
):
    """
    Classify each detection as Match, Unmatch or Ignored.

    1. Debounce: keep the earliest detection and label every later one within
       `refractory_period` of it Ignored; the next detection after that is kept,
       and so on through the stream.
    2. Match kept detections to events one-to-one: Tier 1 events first, then
       Tier 2, each chronologically; an event takes the first unused detection
       in its window (calculate_match_window), so pre-event detections never
       match. Kept detections left over are Unmatch.

    Parameters
    ----------
    detected_timestamps : array-like of datetime-like
        Detector output timestamps.
    events : pandas.DataFrame
        event_id, start_date, end_date, date_precision, region, and optionally
        tier (1 or 2; every event is Tier 2 without it).
    region : str
        Region filter (NEM events always included).
    point_window, interval_grace : pandas.Timedelta
        Match-window tolerances for single-date and long events.
    refractory_period : pandas.Timedelta
        Debounce window (default 14 days).

    Returns
    -------
    pandas.DataFrame
        timestamp, label, event_id, tier (1/2 for a Match), delay_days; Tier 1
        matches first, then Tier 2, then Unmatch and Ignored rows.
    """
    missing = [c for c in _REQUIRED_EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"events is missing columns: {missing}")

    if refractory_period < pd.Timedelta(0):
        raise ValueError("refractory_period must be non-negative.")

    catalogue = events[events["region"].isin(["NEM", region])].copy()
    catalogue["start_date"] = pd.to_datetime(catalogue["start_date"])
    catalogue["end_date"] = pd.to_datetime(catalogue["end_date"])

    if (catalogue["end_date"] < catalogue["start_date"]).any():
        raise ValueError("an event has end_date before start_date")

    if "tier" in catalogue.columns:
        tiers = pd.to_numeric(catalogue["tier"], errors="raise")
        if tiers.isna().any() or not tiers.isin([1, 2]).all():
            raise ValueError("Event tiers must be 1 or 2.")
        catalogue["tier"] = tiers.astype(int)
    else:
        catalogue["tier"] = 2

    catalogue = catalogue.sort_values(
        ["tier", "start_date", "event_id"]
    ).reset_index(drop=True)

    detected_raw = pd.DatetimeIndex(pd.to_datetime(list(detected_timestamps))).sort_values()

    kept_timestamps = []
    ignored_list = []
    refractory_until = None
    for timestamp in detected_raw:
        if refractory_until is not None and timestamp < refractory_until:
            ignored_list.append(timestamp)
        else:
            kept_timestamps.append(timestamp)
            refractory_until = timestamp + refractory_period

    detected = pd.DatetimeIndex(kept_timestamps)
    ignored_timestamps = pd.DatetimeIndex(ignored_list)

    used = [False] * len(detected)
    results = []

    for event in catalogue.itertuples(index=False):
        start, window_end = calculate_match_window(
            event.start_date,
            event.end_date,
            event.date_precision,
            point_window=point_window,
            interval_grace=interval_grace,
        )

        hit = next(
            (
                i
                for i, timestamp in enumerate(detected)
                if not used[i] and start <= timestamp < window_end
            ),
            None,
        )
        if hit is not None:
            used[hit] = True
            delay = (detected[hit] - start) / pd.Timedelta(days=1)
            results.append(
                {
                    "timestamp": detected[hit],
                    "label": "Match",
                    "event_id": event.event_id,
                    "tier": event.tier,
                    "delay_days": delay,
                }
            )

    for i, timestamp in enumerate(detected):
        if not used[i]:
            results.append(
                {
                    "timestamp": timestamp,
                    "label": "Unmatch",
                    "event_id": None,
                    "tier": None,
                    "delay_days": float("nan"),
                }
            )

    for timestamp in ignored_timestamps:
        results.append(
            {
                "timestamp": timestamp,
                "label": "Ignored",
                "event_id": None,
                "tier": None,
                "delay_days": float("nan"),
            }
        )

    return pd.DataFrame(
        results,
        columns=[
            "timestamp",
            "label",
            "event_id",
            "tier",
            "delay_days",
        ],
    ).sort_values(
        ["tier", "timestamp"], na_position="last"
    ).reset_index(drop=True)


# evaluate_aemo_detections (retired 2026-10-15; T2 calls match_unmatch and
# calculate_event_metrics directly) is in git history.


def calculate_event_metrics(match_results, events, region):
    """
    Calculate overall documented-event matching metrics.

    Overall `precision` is matched / effective detections (effective
    excludes rows labelled "Ignored" by the refractory period -- see
    REFRACTORY_PERIOD / match_unmatch -- since those are treated as
    repeats of an already-credited drift, not independent detections).

    `precision_t1` and `precision_t2` use the same pooled denominator as
    the overall `precision`: each tier's own match count divided by all
    effective (accepted) detections, i.e. `precision_t1 = n_matched_t1 /
    n_effective`. This means a detection matched to the other tier still
    counts in a tier's own denominator even though it isn't a hit for
    that tier -- both `precision_t1` and `precision_t2` are therefore
    bounded by the overall `precision` and share its `nan` condition
    (zero effective detections).
    """
    matched = match_results[
        match_results["label"] == "Match"
    ]

    unmatched = match_results[
        match_results["label"] == "Unmatch"
    ]

    ignored = match_results[
        match_results["label"] == "Ignored"
    ]

    catalogue = events[
        events["region"].isin(["NEM", region])
    ]

    n_events = len(catalogue)
    n_total = len(match_results)
    n_effective = len(matched) + len(unmatched)

    n_matched_t1 = matched.loc[
        matched["tier"] == 1,
        "event_id",
    ].nunique()

    n_matched_t2 = matched.loc[
        matched["tier"] == 2,
        "event_id",
    ].nunique()

    n_matched_events = n_matched_t1 + n_matched_t2

    return {
        "n_total_detections": n_total,
        "n_ignored_detections": len(ignored),
        "n_effective_detections": n_effective,
        "n_matched_t1": n_matched_t1,
        "n_matched_t2": n_matched_t2,
        "n_unmatched_events": n_events - n_matched_events,
        "n_matched_detections": len(matched),
        "n_unmatched_detections": len(unmatched),
        "mean_delay_days": (
            float(matched["delay_days"].mean())
            if len(matched) > 0
            else float("nan")
        ),
        "precision": (
            len(matched) / n_effective
            if n_effective > 0
            else float("nan")
        ),
        "precision_t1": (
            n_matched_t1 / n_effective
            if n_effective > 0
            else float("nan")
        ),
        "precision_t2": (
            n_matched_t2 / n_effective
            if n_effective > 0
            else float("nan")
        ),
        "event_recall": (
            n_matched_events / n_events
            if n_events > 0
            else float("nan")
        ),
    }

# Adaptation (Step 5) and interval (Step 6) metrics


def calculate_absolute_error(y_true, y_pred):
    """Point-wise |y_true - y_pred| for per-timestamp curve files."""

    y_true = pd.Series(y_true, dtype=float).reset_index(drop=True)
    y_pred = pd.Series(y_pred, dtype=float).reset_index(drop=True)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError("y_true and y_pred must not contain missing values.")

    return (y_true - y_pred).abs()


FORECAST_REGIMES = ("pre_drift", "drift", "post_drift")


def calculate_forecast_metrics_by_regime(y_true, y_pred, regime_labels):
    """
    MAE and observation count per regime (T3), pooled over every timestamp
    in the regime.

    `regime_labels` must be aligned to the forecast timestamps; "pre_drift"
    and "pre-drift" are both accepted. Returns {regime: {"mae",
    "n_observations"}} keyed pre_drift / drift / post_drift; empty regimes
    are omitted.
    """

    y_true = pd.Series(y_true, dtype=float).reset_index(drop=True)
    y_pred = pd.Series(y_pred, dtype=float).reset_index(drop=True)
    labels = pd.Series(regime_labels).reset_index(drop=True)

    if not (len(y_true) == len(y_pred) == len(labels)):
        raise ValueError("y_true, y_pred and regime_labels must have the same length.")

    if labels.isna().any():
        raise ValueError("regime_labels must not contain missing values.")

    labels = labels.astype(str).str.replace("-", "_", regex=False)
    unexpected = set(labels.unique()) - set(FORECAST_REGIMES)
    if unexpected:
        raise ValueError(
            f"regime_labels contains unsupported regimes: {sorted(unexpected)}."
        )

    metrics = {}
    for regime in FORECAST_REGIMES:
        mask = (labels == regime).to_numpy()
        if not mask.any():
            continue
        metrics[regime] = {
            "mae": calculate_mae(y_true[mask], y_pred[mask]),
            "n_observations": int(mask.sum()),
        }
    return metrics


def calculate_adaptation_gain(arm_drift_mae, arm_a_drift_mae, n_retrains):
    """
    Drift-MAE difference vs Arm A (T4): arm - arm A, negative = better,
    plus the difference per retrain (NaN with no retrains). Pair arm and
    Arm A on region and seed first.
    """

    try:
        arm_drift_mae = float(arm_drift_mae)
        arm_a_drift_mae = float(arm_a_drift_mae)
    except (TypeError, ValueError) as exc:
        raise ValueError("MAE values must be finite, non-negative numbers.") from exc

    if (
        not math.isfinite(arm_drift_mae)
        or not math.isfinite(arm_a_drift_mae)
        or arm_drift_mae < 0
        or arm_a_drift_mae < 0
    ):
        raise ValueError("MAE values must be finite, non-negative numbers.")

    if isinstance(n_retrains, bool):
        raise TypeError("n_retrains must be a non-negative integer.")

    try:
        retrain_count = int(n_retrains)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("n_retrains must be a non-negative integer.") from exc

    if retrain_count != n_retrains or retrain_count < 0:
        raise ValueError("n_retrains must be a non-negative integer.")

    difference = arm_drift_mae - arm_a_drift_mae
    difference_per_retrain = (
        difference / retrain_count if retrain_count > 0 else float("nan")
    )

    return {
        "drift_mae_difference_vs_arm_a": float(difference),
        "drift_mae_difference_per_retrain": float(difference_per_retrain),
    }


def calculate_pinball_loss(y_true, y_quantile, quantile):
    """
    Mean pinball loss of one quantile forecast.

    An infinite quantile forecast (a conformal interval with an infinite
    radius) is valid input and makes the loss infinite; it is never
    capped or turned into NaN.
    """

    y_true = pd.Series(y_true, dtype=float)
    y_quantile = pd.Series(y_quantile, dtype=float)

    if len(y_true) != len(y_quantile):
        raise ValueError("y_true and y_quantile must have the same length.")

    if y_true.isna().any() or y_quantile.isna().any():
        raise ValueError("y_true and y_quantile must not contain missing values.")

    if not 0 < quantile < 1:
        raise ValueError("quantile must be strictly between 0 and 1.")

    error = y_true.to_numpy() - y_quantile.to_numpy()
    losses = np.where(error >= 0, quantile * error, (quantile - 1) * error)
    return float(np.mean(losses))


def calculate_interval_pinball_loss(y_true, lower, upper, alpha=0.10):
    """
    Pinball loss of a (1 - alpha) interval: mean of the q = alpha/2 and
    q = 1 - alpha/2 losses (0.05 / 0.95 for a 90% interval).
    """

    y_true = pd.Series(y_true, dtype=float)
    lower = pd.Series(lower, dtype=float)
    upper = pd.Series(upper, dtype=float)

    if len(y_true) != len(lower) or len(y_true) != len(upper):
        raise ValueError("y_true, lower, and upper must have the same length.")

    if y_true.isna().any() or lower.isna().any() or upper.isna().any():
        raise ValueError("y_true, lower, and upper must not contain missing values.")

    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1.")

    if (lower.to_numpy() > upper.to_numpy()).any():
        raise ValueError("lower bounds must not exceed upper bounds.")

    lower_loss = calculate_pinball_loss(y_true, lower, alpha / 2)
    upper_loss = calculate_pinball_loss(y_true, upper, 1 - alpha / 2)
    return float((lower_loss + upper_loss) / 2)

# Step 6: uncertainty evaluation metrics
# emperical coverage, mean interval width, normalised interval width, worst 24-hour coverage

def _validate_interval_inputs(
    lower_bound,
    upper_bound,
):
    """
    Validate prediction-interval bounds and return float Series.

    Infinite bounds are valid (a conformal interval with an infinite
    radius); missing values are not. Both Series are re-indexed 0..n-1 so
    the metrics compare by position, not by index label.
    """
    lower_bound = pd.Series(lower_bound, dtype=float).reset_index(drop=True)
    upper_bound = pd.Series(upper_bound, dtype=float).reset_index(drop=True)

    if len(lower_bound) != len(upper_bound):
        raise ValueError(
            "lower_bound and upper_bound must have the same length."
        )

    if len(lower_bound) == 0:
        raise ValueError(
            "Uncertainty evaluation inputs must not be empty."
        )

    if lower_bound.isna().any() or upper_bound.isna().any():
        raise ValueError(
            "Uncertainty evaluation inputs must not contain missing values."
        )

    if (lower_bound > upper_bound).any():
        raise ValueError(
            "lower_bound must not exceed upper_bound."
        )

    return lower_bound, upper_bound


def _interval_covered(
    y_true,
    lower_bound,
    upper_bound,
):
    """
    Per-observation coverage flags (bool Series), both boundaries included.
    """
    y_true = pd.Series(y_true, dtype=float).reset_index(drop=True)
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )

    if len(y_true) != len(lower_bound):
        raise ValueError(
            "y_true, lower_bound, and upper_bound "
            "must have the same length."
        )

    if not np.isfinite(y_true.to_numpy()).all():
        raise ValueError(
            "y_true must contain only finite values."
        )

    return (y_true >= lower_bound) & (y_true <= upper_bound)


def calculate_empirical_coverage(
    y_true,
    lower_bound,
    upper_bound,
):
    """
    Calculate empirical prediction-interval coverage.

    Coverage is the proportion of observed values lying inside the
    supplied prediction interval, including both interval boundaries.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    Returns
    -------
    float
        Fraction of observations covered by the interval.
    """
    covered = _interval_covered(y_true, lower_bound, upper_bound)

    return float(covered.mean())


def calculate_mean_interval_width(
    lower_bound,
    upper_bound,
):
    """
    Calculate mean prediction-interval width.

    Parameters
    ----------
    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    Returns
    -------
    float
        Mean interval width.
    """
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )

    return float(
        (upper_bound - lower_bound).mean()
    )


def calculate_width_scale(y_train):
    """
    Denominator for the normalised interval width: mean actual demand over
    the TRAIN window.

    Compute it once per region from TRAIN only and pass the same value to
    every method and regime; never recompute it on calibration, test or a
    single regime.

    Parameters
    ----------
    y_train : array-like
        Actual values in the TRAIN window. Missing values are ignored.

    Returns
    -------
    float
        Mean of the TRAIN actuals.
    """
    y_train = pd.Series(y_train, dtype=float).dropna()

    if y_train.empty:
        raise ValueError("y_train must contain at least one value.")

    scale = float(y_train.mean())

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("width scale must be a positive finite number.")

    return scale


def calculate_normalised_interval_width(
    lower_bound,
    upper_bound,
    scale,
):
    """
    Calculate mean prediction-interval width divided by a fixed scale.

    NormalisedMeanWidth = mean(upper - lower) / scale. For AEMO the scale
    is the region's mean TRAIN demand (`calculate_width_scale`); for the
    synthetic simulation it is the fixed value 10.

    Parameters
    ----------
    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    scale : float
        Positive denominator, shared by every method and regime.

    Returns
    -------
    float
        Normalised mean interval width.
    """
    if isinstance(scale, bool) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a positive finite number.")

    return calculate_mean_interval_width(lower_bound, upper_bound) / float(scale)


def calculate_rolling_coverage(
    y_true,
    lower_bound,
    upper_bound,
    window=48,
):
    """
    Calculate rolling empirical coverage.

    Default:
        48 half-hour intervals = 24 hours (window=8 gives 4 hours).

    Pass the full chronological stream, not one regime's rows, so every
    window covers consecutive timestamps.

    Parameters
    ----------
    y_true : array-like
        Observed values.

    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    window : int
        Number of observations in the rolling window.

    Returns
    -------
    pd.Series
        Coverage of the window ending at each observation; NaN until the
        first full window.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer.")

    covered = _interval_covered(y_true, lower_bound, upper_bound)

    return covered.astype(float).rolling(window=window).mean()


HALF_HOUR = pd.Timedelta(minutes=30)


def _eligible_window_ends(
    length,
    window=48,
    regime_labels=None,
    regime=None,
    timestamps=None,
    interval=HALF_HOUR,
):
    """
    Boolean mask over observations: True where the `window` observations
    ending there form an eligible window.

    A window is eligible when it is complete (`window` observations), when
    every observation carries `regime` (if `regime_labels` is given), and
    when its timestamps are consecutive -- the first and last are exactly
    (window - 1) * interval apart -- so a window never spans a gap (if
    `timestamps` is given).
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer.")

    if (regime_labels is None) != (regime is None):
        raise ValueError("regime_labels and regime must be given together.")

    eligible = pd.Series(np.arange(length) >= window - 1)

    if regime_labels is not None:
        labels = pd.Series(regime_labels).reset_index(drop=True)

        if len(labels) != length:
            raise ValueError(
                "regime_labels must have the same length as y_true."
            )

        if labels.isna().any():
            raise ValueError("regime_labels must not contain missing values.")

        labels = labels.astype(str).str.replace("-", "_", regex=False)
        in_regime = labels == str(regime).replace("-", "_")
        eligible &= in_regime.astype(float).rolling(window=window).sum() == window

    if timestamps is not None:
        stamps = pd.Series(pd.to_datetime(timestamps)).reset_index(drop=True)

        if len(stamps) != length:
            raise ValueError("timestamps must have the same length as y_true.")

        span = stamps - stamps.shift(window - 1)
        eligible &= span == (window - 1) * pd.Timedelta(interval)

    return eligible


def calculate_complete_window_count(
    length,
    window=48,
    regime_labels=None,
    regime=None,
    timestamps=None,
):
    """
    Number of eligible rolling windows (T5 "complete 24-hour windows").

    Uses exactly the eligibility rule of `calculate_worst_rolling_coverage`:
    complete, inside one regime, consecutive timestamps. For one continuous
    segment of n observations this is max(0, n - window + 1), summed over
    segments. Windows overlap, so the count is not a number of independent
    days.

    Parameters
    ----------
    length : int
        Number of observations in the full chronological stream.
    window, regime_labels, regime, timestamps
        As in `calculate_worst_rolling_coverage`.

    Returns
    -------
    int
    """
    return int(
        _eligible_window_ends(
            length,
            window=window,
            regime_labels=regime_labels,
            regime=regime,
            timestamps=timestamps,
        ).sum()
    )


def calculate_worst_rolling_coverage(
    y_true,
    lower_bound,
    upper_bound,
    window=48,
    regime_labels=None,
    regime=None,
    timestamps=None,
):
    """
    Calculate the worst (lowest) rolling coverage (T5 "worst 24h coverage").

    With `regime_labels` and `regime`, only windows lying entirely inside
    that regime count, so the by-regime value never mixes regimes or joins
    two separate events. Rolling is still done on the full stream.

    Parameters
    ----------
    y_true : array-like
        Observed values, full chronological stream.

    lower_bound : array-like
        Lower prediction-interval bounds.

    upper_bound : array-like
        Upper prediction-interval bounds.

    window : int
        Number of observations in the rolling window (48 = 24 hours).

    regime_labels : array-like, optional
        Regime label per observation; "pre_drift" and "pre-drift" are both
        accepted.

    regime : str, optional
        Regime to report. Required together with `regime_labels`.

    timestamps : array-like of datetime-like, optional
        Timestamp per observation. When given, a window counts only if its
        timestamps are consecutive half-hours, so it never spans a gap.

    Returns
    -------
    float
        Lowest window coverage; NaN when no full window qualifies.
    """
    rolling = calculate_rolling_coverage(
        y_true,
        lower_bound,
        upper_bound,
        window=window,
    )

    eligible = _eligible_window_ends(
        len(rolling),
        window=window,
        regime_labels=regime_labels,
        regime=regime,
        timestamps=timestamps,
    )
    rolling = rolling[eligible.to_numpy()].dropna()

    if rolling.empty:
        return float("nan")

    return float(rolling.min())


def calculate_uq_metrics_by_regime(
    y_true,
    lower_bound,
    upper_bound,
    regime_labels,
    window=48,
    width_scale=None,
    timestamps=None,
):
    """
    Coverage, mean interval width and worst rolling coverage per regime (T5).

    Pass the full chronological stream. Coverage and width are pooled over
    every timestamp of the regime; the worst rolling coverage only counts
    windows lying entirely inside it (NaN when none does). `regime_labels`
    follows `calculate_forecast_metrics_by_regime`: "pre_drift" and
    "pre-drift" are both accepted and empty regimes are omitted. With
    `width_scale` (see `calculate_width_scale`), each regime also gets a
    "normalised_interval_width" over that same fixed denominator. With
    `timestamps`, windows spanning a gap in timestamps are not eligible.
    "n_complete_windows" counts the eligible windows behind the worst
    rolling coverage (`calculate_complete_window_count`).

    Returns
    -------
    dict
        {regime: {"coverage", "mean_interval_width",
        ["normalised_interval_width",] "worst_rolling_coverage",
        "n_complete_windows", "n_observations"}} keyed pre_drift / drift /
        post_drift.
    """
    y_true = pd.Series(y_true, dtype=float).reset_index(drop=True)
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )
    labels = pd.Series(regime_labels).reset_index(drop=True)

    if not (len(y_true) == len(lower_bound) == len(labels)):
        raise ValueError(
            "y_true, lower_bound, upper_bound and regime_labels "
            "must have the same length."
        )

    if labels.isna().any():
        raise ValueError("regime_labels must not contain missing values.")

    labels = labels.astype(str).str.replace("-", "_", regex=False)
    unexpected = set(labels.unique()) - set(FORECAST_REGIMES)
    if unexpected:
        raise ValueError(
            f"regime_labels contains unsupported regimes: {sorted(unexpected)}."
        )

    metrics = {}
    for regime in FORECAST_REGIMES:
        mask = (labels == regime).to_numpy()
        if not mask.any():
            continue
        metrics[regime] = {
            "coverage": calculate_empirical_coverage(
                y_true[mask], lower_bound[mask], upper_bound[mask]
            ),
            "mean_interval_width": calculate_mean_interval_width(
                lower_bound[mask], upper_bound[mask]
            ),
            "worst_rolling_coverage": calculate_worst_rolling_coverage(
                y_true,
                lower_bound,
                upper_bound,
                window=window,
                regime_labels=labels,
                regime=regime,
                timestamps=timestamps,
            ),
            "n_complete_windows": calculate_complete_window_count(
                len(labels),
                window=window,
                regime_labels=labels,
                regime=regime,
                timestamps=timestamps,
            ),
            "n_observations": int(mask.sum()),
        }
        if width_scale is not None:
            metrics[regime]["normalised_interval_width"] = (
                calculate_normalised_interval_width(
                    lower_bound[mask], upper_bound[mask], width_scale
                )
            )
    return metrics


# Step 6: escalation metrics (T6, F5)
# review rate, retained coverage, escalation curve, review rate for a target
# coverage, coverage at a review budget
#
# A forecast is escalated when its interval width is strictly greater than
# the threshold tau -- the rule the UQ classes apply to their own escalate
# flag (uncertainty.conformal_common.escalation_flags, AgACI.predict). Here
# it is evaluated at many tau over intervals already issued: escalation
# hands a forecast to a human and never feeds back into the UQ state, so the
# bounds do not depend on tau.

# Widths are rounded to this many decimals (MW) before ranking, so float
# noise in a recomputed upper - lower (about 1e-12 MW for a constant-width
# method) is not mistaken for distinct operating points. Well below the
# smallest real AgACI width step (about 1e-3 MW).
ESCALATION_WIDTH_DECIMALS = 6
ESCALATION_TARGET_COVERAGES = (0.90, 0.95)
ESCALATION_REVIEW_BUDGETS = (0.05, 0.10, 0.20)
ESCALATION_CURVE_COLUMNS = [
    "threshold",
    "review_rate",
    "retained_coverage",
    "n_retained",
    "n_reviewed",
    "n_observations",
]


def _escalation_widths(
    lower_bound,
    upper_bound,
    decimals=ESCALATION_WIDTH_DECIMALS,
):
    """
    Interval widths rounded to `decimals`, so tied widths compare equal.
    """
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )

    return np.round((upper_bound - lower_bound).to_numpy(), decimals)


def _escalation_scores(
    lower_bound,
    upper_bound,
    score=None,
    decimals=ESCALATION_WIDTH_DECIMALS,
):
    """
    The escalation score of every forecast, rounded to `decimals` so tied
    scores compare equal: the interval width by default, or `score` (one
    finite value per forecast, e.g. `calculate_nonconformity_score`).
    """
    if score is None:
        return _escalation_widths(lower_bound, upper_bound, decimals)

    score = np.asarray(score, dtype=float)
    if score.shape != (len(pd.Series(lower_bound)),):
        raise ValueError("score must have one value per forecast.")
    if not np.isfinite(score).all():
        raise ValueError("score must contain only finite values.")

    return np.round(score, decimals)


def calculate_nonconformity_score(
    y_true,
    y_pred,
    lookback,
    history_true=None,
    history_pred=None,
):
    """
    Escalation score from past nonconformity: the mean absolute error
    |y - y_hat| over the `lookback` half-hours before each forecast.

    Forecast t only sees errors up to t-1, so the score is known when
    forecast t is issued (intervals are issued one half-hour ahead and each
    outcome is observed before the next). `lookback=1` is the previous
    half-hour's error alone.

    Parameters
    ----------
    y_true, y_pred : array-like
        Observed values and point forecasts of the scored stream.

    lookback : int
        Number of past half-hours averaged.

    history_true, history_pred : array-like, optional
        The observations and forecasts immediately before the stream (e.g.
        the Calibration window), carried in so the first forecasts have a
        full look-back. Without them, forecasts with fewer than `lookback`
        past errors get NaN.

    Returns
    -------
    numpy.ndarray
        One score per forecast of the stream.
    """
    if isinstance(lookback, bool) or not isinstance(lookback, (int, np.integer)) or lookback < 1:
        raise ValueError("lookback must be a positive integer.")

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length.")

    n_history = 0
    if history_true is not None or history_pred is not None:
        history_true = np.asarray(history_true, dtype=float)
        history_pred = np.asarray(history_pred, dtype=float)
        if history_true.shape != history_pred.shape:
            raise ValueError("history_true and history_pred must have the same length.")
        n_history = len(history_true)
        y_true = np.r_[history_true, y_true]
        y_pred = np.r_[history_pred, y_pred]

    errors = pd.Series(np.abs(y_true - y_pred))
    score = errors.rolling(window=lookback).mean().shift(1)

    return score.to_numpy()[n_history:]


def calculate_review_rate(
    lower_bound,
    upper_bound,
    threshold,
    decimals=ESCALATION_WIDTH_DECIMALS,
):
    """
    Fraction of forecasts escalated for review at width threshold `threshold`.

    A forecast is escalated when its interval width is strictly greater
    than `threshold`.

    Returns
    -------
    float
        Review rate in [0, 1].
    """
    width = _escalation_widths(lower_bound, upper_bound, decimals)

    return float((width > threshold).mean())


def calculate_retained_coverage(
    y_true,
    lower_bound,
    upper_bound,
    threshold,
    decimals=ESCALATION_WIDTH_DECIMALS,
):
    """
    Empirical coverage among the forecasts kept (width <= `threshold`).

    Returns
    -------
    float
        Retained coverage; NaN when every forecast is escalated (an empty
        retained set has no coverage, not 100%).
    """
    covered = _interval_covered(y_true, lower_bound, upper_bound).to_numpy()
    kept = _escalation_widths(lower_bound, upper_bound, decimals) <= threshold

    if not kept.any():
        return float("nan")

    return float(covered[kept].mean())


def calculate_escalation_curve(
    y_true,
    lower_bound,
    upper_bound,
    thresholds=None,
    decimals=ESCALATION_WIDTH_DECIMALS,
    score=None,
):
    """
    Review rate and retained coverage at every score threshold (F5).

    A forecast is escalated when its score is strictly greater than the
    threshold. The score is the interval width by default (the rule the UQ
    classes apply to their own flag); pass `score` to sweep another signal,
    e.g. `calculate_nonconformity_score`. Coverage is always judged against
    the intervals as issued.

    Parameters
    ----------
    y_true, lower_bound, upper_bound : array-like
        Observed values and the intervals as issued.

    thresholds : array-like, optional
        Score thresholds to evaluate. Default: every distinct (rounded)
        score. -inf (review everything) is always added. The largest score
        is the no-review point. Pass a shared grid to compare regimes at
        the same thresholds.

    decimals : int
        Rounding applied to the scores before ranking; tied scores are
        reviewed together.

    score : array-like, optional
        One finite escalation score per forecast; default the width.

    Returns
    -------
    pandas.DataFrame
        One row per threshold, ascending: threshold, review_rate,
        retained_coverage (NaN when nothing is retained), n_retained,
        n_reviewed, n_observations.
    """
    covered = _interval_covered(y_true, lower_bound, upper_bound).to_numpy()
    scores = _escalation_scores(lower_bound, upper_bound, score, decimals)

    # Cumulative counts over the scores in ascending order: `positions[i]`
    # forecasts have score <= thresholds[i] and are kept.
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    cumulative_hits = np.r_[0, np.cumsum(covered[order])]

    if thresholds is None:
        thresholds = np.unique(sorted_scores)
    thresholds = np.unique(np.r_[-np.inf, np.asarray(thresholds, dtype=float)])
    n_retained = np.searchsorted(sorted_scores, thresholds, side="right")

    n_observations = len(scores)
    retained_coverage = np.divide(
        cumulative_hits[n_retained],
        n_retained,
        out=np.full(len(n_retained), np.nan),
        where=n_retained > 0,
    )

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "review_rate": (n_observations - n_retained) / n_observations,
            "retained_coverage": retained_coverage,
            "n_retained": n_retained,
            "n_reviewed": n_observations - n_retained,
            "n_observations": n_observations,
        },
        columns=ESCALATION_CURVE_COLUMNS,
    )


def calculate_escalation_curves_by_regime(
    y_true,
    lower_bound,
    upper_bound,
    regime_labels,
    decimals=ESCALATION_WIDTH_DECIMALS,
    score=None,
):
    """
    Escalation curve within each regime, on one shared threshold grid (F5).

    Pass the full chronological stream (and, with `score`, one score per
    forecast of it; default the interval width). The grid is every
    distinct score of the full stream plus -inf; each regime's rates use
    that regime's forecasts only. `regime_labels` follows
    `calculate_forecast_metrics_by_regime`: "pre_drift" and "pre-drift"
    are both accepted and empty regimes are omitted.

    Returns
    -------
    pandas.DataFrame
        The curves stacked, with a leading "regime" column keyed
        pre_drift / drift / post_drift.
    """
    y_true = pd.Series(y_true, dtype=float).reset_index(drop=True)
    lower_bound, upper_bound = _validate_interval_inputs(
        lower_bound,
        upper_bound,
    )
    labels = pd.Series(regime_labels).reset_index(drop=True)

    if not (len(y_true) == len(lower_bound) == len(labels)):
        raise ValueError(
            "y_true, lower_bound, upper_bound and regime_labels "
            "must have the same length."
        )

    if labels.isna().any():
        raise ValueError("regime_labels must not contain missing values.")

    labels = labels.astype(str).str.replace("-", "_", regex=False)
    unexpected = set(labels.unique()) - set(FORECAST_REGIMES)
    if unexpected:
        raise ValueError(
            f"regime_labels contains unsupported regimes: {sorted(unexpected)}."
        )

    scores = _escalation_scores(lower_bound, upper_bound, score, decimals)
    grid = np.unique(scores)
    curves = []
    for regime in FORECAST_REGIMES:
        mask = (labels == regime).to_numpy()
        if not mask.any():
            continue
        curve = calculate_escalation_curve(
            y_true[mask],
            lower_bound[mask],
            upper_bound[mask],
            thresholds=grid,
            decimals=decimals,
            score=scores[mask],
        )
        curve.insert(0, "regime", regime)
        curves.append(curve)

    return pd.concat(curves, ignore_index=True)


def calculate_review_rate_for_coverage(
    escalation_curve,
    target_coverage,
    min_retained=1,
):
    """
    Smallest review rate whose retained coverage reaches `target_coverage` (T6).

    Only thresholds keeping at least `min_retained` forecasts count, so a
    handful of retained forecasts cannot reach the target by chance.

    Returns
    -------
    float
        The review rate; 0.0 when no review already reaches the target; NaN
        when no threshold does (report "unreachable").
    """
    if min_retained < 1:
        raise ValueError(
            "min_retained must be at least 1: an empty retained set has no coverage."
        )

    reaching = escalation_curve[
        (escalation_curve["retained_coverage"] >= target_coverage)
        & (escalation_curve["n_retained"] >= min_retained)
    ]
    if reaching.empty:
        return float("nan")

    return float(reaching["review_rate"].min())


def calculate_coverage_at_review_rate(
    escalation_curve,
    review_budget,
):
    """
    Retained coverage at a review budget (T6 "coverage @ 5%" etc.).

    Uses the largest attainable review rate not above `review_budget`; the
    threshold is chosen by review rate, never by which coverage is best.
    The no-review point always qualifies.

    Returns
    -------
    tuple of float
        (retained coverage, the review rate actually used).
    """
    within = escalation_curve[escalation_curve["review_rate"] <= review_budget]
    row = within.loc[within["review_rate"].idxmax()]

    return float(row["retained_coverage"]), float(row["review_rate"])


def calculate_escalation_operating_points(
    escalation_curve,
    target_coverages=ESCALATION_TARGET_COVERAGES,
    review_budgets=ESCALATION_REVIEW_BUDGETS,
    min_retained=1,
):
    """
    T6 operating points of one escalation curve (one run, one regime).

    Returns
    -------
    dict
        "review_rate_for_coverage_<target>" per target (NaN = unreachable);
        "coverage_at_review_<budget>" and "actual_review_at_<budget>" per
        budget; "n_operating_points", the number of distinct review rates
        on the curve. Targets and budgets are formatted to two decimals,
        e.g. review_rate_for_coverage_0.90, coverage_at_review_0.05.
    """
    points = {}
    for target in target_coverages:
        points[f"review_rate_for_coverage_{target:.2f}"] = (
            calculate_review_rate_for_coverage(
                escalation_curve, target, min_retained
            )
        )
    for budget in review_budgets:
        coverage, review_rate = calculate_coverage_at_review_rate(
            escalation_curve, budget
        )
        points[f"coverage_at_review_{budget:.2f}"] = coverage
        points[f"actual_review_at_{budget:.2f}"] = review_rate
    points["n_operating_points"] = int(escalation_curve["review_rate"].nunique())

    return points


# Detection periodicity (validation): are alarms tied to the clock or calendar
# rather than to change? Alarm times are in days unless stated.

def calculate_alarm_gaps(alarm_days):
    """
    Days between consecutive alarms (sorted); empty for fewer than two.
    """
    days = np.sort(np.asarray(alarm_days, dtype=float))
    return np.diff(days)


def calculate_gap_cv(alarm_days):
    """
    Coefficient of variation (population sd / mean) of the gaps between
    alarms: ~1 for alarms at random times, well below 1 for evenly spaced
    (periodic) alarms, above 1 for clustered alarms. NaN with fewer than
    two gaps.
    """
    gaps = calculate_alarm_gaps(alarm_days)
    if len(gaps) < 2 or gaps.mean() == 0:
        return float("nan")
    return float(gaps.std() / gaps.mean())


def calculate_gap_share(alarm_days, low, high):
    """
    Share of gaps between alarms in [low, high) days; NaN without gaps.
    """
    gaps = calculate_alarm_gaps(alarm_days)
    if len(gaps) == 0:
        return float("nan")
    return float(((gaps >= low) & (gaps < high)).mean())


def calculate_phase_concentration(timestamps, period_days, origin):
    """
    Rayleigh test: do alarms fall at the same phase of a cycle?

    Each alarm's phase is its time since `origin`, modulo `period_days`
    (e.g. 7 with a Monday origin for day of week; 365.2425 with a 1 January
    origin for day of year). Returns (R, p): R is the mean resultant length
    (0 = spread evenly, 1 = all at one phase) and p the Rayleigh test p-value
    for uniform phases (Zar's approximation). (NaN, NaN) for fewer than three
    alarms.
    """
    times = pd.DatetimeIndex(pd.to_datetime(list(timestamps)))
    n = len(times)
    if n < 3:
        return float("nan"), float("nan")

    days = (times - pd.Timestamp(origin)) / pd.Timedelta(days=1)
    angles = 2 * np.pi * np.mod(np.asarray(days, dtype=float), period_days) / period_days
    resultant = float(np.abs(np.exp(1j * angles).mean()))
    z = n * resultant**2
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n**2 - n * z)) - (1 + 2 * n))
    return resultant, float(min(max(p, 0.0), 1.0))


def simulate_random_alarm_days(n_alarms, span_days, min_gap_days=0.0, n_sim=5000, seed=0):
    """
    Random-timing baseline: `n_sim` sequences of `n_alarms` alarm times in
    [0, span_days], uniform over every arrangement at least `min_gap_days`
    apart (the refractory debounce). Returns an (n_sim, n_alarms) array of
    sorted days.
    """
    if n_alarms < 1:
        raise ValueError("n_alarms must be positive.")
    free = span_days - (n_alarms - 1) * min_gap_days
    if free < 0:
        raise ValueError("n_alarms alarms do not fit in span_days at min_gap_days apart.")

    rng = np.random.default_rng(seed)
    base = np.sort(rng.uniform(0, free, size=(n_sim, n_alarms)), axis=1)
    return base + np.arange(n_alarms) * min_gap_days
