"""
Shared evaluation metrics for the concept-drift forecasting pipeline.

This module provides common evaluation functions used across the
project so that forecasting, drift detection, adaptation, and
uncertainty components are assessed consistently.


CURRENT METRICS
---------------

Forecast evaluation:

    calculate_mae()
        Calculates Mean Absolute Error (MAE) between observed and
        predicted demand.

    calculate_rolling_mae()
        Calculates rolling MAE over the forecast stream.

        The default window is 336 observations:

            48 half-hourly observations/day × 7 days = 336 observations.

            Rolling MAE is used to track forecasting performance over time,
            including degradation of frozen baseline models and comparison
            of forecasting performance across the adaptation arms.


Synthetic drift-detection evaluation:

    calculate_detection_delay()
        Calculates mean delay between known synthetic drift events and
        successfully matched detector outputs.

    calculate_false_alarms_per_10000()
        Calculates unmatched detector outputs normalised per 10,000
        observations.

    calculate_missed_detections()
        Counts known synthetic drift events for which no detector output
        is successfully matched.

    evaluate_detections()
        Applies the shared synthetic event-matching protocol and returns
        the required detection metrics together with diagnostic matching
        information.


SYNTHETIC DETECTION EVALUATION
------------------------------
Synthetic data provide known ground-truth changepoints and therefore
support objective calculation of:

    - detection delay
    - false alarms per 10,000 observations
    - missed detections

The evaluation supports:

    - no drift
    - sudden drift
    - gradual drift
    - recurring drift

These metrics are used to validate detector behaviour on controlled
data before detectors are applied to real AEMO streams.


EVENT-MATCHING PROTOCOL
-----------------------
Detector outputs must be associated with known synthetic drift events
before delay, false alarms, and missed detections can be calculated.

For sudden and recurring drift, detections are matched using a
configurable post-changepoint tolerance.

The default tolerance is 336 observations:

    48 half-hourly observations/day × 7 days = 336 observations.

A detection is considered matched when it occurs at or after the true
changepoint and within the configured tolerance.

The first unused detection inside the matching window is assigned to
that drift event.

The 336-observation tolerance is an evaluation design choice used to
provide a fixed and reproducible event-association rule across
detectors and experiments.

It is not a detector threshold.

The tolerance determines whether a detection can be associated with an
event; it does not replace or redefine the measured detection delay.

For a successfully matched point event:

    detection_delay = detected_index - true_changepoint_index

If no detection occurs within the matching window, the true event is
counted as a missed detection.


GRADUAL DRIFT
-------------
Gradual drift is represented as one transition interval:

    [drift_start, drift_end]

rather than as two independent drift events.

A detection occurring from drift_start through drift_end can therefore
be associated with the same gradual-drift event.

A configurable grace period is also allowed after drift_end. If no
separate grace period is supplied, it defaults to the same value as the
matching tolerance.

Detection delay for a matched gradual event is measured from
drift_start.


RECURRING DRIFT
---------------
Recurring drift contains two true transition points, for example:

    A -> B -> A

Each transition is evaluated as a separate drift event.

A detector output can be matched to at most one true event.


NO-DRIFT CONTROL
----------------
The no-drift synthetic stream contains no true drift events.

Therefore:

    - every detector output is treated as a false alarm
    - detection delay is undefined
    - missed detections are zero

The no-drift control is used to verify that detectors do not raise
unnecessary alarms on a stable stream.


FALSE ALARMS
------------
Any synthetic detector output that is not matched to a known true drift
event is treated as a false alarm.

False alarms are normalised by stream length as:

    false_alarms_per_10000 =
        number_of_false_alarms
        / number_of_observations
        * 10000


REAL-DATA DETECTION EVALUATION
------------------------------
Real-data detection evaluation for AEMO SA1 and NSW1 is planned.

Unlike the synthetic benchmark, AEMO data do not provide ground-truth
changepoints.

Detected changepoints will therefore be compared with historically
documented events as contextual reference points rather than treated as
known true drift events.

Planned real-data detection outputs include:

    - delay from a documented event to a matched detection
    - documented events with no matched detection
    - detections unmatched to documented events

Unmatched AEMO detections will not automatically be classified as false
alarms because documented events are not exhaustive ground truth.

A separate documented-event matching rule will be defined for AEMO
evaluation rather than assuming that the synthetic 336-observation
matching tolerance is appropriate for real events.

AEMO detections are classified as Match, Unmatch, or Ignored. Before
event matching runs, the raw detection stream is debounced
chronologically by REFRACTORY_PERIOD (14 days): the earliest detection
is kept and starts a 14-day window, every later detection inside that
window is Ignored, and the next detection after the window elapses is
kept and starts the next window, repeating for the whole stream. This
applies to the raw stream generally, not only to detections following
a matched one. Only kept detections are matched against documented
events; Ignored detections are excluded from precision. Overall
precision is matched / effective (Match + Unmatch) detections.
precision_t1 and precision_t2 are each computed against only that
tier's own matches plus the unmatched detections -- never against the
other tier's matches -- so Tier 2, which has far more documented
events than Tier 1, cannot dilute Tier 1's precision by sharing one
pooled denominator.


INPUT VALIDATION
----------------
The implemented evaluation functions verify that:

    - y_true and y_pred have equal lengths
    - y_true and y_pred contain no missing values
    - changepoint indices are valid integer indices
    - changepoint indices fall within the evaluated stream
    - detected and true changepoints use the same coordinate system
    - synthetic drift types contain the expected number of true events


PIPELINE INTEGRATION
--------------------
This module provides shared metric implementations for the full
experimental pipeline.

Different stages use different subsets of these metrics:

    - Baseline degradation evaluation uses calculate_mae() and
      calculate_rolling_mae().

    - Synthetic drift-detection evaluation uses
      calculate_detection_delay(),
      calculate_false_alarms_per_10000(), and
      calculate_missed_detections() through the shared
      evaluate_detections() workflow.

    - Real AEMO detection evaluation will extend this module with
      documented-event matching because real data do not provide
      ground-truth changepoints.

    - Adaptation evaluation will reuse calculate_mae() and
      calculate_rolling_mae() from this module to compare forecasting
      performance across the four adaptation arms. Retraining count,
      training samples, and wall-clock cost are recorded by the
      experiment workflow rather than calculated as forecast-error
      metrics here.

    - Uncertainty evaluation will use the planned coverage and
      interval-width metrics implemented in this module.

Per-run evaluation metrics are calculated here. Aggregation across
seeds, including mean ± standard deviation where required, is performed
by the experiment workflow.

Experiment results are recorded separately through the shared
runs.csv logging component.

PLANNED METRICS
---------------
Real-data detection evaluation:

    - documented-event matching
    - delay from documented event to matched detection
    - unmatched detections
    - documented events with no matched detection


Uncertainty evaluation:

    - empirical coverage
    - mean interval width
    - normalised interval width
    - worst 24-hour coverage
"""

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

    Matching rules
    --------------
    none
        There are no true drift events. Every detection is a false alarm.

    sudden / recurring
        Each true point changepoint is matched to the first unused
        detection in [true_cp, true_cp + tolerance].

    gradual
        The generator returns [drift_start, drift_end] for one gradual
        transition window, not two separate drift events. The first unused
        detection in [drift_start, drift_end + gradual_grace] is matched
        to that single drift event. Delay is measured from drift_start.

    A detection can match at most one true drift event.

    Parameters
    ----------
    detected_changepoints : array-like
        Detector output indices, relative to the detector input stream.

    true_changepoints : array-like
        Ground-truth indices, expressed in the same index system as
        detected_changepoints.

    n_observations : int
        Number of observations in the detector input stream.

    drift_type : str
        One of: "none", "sudden", "gradual", "recurring".

    tolerance : int
        Maximum allowed post-changepoint delay for sudden and recurring
        point events. Default is 336 observations (7 days of half-hourly
        data). The tolerance is configurable.

    gradual_grace : int, optional
        Additional post-window allowance for gradual drift. If None,
        the same value as tolerance is used.

    Returns
    -------
    dict
        Matching details containing:
            matched_pairs
            delays
            false_alarm_indices
            missed_true_events
            n_true_events
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
POINT_WINDOW = pd.Timedelta(days=7)
INTERVAL_GRACE = pd.Timedelta(days=7)

# Mandatory waiting time after a matched drift detection. Any further
# detection that falls inside this window and does not itself match a
# documented event is treated as a repeat of the same already-credited
# drift rather than a fresh false alarm, and is labelled "Ignored"
# instead of "Unmatch". Detections still get first crack at matching a
# documented event before refractory suppression is applied, so a
# detection inside the window can still match a different, later event.
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


def assign_regime(detected_timestamps, event_windows):
    """
    Assign one regime and event ID to every detection timestamp.

    Priority:
        drift > pre_drift > event-linked post_drift
        > general post_drift > pre_drift fallback

    Attribution:
        - drift:
          Use the matching event ID.
        - pre_drift:
          Use the upcoming event ID.
        - event-linked post_drift:
          Use the event ID when the timestamp falls between that
          event's drift_end and post_drift_end.
        - general post_drift:
          If an earlier event exists but the timestamp falls outside
          every event window, use event_id="unassigned".
        - before all events:
          Use regime="pre_drift" and event_id="unassigned".

    When multiple windows overlap:
        - drift and pre_drift use the earliest event by drift_start;
        - post_drift uses the most recently started applicable event.
    """
    required_columns = {
        "event_id",
        "drift_start",
        "drift_end",
        "pre_drift_start",
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
            pd.to_datetime(list(detected_timestamps))
        )
        .unique()
        .sort_values()
    )

    windows = event_windows.copy()

    for column in (
        "drift_start",
        "drift_end",
        "pre_drift_start",
        "post_drift_end",
    ):
        windows[column] = pd.to_datetime(windows[column])

    windows = (
        windows
        .sort_values(["drift_start", "event_id"])
        .reset_index(drop=True)
    )

    records = []

    for timestamp in timestamps:
        # Priority 1: inside a documented drift window.
        drift_matches = windows[
            (windows["drift_start"] <= timestamp)
            & (timestamp < windows["drift_end"])
        ]

        if not drift_matches.empty:
            selected_event = drift_matches.iloc[0]

            regime = "drift"
            event_id = selected_event["event_id"]

        else:
            # Priority 2: inside an upcoming event's pre-drift window.
            pre_drift_matches = windows[
                (windows["pre_drift_start"] <= timestamp)
                & (timestamp < windows["drift_start"])
            ]

            if not pre_drift_matches.empty:
                selected_event = pre_drift_matches.iloc[0]

                regime = "pre_drift"
                event_id = selected_event["event_id"]

            else:
                # Priority 3: inside a bounded post-drift window.
                post_drift_matches = windows[
                    (windows["drift_end"] <= timestamp)
                    & (timestamp < windows["post_drift_end"])
                ]

                if not post_drift_matches.empty:
                    # Attribute overlapping post-drift windows to the
                    # most recently started applicable event.
                    selected_event = (
                        post_drift_matches
                        .sort_values(
                            ["drift_start", "event_id"],
                            ascending=[False, True],
                        )
                        .iloc[0]
                    )

                    regime = "post_drift"
                    event_id = selected_event["event_id"]

                # Priority 4: outside all event windows, but at least
                # one event has already occurred.
                elif (windows["drift_start"] <= timestamp).any():
                    regime = "post_drift"
                    event_id = "unassigned"

                # Priority 5: before every event and outside an
                # explicit pre-drift window.
                else:
                    regime = "pre_drift"
                    event_id = "unassigned"

        records.append(
            {
                "timestamp": timestamp,
                "regime": regime,
                "event_id": event_id,
            }
        )

    return pd.DataFrame(
        records,
        columns=["timestamp", "regime", "event_id"],
    )


def match_unmatch(
    detected_timestamps,
    events,
    region,
    point_window=POINT_WINDOW,
    interval_grace=INTERVAL_GRACE,
    refractory_period=REFRACTORY_PERIOD,
):
    """
    Classify each detection as Match, Unmatch, or Ignored.

    Matching rules:
    - Single-date event (date_precision == day): match window is
      [event_start, event_start + tolerance].
    - Long event (date_precision != day): match window is
      [event_start, event_end + tolerance].
    - Pre-event detections cannot match.
    - Tier 1 events are ranked above Tier 2 and are allocated before any
      Tier 2 event can take a detection. Matching within each tier is
      chronological. A detection that could match both a Tier 1 and a
      Tier 2 event is therefore always attributed to the Tier 1 event.
    - Matching is one-to-one: the first unused detection inside a window
      is assigned to that event.
    - One detection matches at most one event. Matched Tier 1 and Tier 2
      detections both count as Match.

    Before event matching runs, the raw detection stream is passed
    through a chronological refractory filter: the earliest detection
    is kept and starts a `refractory_period` window; every later
    detection that falls inside that window is labelled Ignored; the
    first detection after the window elapses is kept and starts the
    next window; and so on for the whole stream. Only surviving (kept)
    detections are eligible for event matching. This applies uniformly
    to the whole stream, not only to detections following a matched
    one -- two detections 3 days apart are debounced to one even if
    neither, either, or both would otherwise have matched a documented
    event.

    Output rows are tier-ranked: Tier 1 matches first, then Tier 2
    matches, then Unmatch rows, then Ignored rows.

    Parameters
    ----------
    detected_timestamps : array-like of datetime-like
        Detector output timestamps.
    events : pandas.DataFrame
        Must contain: event_id, start_date, end_date, date_precision,
        region. If it contains a tier column, tiers must be 1 or 2;
        otherwise every event is treated as Tier 2.
    region : str
        Filter events to this region.
    point_window : pandas.Timedelta
        Window for matching point events.
    interval_grace : pandas.Timedelta
        Grace period for matching interval events.
    refractory_period : pandas.Timedelta
        Mandatory waiting time after a kept detection during which
        further detections in the raw stream are ignored, regardless of
        whether they would otherwise match, before event matching runs.
        Default is 14 days.

    Returns
    -------
    pandas.DataFrame
        Columns: timestamp, label (Match/Unmatch/Ignored), event_id
        (event_id or None), tier (1 or 2 for a Match, None otherwise),
        delay_days (float or NaN). Rows are sorted by tier (Tier 1
        before Tier 2, Unmatch/Ignored last), then timestamp.
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
        if refractory_until is not None and timestamp <= refractory_until:
            ignored_list.append(timestamp)
        else:
            kept_timestamps.append(timestamp)
            refractory_until = timestamp + refractory_period

    detected = pd.DatetimeIndex(kept_timestamps)
    ignored_timestamps = pd.DatetimeIndex(ignored_list)

    used = [False] * len(detected)
    results = []

    for event in catalogue.itertuples(index=False):
        start = event.start_date
        if str(event.date_precision).lower() == "day":
            window_end = start + point_window
        else:
            window_end = event.end_date + interval_grace

        hit = next(
            (
                i
                for i, timestamp in enumerate(detected)
                if not used[i] and start <= timestamp <= window_end
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


def evaluate_aemo_detections(
    detected_timestamps,
    events,
    region,
    point_window=POINT_WINDOW,
    interval_grace=INTERVAL_GRACE,
    refractory_period=REFRACTORY_PERIOD,
    pre_drift_days=REGIME_PRE_DRIFT_DAYS,
    post_drift_days=REGIME_POST_DRIFT_DAYS,
):
    """
    Full evaluation pipeline: regime labels, match/unmatch, metrics.

    Parameters
    ----------
    detected_timestamps : array-like of datetime-like
        Detector output timestamps.
    events : pandas.DataFrame
        Full events catalogue with region column.
    region : str
        Filter events to this region.
    point_window : pandas.Timedelta
        Window for matching point events.
    interval_grace : pandas.Timedelta
        Grace period for matching interval events.
    refractory_period : pandas.Timedelta
        Mandatory waiting time after a kept detection during which
        further detections in the raw stream are ignored, regardless of
        whether they would otherwise match, before event matching runs.
        Default is 14 days.
    pre_drift_days : int
        Days before event for pre-drift window.
    post_drift_days : int
        Days after event for post-drift window.

    Returns
    -------
    dict
        match_results : pandas.DataFrame from match_unmatch()
        regime_labels : pandas.DataFrame from assign_regime()
        metrics : dict from calculate_event_metrics()
    """
    match_results = match_unmatch(
    detected_timestamps,
    events,
    region,
    point_window=point_window,
    interval_grace=interval_grace,
    refractory_period=refractory_period,
    )

    event_windows = build_event_windows(
        events,
        region,
        pre_drift_days,
        post_drift_days,
    )

    regime_results = assign_regime(
        detected_timestamps,
        event_windows,
    )

    event_metrics = calculate_event_metrics(
        match_results,
        events,
        region,
    )

    regime_metrics = calculate_regime_metrics(
        match_results,
        regime_results,
    )

    return {
        "match_results": match_results,
        "regime_results": regime_results,
        "event_metrics": event_metrics,
        "regime_metrics": regime_metrics,
    }


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

def calculate_regime_metrics(match_results, regime_results):
    """
    Calculate detection metrics separately for each regime.

    Overall `precision` is matched / effective detections per regime
    (effective excludes rows labelled "Ignored" by the refractory
    period -- see REFRACTORY_PERIOD / match_unmatch). `precision_t1`
    and `precision_t2` use the same pooled per-regime denominator as
    the overall `precision` -- each tier's own match count divided by
    that regime's effective detections (see calculate_event_metrics for
    the same convention at the whole-run level).

    Parameters
    ----------
    match_results : pandas.DataFrame
        Output from match_unmatch().
        Required columns:
            timestamp, label, tier

    regime_results : pandas.DataFrame
        Output from assign_regime().
        Required columns:
            timestamp, regime

    Returns
    -------
    dict
        Detection metrics for pre-drift, drift and post-drift.
    """
    required_match_columns = {"timestamp", "label", "tier"}
    required_regime_columns = {"timestamp", "regime"}

    missing_match = required_match_columns - set(match_results.columns)
    if missing_match:
        raise ValueError(
            f"match_results is missing columns: {sorted(missing_match)}"
        )

    missing_regime = required_regime_columns - set(regime_results.columns)
    if missing_regime:
        raise ValueError(
            f"regime_results is missing columns: {sorted(missing_regime)}"
        )

    # Combine each detection's matching result with its regime.
    audit = regime_results.merge(
        match_results,
        on="timestamp",
        how="left",
        validate="one_to_one",
    )

    regime_metrics = {}

    for regime in ("pre_drift", "drift", "post_drift"):
        rows = audit[audit["regime"] == regime]

        n_detections = len(rows)
        n_ignored = int((rows["label"] == "Ignored").sum())
        n_effective = n_detections - n_ignored

        tier1_matched = int(
            (
                (rows["label"] == "Match")
                & (rows["tier"] == 1)
            ).sum()
        )

        tier2_matched = int(
            (
                (rows["label"] == "Match")
                & (rows["tier"] == 2)
            ).sum()
        )

        n_matched = tier1_matched + tier2_matched
        n_unmatched = n_effective - n_matched

        regime_metrics[regime] = {
            "detections": n_detections,
            "ignored_detections": n_ignored,
            "tier1_matched": tier1_matched,
            "tier2_matched": tier2_matched,
            "matched_detections": n_matched,
            "unmatched_detections": n_unmatched,
            "precision": (
                n_matched / n_effective
                if n_effective > 0
                else float("nan")
            ),
            "precision_t1": (
                tier1_matched / n_effective
                if n_effective > 0
                else float("nan")
            ),
            "precision_t2": (
                tier2_matched / n_effective
                if n_effective > 0
                else float("nan")
            ),
        }

    return regime_metrics
# Planned evaluation metrics
#
# Uncertainty evaluation (Step 6):
# - empirical coverage
# - mean interval width
# - normalised interval width
# - worst 24-hour coverage
