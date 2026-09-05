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
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "y_true and y_pred must not contain missing values."
        )

    return float(
        (y_true - y_pred).abs().mean()
    )


def calculate_rolling_mae(
    y_true,
    y_pred,
    window=48 * 7
):
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
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if y_true.isna().any() or y_pred.isna().any():
        raise ValueError(
            "y_true and y_pred must not contain missing values."
        )

    absolute_error = (
        y_true - y_pred
    ).abs()

    return absolute_error.rolling(
        window=window
    ).mean()



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
            "drift_type must be 'none', 'sudden', "
            "'gradual', or 'recurring'."
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
            raise ValueError(
                "No-drift evaluation requires true_changepoints=[]"
            )

        return {
            "matched_pairs": [],
            "delays": [],
            "false_alarm_indices": detected,
            "missed_true_events": 0,
            "n_true_events": 0,
        }

    if drift_type == "sudden" and len(truth) != 1:
        raise ValueError(
            "Sudden drift requires exactly one true changepoint."
        )

    if drift_type == "gradual" and len(truth) != 2:
        raise ValueError(
            "Gradual drift requires [drift_start, drift_end]."
        )

    if drift_type == "recurring" and len(truth) != 2:
        raise ValueError(
            "Recurring drift requires exactly two true changepoints."
        )

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
            cp for cp in detected
            if cp in unused and drift_start <= cp <= match_end
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
                cp for cp in detected
                if cp in unused and true_cp <= cp <= match_end
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

    false_alarm_indices = [
        cp for cp in detected
        if cp in unused
    ]

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

    n_false_alarms = len(
        result["false_alarm_indices"]
    )

    return float(
        n_false_alarms
        / n_observations
        * 10000
    )


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

    return int(
        result["missed_true_events"]
    )


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
        len(result["false_alarm_indices"])
        / n_observations
        * 10000
    )

    return {
        "detection_delay": detection_delay,
        "false_alarms_per_10000": false_alarms_per_10000,
        "missed_detections": int(
            result["missed_true_events"]
        ),
        "matched_pairs": result["matched_pairs"],
        "false_alarm_indices": result["false_alarm_indices"],
        "delays": result["delays"],
    }


# Planned evaluation metrics
#
# Real-data detection evaluation:
# - documented-event matching
# - delay from documented event to matched detection
# - unmatched detections
# - documented events with no matched detection
#
# Uncertainty evaluation (Step 6):
# - empirical coverage
# - mean interval width
# - normalised interval width
# - worst 24-hour coverage
