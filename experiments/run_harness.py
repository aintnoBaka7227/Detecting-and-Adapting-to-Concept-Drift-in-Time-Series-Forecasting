"""`record_run` — turn one completed run into results/runs.csv rows.

It does not run the model. The caller (`run_*.py`) fits/predicts or detects,
then hands the arrays here; this module only routes them through
``drift_lab.evaluation`` and appends the schema rows.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd

from drift_lab import evaluation
from drift_lab.config import REGIME_DRIFT_MARGIN
from experiments import results_io

ROLLING_WINDOW = 48 * 7  # 7 days of half-hourly observations
DETECTION_TOLERANCE = 48 * 7
REGIME_NAMES = ("pre-drift", "drift", "post-drift")


def config_of(obj) -> dict:
    """The public (non-underscore) constructor attributes of a model / detector."""
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def record_run(
    *,
    method: str,
    dataset: str,
    seed: int | None,
    config: dict,
    wall_clock_s: float,
    split_id: str,
    region: str | None = None,
    train_samples: int = 0,
    n_retrains: int = 0,
    forecast: tuple | None = None,
    detection: tuple | None = None,
    changepoints: list[int] | None = None,
    point_tolerance: pd.Timedelta | None = None,
    period_grace: pd.Timedelta | None = None,
    samples_per_year: float | None = None,
    test_period_days: float | None = None,
) -> pd.DataFrame:
    """Append one method's metrics to runs.csv and return the rows.

    Pass exactly one of:
      forecast  = (y_true, y_pred, index)        -> group "baseline"
      detection = (detected, truth, n_samples)   -> group "detection"
        `truth` a list of int indices  -> synthetic; `dataset` must be
          "synthetic_<drift_type>" and evaluation.evaluate_detections scores it.
        `truth` a DataFrame            -> AEMO documented-event matching
          (Table T2); `detected` is timestamps, the frame is the event
          catalogue filtered to the tier/region being scored.
        `truth=None`                   -> no scoring, just log `n_detections`.

    `point_tolerance` / `period_grace`, documented-event matching only:
    override `evaluation.match_unmatch`'s defaults
    (`POINT_WINDOW` / `INTERVAL_GRACE`). Leave unset
    for the frozen Table T2 run — the supervisor's rule there is to fix the
    matching tolerance *before* seeing results, not loosen it after. An
    override belongs to an explicitly exploratory script that says so in
    its own docstring; it lands in the run's `config`, so it still gets its
    own `config_hash` and never silently mixes with the frozen tolerance's
    rows.

    `seed=None` for a deterministic run (no stochasticity to average over) —
    written as NaN, never a fabricated seed value.

    `split_id` identifies the data-partitioning scheme the arrays came from
    (e.g. "aemo_frozen_v1" for config.SPLIT, "synth_n20000" for a generator
    config) — not "this batch of rows" (that's dataset+method+config_hash).
    Required, no default: a generic default is exactly how a synthetic run
    ends up mislabelled with an AEMO-shaped split_id.

    `changepoints`, forecast runs only: splits the metrics into
    pre-drift / drift / post-drift rows instead of one `regime="full"` set.
    Not ground truth — either the dataset's known changepoints (synthetic)
    or a detector's output (AEMO), passed by the caller.

    `samples_per_year`, synthetic detection runs only: when supplied, also
    logs `n_detections`, `n_false_alarms` and `false_alarms_per_year`
    (`n_false_alarms / n_samples * samples_per_year`) alongside the
    standard `detection_delay` / `false_alarms_per_10000` /
    `missed_detections` rows. Lets a tuning run pick the annualisation
    appropriate to the cadence it's calibrating for (e.g. 365 for a
    daily-aggregated deployment vs. 48*365 for half-hourly) instead of
    every caller sharing one fixed assumption. Omitted (the default),
    behaviour is unchanged from before this parameter existed.

    `test_period_days`, AEMO documented-event matching only: when
    supplied, also logs `n_effective_detections` (accepted = Match +
    Unmatch, excluding refractory-suppressed `Ignored` rows) and
    `accepted_detections_per_year` (`n_effective_detections /
    (test_period_days / 365)`). Omitted (the default), behaviour is
    unchanged from before this parameter existed.
    """
    if (forecast is None) == (detection is None):
        raise ValueError("pass exactly one of forecast= / detection=")

    chash = results_io.config_hash(config)
    results_io.dump_config(chash, config)  # config_hash -> actual values, for produce_*.py
    common = {
        "method": method,
        "dataset": dataset,
        "region": region or "-",
        "seed": seed,
        "config_hash": chash,
        "split_id": split_id,
        "n_retrains": n_retrains,
        "train_samples": train_samples,
        "wall_clock_s": wall_clock_s,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    if forecast is not None:
        rows = build_forecast_rows(dataset, seed, region, forecast, changepoints, common)
    else:
        rows = build_detection_rows(
            dataset,
            detection,
            common,
            region,
            point_tolerance,
            period_grace,
            samples_per_year,
            test_period_days,
        )

    results_io.append_runs(rows)
    return pd.DataFrame(rows, columns=results_io.RUN_COLUMNS)


def build_forecast_rows(dataset, seed, region, forecast, changepoints, common) -> list[dict]:
    y_true, y_pred, index = forecast
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    curve = evaluation.calculate_rolling_mae(y_true, y_pred, window=ROLLING_WINDOW)
    curve = pd.Series(curve.to_numpy(), index=pd.Index(index, name="index"))
    common = {**common, "group": "baseline"}
    results_io.dump_curve(common["config_hash"], dataset, region, seed, curve)
    curve_values = curve.to_numpy()

    if not changepoints:
        return [
            build_row(common, "mae", evaluation.calculate_mae(y_true, y_pred), "full"),
            build_row(common, "rolling_mae_7d_mean", np.nanmean(curve_values), "full"),
            build_row(common, "rolling_mae_7d_max", np.nanmax(curve_values), "full"),
        ]

    labels = label_regimes(changepoints, len(y_true), REGIME_DRIFT_MARGIN)
    rows = []
    for regime in REGIME_NAMES:
        mask = labels == regime
        if not mask.any():
            continue
        rows.append(build_row(common, "mae", evaluation.calculate_mae(y_true[mask], y_pred[mask]), regime))
        segment = curve_values[mask]
        if np.isfinite(segment).any():
            rows.append(build_row(common, "rolling_mae_7d_mean", np.nanmean(segment), regime))
            rows.append(build_row(common, "rolling_mae_7d_max", np.nanmax(segment), regime))
    return rows


def build_detection_rows(
    dataset,
    detection,
    common,
    region,
    point_tolerance=None,
    period_grace=None,
    samples_per_year=None,
    test_period_days=None,
) -> list[dict]:
    detected, truth, n_samples = detection
    common = {**common, "group": "detection"}

    if isinstance(truth, pd.DataFrame):
        # AEMO: `truth` is a documented-event catalogue (the caller filtered
        # it to the tier + region being scored), `detected` is timestamps.
        return build_documented_event_rows(
            detected, truth, common, region, point_tolerance, period_grace, test_period_days
        )

    if truth is None:
        # No ground truth and no event catalogue: just log the count.
        return [build_row(common, "n_detections", len(detected), "full")]

    if not dataset.startswith("synthetic_"):
        raise ValueError("detection runs with ground truth expect dataset='synthetic_<drift_type>'")
    drift_type = dataset.removeprefix("synthetic_")

    result = evaluation.evaluate_detections(
        detected_changepoints=detected,
        true_changepoints=truth,
        n_observations=n_samples,
        drift_type=drift_type,
        tolerance=DETECTION_TOLERANCE,
    )
    rows = [
        build_row(common, name, result[name], "full")
        for name in ("detection_delay", "false_alarms_per_10000", "missed_detections")
    ]

    if samples_per_year is not None:
        n_false_alarms = len(result["false_alarm_indices"])
        rows.append(build_row(common, "n_detections", len(detected), "full"))
        rows.append(build_row(common, "n_false_alarms", n_false_alarms, "full"))
        rows.append(
            build_row(
                common,
                "false_alarms_per_year",
                n_false_alarms / n_samples * samples_per_year,
                "full",
            )
        )

    return rows


def build_documented_event_rows(
    detected_timestamps,
    events,
    common,
    region,
    point_tolerance=None,
    period_grace=None,
    test_period_days=None,
) -> list[dict]:
    """AEMO documented-event matching (Table T2). The events are NOT ground
    truth — an unmatched detection is `unmatched`, never a false positive.
    Raw timestamps are dumped for the figure / re-derivation; runs.csv gets
    every scalar `calculate_event_metrics` produces (including the
    Tier 1/Tier 2 breakdown and `event_recall`) plus one `delay_<event_id>`
    row per match."""
    window_kwargs = {}
    if point_tolerance is not None:
        window_kwargs["point_window"] = point_tolerance
    if period_grace is not None:
        window_kwargs["interval_grace"] = period_grace

    match_results = evaluation.match_unmatch(detected_timestamps, events, region, **window_kwargs)
    metrics = evaluation.calculate_event_metrics(match_results, events, region)

    results_io.dump_detections(
        common["config_hash"], "aemo", common["region"], common["seed"], detected_timestamps
    )
    rows = [
        build_row(common, "n_detections", metrics["n_total_detections"], "full"),
        build_row(common, "n_ignored_detections", metrics["n_ignored_detections"], "full"),
        build_row(common, "n_matched", metrics["n_matched_detections"], "full"),
        build_row(common, "n_matched_t1", metrics["n_matched_t1"], "full"),
        build_row(common, "n_matched_t2", metrics["n_matched_t2"], "full"),
        build_row(common, "n_unmatched_events", metrics["n_unmatched_events"], "full"),
        build_row(common, "n_unmatched_detections", metrics["n_unmatched_detections"], "full"),
        build_row(common, "precision", metrics["precision"], "full"),
        build_row(common, "precision_t1", metrics["precision_t1"], "full"),
        build_row(common, "precision_t2", metrics["precision_t2"], "full"),
        build_row(common, "event_recall", metrics["event_recall"], "full"),
        build_row(common, "mean_delay_days", metrics["mean_delay_days"], "full"),
    ]

    if test_period_days is not None:
        n_effective = metrics["n_effective_detections"]
        rows.append(build_row(common, "n_effective_detections", n_effective, "full"))
        rows.append(
            build_row(
                common,
                "accepted_detections_per_year",
                n_effective / (test_period_days / 365.0),
                "full",
            )
        )

    matched = match_results[match_results["label"] == "Match"]
    rows += [
        build_row(common, f"delay_{row.event_id}", row.delay_days, "full")
        for row in matched.itertuples()
    ]
    return rows


def label_regimes(changepoints, n: int, margin: int) -> np.ndarray:
    """pre-drift before the first changepoint; drift = [cp, cp+margin) at
    each; post-drift everywhere else after the first. Point changepoints —
    an interval-shaped truth (e.g. gradual's [start, end]) is treated as two
    point events for now."""
    labels = np.full(n, "pre-drift", dtype=object)
    labels[min(changepoints):] = "post-drift"
    for cp in changepoints:
        labels[cp : min(n, cp + margin)] = "drift"
    return labels


def build_row(common: dict, metric_name: str, metric_value, regime: str) -> dict:
    return {
        **common,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "regime": regime,
    }
