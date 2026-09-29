"""T3/T4 evaluation helpers and record_run's regime / retrain logging."""

import math

import numpy as np
import pandas as pd
import pytest

from drift_lab.evaluation import (
    calculate_absolute_error,
    calculate_adaptation_gain,
    calculate_forecast_metrics_by_regime,
    calculate_interval_pinball_loss,
    calculate_pinball_loss,
)
from experiments import results_io
from experiments.run_harness import record_run


@pytest.fixture
def results_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results_io, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(results_io, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def test_forecast_metrics_by_regime_pools_errors_and_skips_empty():
    metrics = calculate_forecast_metrics_by_regime(
        [10, 10, 10, 10], [11, 13, 10, 6], ["pre_drift", "pre_drift", "drift", "drift"]
    )
    assert metrics == {
        "pre_drift": {"mae": 2.0, "n_observations": 2},
        "drift": {"mae": 2.0, "n_observations": 2},
    }
    assert calculate_absolute_error([10, 10], [11, 7]).tolist() == [1, 3]


def test_forecast_metrics_by_regime_accepts_hyphenated_labels():
    metrics = calculate_forecast_metrics_by_regime([1, 2], [2, 2], ["pre-drift", "post-drift"])
    assert set(metrics) == {"pre_drift", "post_drift"}


@pytest.mark.parametrize(
    "labels",
    [["drift"], ["drift", None], ["drift", "stable"]],
)
def test_forecast_metrics_by_regime_rejects_bad_labels(labels):
    with pytest.raises(ValueError):
        calculate_forecast_metrics_by_regime([1, 2], [1, 2], labels)


def test_adaptation_gain_sign_and_per_retrain():
    assert calculate_adaptation_gain(40.0, 50.0, 4) == {
        "drift_mae_difference_vs_arm_a": -10.0,
        "drift_mae_difference_per_retrain": -2.5,
    }
    gain = calculate_adaptation_gain(50.0, 50.0, 0)
    assert gain["drift_mae_difference_vs_arm_a"] == 0.0
    assert math.isnan(gain["drift_mae_difference_per_retrain"])


@pytest.mark.parametrize(
    ("arm", "arm_a", "n", "error"),
    [
        (float("nan"), 1.0, 1, ValueError),
        (-1.0, 1.0, 1, ValueError),
        (1.0, 1.0, 2.5, ValueError),
        (1.0, 1.0, -1, ValueError),
        (1.0, 1.0, True, TypeError),
    ],
)
def test_adaptation_gain_rejects_invalid_input(arm, arm_a, n, error):
    with pytest.raises(error):
        calculate_adaptation_gain(arm, arm_a, n)


def test_pinball_loss_matches_definition():
    # y=10: q=0.9 forecast 8 -> under by 2 -> 0.9*2; forecast 12 -> over by 2 -> 0.1*2
    assert calculate_pinball_loss([10, 10], [8, 12], 0.9) == pytest.approx((1.8 + 0.2) / 2)
    assert calculate_interval_pinball_loss([10], [8], [12], alpha=0.10) == pytest.approx(
        (0.05 * 2 + 0.05 * 2) / 2
    )
    assert calculate_pinball_loss([10, 12], [11, 11], 0.5) == pytest.approx(0.5)  # MAE / 2
    with pytest.raises(ValueError):
        calculate_pinball_loss([1], [1], 1.0)
    with pytest.raises(ValueError):
        calculate_interval_pinball_loss([1], [2], [1])  # lower above upper


def test_record_run_with_regime_labels_logs_regimes_curve_and_retrains(results_in_tmp):
    n = 1000
    index = pd.date_range("2020-03-01", periods=n, freq="30min")
    labels = np.array(["pre_drift"] * 300 + ["drift"] * 300 + ["post_drift"] * 400)
    retrains = [pd.Timestamp("2020-03-05 23:30")]

    rows = record_run(
        method="nhits/C_drift_full_history",
        dataset="aemo",
        region="SA1",
        seed=1,
        config={"arm": "C_drift_full_history"},
        wall_clock_s=2.0,
        split_id="test_v1",
        n_retrains=1,
        forecast=(np.full(n, 100.0), np.full(n, 90.0), index),
        regime_labels=labels,
        retrain_timestamps=retrains,
    )

    mae = rows[rows["metric_name"] == "mae"].set_index("regime")["metric_value"]
    assert mae.to_dict() == {"full": 10.0, "pre-drift": 10.0, "drift": 10.0, "post-drift": 10.0}
    counts = rows[rows["metric_name"] == "n_observations"].set_index("regime")["metric_value"]
    assert counts.to_dict() == {"pre-drift": 300, "drift": 300, "post-drift": 400}

    chash = rows["config_hash"].iloc[0]
    curve = pd.read_csv(results_io.curve_path(chash, "aemo", "SA1", 1))
    assert list(curve.columns) == [
        "timestamp", "actual", "forecast", "absolute_error", "rolling_mae_7d", "regime", "arm", "seed",
    ]
    saved = pd.read_csv(results_io.retrains_path(chash, "aemo", "SA1", 1), parse_dates=["timestamp"])
    assert list(saved["timestamp"]) == retrains


def test_record_run_rejects_changepoints_with_regime_labels(results_in_tmp):
    index = pd.date_range("2020-03-01", periods=10, freq="30min")
    with pytest.raises(ValueError):
        record_run(
            method="m", dataset="aemo", seed=None, config={}, wall_clock_s=0.0, split_id="t",
            forecast=(np.ones(10), np.ones(10), index), changepoints=[5], regime_labels=["drift"] * 10,
        )
