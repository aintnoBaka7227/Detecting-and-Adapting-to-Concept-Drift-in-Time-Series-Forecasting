"""record_run: schema, config hash, regime split, curve dump."""

import numpy as np
import pandas as pd
import pytest

from drift_lab.forecasting.seasonal_naive import SeasonalNaive
from experiments import results_io
from experiments.run_harness import config_of, record_run


@pytest.fixture(autouse=True)
def results_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(results_io, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(results_io, "RUNS_CSV", tmp_path / "runs.csv")
    monkeypatch.setattr(results_io, "RUNS_DIR", tmp_path / "runs")
    return tmp_path


def build_forecast_arrays(n=800):
    index = pd.date_range("2020-03-01", periods=n, freq="30min")
    rng = np.random.default_rng(0)
    y_true = rng.normal(1000, 50, n)
    y_pred = y_true + rng.normal(0, 10, n)
    return y_true, y_pred, index


def test_config_of_keeps_only_public_attrs():
    model = SeasonalNaive(season_length=48).fit(
        pd.DataFrame(index=pd.date_range("2020", periods=3, freq="30min")),
        pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020", periods=3, freq="30min")),
    )
    assert config_of(model) == {"season_length": 48}


def test_forecast_run_writes_expected_rows(results_in_tmp):
    rows = record_run(
        method="seasonal_naive",
        dataset="aemo",
        region="SA1",
        seed=0,
        config={"season_length": 48},
        wall_clock_s=1.5,
        split_id="test_v1",
        train_samples=100,
        forecast=build_forecast_arrays(),
    )
    assert list(rows.columns) == results_io.RUN_COLUMNS
    assert set(rows["group"]) == {"baseline"}
    assert set(rows["regime"]) == {"full"}
    assert set(rows["metric_name"]) == {"mae", "rolling_mae_7d_mean", "rolling_mae_7d_max"}

    on_disk = pd.read_csv(results_io.RUNS_CSV)
    assert len(on_disk) == len(rows)
    curve = results_io.RUNS_DIR / rows["config_hash"].iloc[0] / "curve_aemo_SA1_0.csv"
    assert curve.exists()


def test_detection_run_writes_three_metrics():
    rows = record_run(
        method="adwin",
        dataset="synthetic_sudden",
        seed=1,
        config={"delta": 0.002},
        wall_clock_s=0.2,
        split_id="synth_n2000",
        detection=([120, 900], [100], 2000),
    )
    assert set(rows["group"]) == {"detection"}
    assert set(rows["metric_name"]) == {
        "detection_delay",
        "false_alarms_per_10000",
        "missed_detections",
    }
    assert set(rows["regime"]) == {"full"}


def test_forecast_with_changepoints_splits_by_regime():
    y_true, y_pred, index = build_forecast_arrays(n=3000)
    rows = record_run(
        method="seasonal_naive",
        dataset="aemo",
        region="SA1",
        seed=0,
        config={},
        wall_clock_s=1.0,
        split_id="test_v1",
        forecast=(y_true, y_pred, index),
        changepoints=[1500],
    )
    assert set(rows["regime"]) == {"pre-drift", "drift", "post-drift"}
    assert set(rows["metric_name"]) == {"mae", "rolling_mae_7d_mean", "rolling_mae_7d_max"}
    # one mae row per regime
    assert (rows["metric_name"] == "mae").sum() == 3


def test_aemo_detection_without_truth_logs_n_detections():
    rows = record_run(
        method="adwin",
        dataset="aemo",
        region="SA1",
        seed=0,
        config={"delta": 0.002},
        wall_clock_s=0.1,
        split_id="aemo_frozen_v1",
        detection=([100, 5000, 12000], None, 67249),
    )
    assert set(rows["group"]) == {"detection"}
    assert rows["metric_name"].tolist() == ["n_detections"]
    assert rows["metric_value"].iloc[0] == 3.0
    assert rows["regime"].iloc[0] == "full"


def test_detection_with_truth_requires_synthetic_dataset():
    with pytest.raises(ValueError):
        record_run(
            method="adwin", dataset="aemo", seed=0, config={}, wall_clock_s=0.1,
            split_id="aemo_frozen_v1", detection=([10], [5], 100),
        )


def test_deterministic_run_writes_nan_seed():
    rows = record_run(
        method="dhr_arima", dataset="aemo", region="SA1", seed=None,
        config={}, wall_clock_s=1.0, split_id="aemo_frozen_v1",
        forecast=build_forecast_arrays(),
    )
    assert rows["seed"].isna().all()


def test_config_hash_is_deterministic_and_sensitive():
    common = {
        "method": "m", "dataset": "aemo", "region": "SA1", "seed": 0,
        "wall_clock_s": 1.0, "split_id": "test_v1",
    }
    h1 = record_run(**common, config={"a": 1, "b": 2}, forecast=build_forecast_arrays())["config_hash"].iloc[0]
    h2 = record_run(**common, config={"b": 2, "a": 1}, forecast=build_forecast_arrays())["config_hash"].iloc[0]
    h3 = record_run(**common, config={"a": 9}, forecast=build_forecast_arrays())["config_hash"].iloc[0]
    assert h1 == h2 != h3


def test_requires_exactly_one_result_bundle():
    with pytest.raises(ValueError):
        record_run(method="m", dataset="aemo", seed=0, config={}, wall_clock_s=1.0, split_id="test_v1")
