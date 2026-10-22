from __future__ import annotations

import pandas as pd

from drift_lab.evaluation.evaluation import (
    DRIFT_END_BUFFER,
    assign_event_local_regime,
    assign_timestamp_global_regime,
    build_event_windows,
)


def _windows():
    events = pd.DataFrame(
        {
            "event_id": ["E1"],
            "start_date": [pd.Timestamp("2021-02-01")],
            "end_date": [pd.Timestamp("2021-02-03")],
            "date_precision": ["day"],
            "region": ["SA1"],
        }
    )
    return build_event_windows(events, "SA1")


def _stamps():
    return pd.date_range("2021-01-01", "2021-04-01", freq="30min")


def test_global_rule_labels_every_timestamp_with_a_regime():
    out = assign_timestamp_global_regime(_stamps(), _windows())
    assert len(out) == len(_stamps())
    assert set(out["regime"]) == {"pre_drift", "drift", "post_drift"}


def test_global_rule_extends_drift_by_the_end_buffer():
    windows = _windows()
    out = assign_timestamp_global_regime(_stamps(), windows).set_index("timestamp")["regime"]
    drift_end = windows["drift_end"].iloc[0]
    assert out[drift_end + DRIFT_END_BUFFER - pd.Timedelta(minutes=30)] == "drift"
    assert out[drift_end + DRIFT_END_BUFFER] == "post_drift"
    assert out[windows["drift_start"].iloc[0] - pd.Timedelta(minutes=30)] == "pre_drift"


def test_event_local_rule_leaves_far_away_timestamps_unlabelled():
    out = assign_event_local_regime(_stamps(), _windows())
    assert 0 < len(out) < len(_stamps())
