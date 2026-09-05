"""Synthetic generator: changepoint ground truth + determinism."""

import numpy as np
import pytest

from drift_lab.synthetic.generator import make_series


def test_no_drift_series_has_no_changepoints():
    _y, changepoints = make_series(kind="none", n=5000, noise=1.0, seed=0)
    assert changepoints == []


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("sudden", [5000]),
        ("gradual", [5000, 6000]),  # [drift_start, drift_end], width = min(1000, n//4)
        ("recurring", [20000 // 3, (2 * 20000) // 3]),
    ],
)
def test_changepoint_positions(kind, expected):
    _y, changepoints = make_series(kind=kind, n=20000, noise=1.0, seed=1)
    assert changepoints == expected


def test_same_seed_reproduces_series_exactly():
    a, cps_a = make_series("sudden", n=4000, noise=1.0, seed=7)
    b, cps_b = make_series("sudden", n=4000, noise=1.0, seed=7)
    np.testing.assert_array_equal(a, b)
    assert cps_a == cps_b


def test_noise_does_not_move_changepoints():
    _a, cps_low = make_series("gradual", n=8000, noise=0.1, seed=3)
    _b, cps_high = make_series("gradual", n=8000, noise=5.0, seed=3)
    assert cps_low == cps_high
