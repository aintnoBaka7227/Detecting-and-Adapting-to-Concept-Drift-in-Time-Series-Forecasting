import pytest

from drift_forecasting.synthetic.generator import make_series


@pytest.mark.skip(reason="unskip once make_series() is implemented — Sprint 1, Step 2")
def test_no_drift_series_has_no_changepoints():
    y, changepoints = make_series(kind="none", n=5000, noise=1.0, seed=0)
    assert changepoints == []
