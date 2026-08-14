"""Step 2's required sanity test: a series generated with no drift must
report no changepoints. Unskip once `make_series()` is implemented.

A second test — running an actual detector over a no-drift series and
asserting it finds nothing — belongs here too, but not until Sprint 3
adds a concrete `DriftDetector` (see src/drift_forecasting/detection/base.py).
Add it next to this one when that lands.
"""

import pytest

from drift_forecasting.synthetic.generator import make_series


@pytest.mark.skip(reason="unskip once make_series() is implemented — Sprint 1, Step 2")
def test_no_drift_series_has_no_changepoints():
    y, changepoints = make_series(kind="none", n=5000, noise=1.0, seed=0)
    assert changepoints == []
