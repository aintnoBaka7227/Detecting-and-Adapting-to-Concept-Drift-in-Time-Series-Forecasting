"""Arm A -- frozen baseline: never retrain, whatever `changepoints` says.

No signature change to the frozen `adapt(changepoints, model, data)`
contract -- this arm just always returns `model` unchanged. It does no
retraining, so `data` and `changepoints` are accepted but ignored; they
stay in the signature only because the frozen contract requires it.

Trivial on its own, but Step 5's four-arm comparison needs it as its own
class (not "skip the adapter" in the harness) so it produces the same
shape of run -- same config_hash / n_retrains=0 row -- as arms B/C/D, and
so the "no adaptation at all" case is explicit and reviewable rather than
implicit in whichever scripts happen to omit an adapter call.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from drift_lab.adaptation.base import Adapter
from drift_lab.forecasting.base import Forecaster


class NeverRetrain(Adapter):
    name = "never_retrain"

    def __init__(self) -> None:
        self._retrain_count = 0

    @property
    def retrain_count(self) -> int:
        return self._retrain_count

    def adapt(
        self, changepoints: Sequence[int], model: Forecaster, data: pd.DataFrame
    ) -> Forecaster:
        """Always a no-op: returns `model` unchanged, never increments
        `retrain_count`, regardless of `changepoints` or `data`."""
        return model
