"""Synthetic drift benchmark generator — Step 2.

NOTE: an initial version already exists on branch `team1/synthetic-data`
(synth_data_generator.py). Port/refactor that logic to match the signature
below rather than starting over — the changepoint list this returns is
the ground truth every detector metric in Step 4 is graded against, so
there must be exactly one generator in the codebase, not two that can
drift apart.
"""

from typing import Literal

import numpy as np

DriftKind = Literal["sudden", "gradual", "recurring", "none"]


def make_series(
    kind: DriftKind, n: int, noise: float = 1.0, seed: int | None = None
) -> tuple[np.ndarray, list[int]]:
    """Generate a synthetic series with known changepoints.

    Returns (y, changepoints) where `changepoints` are indices into `y`.
    kind="none" must return changepoints == [] — this is asserted by
    tests/test_synthetic.py and is required before any detector is
    trusted on real data.
    """
    raise NotImplementedError(
        "TODO: port team1/synthetic-data's synth_data_generator.py here — Sprint 1, Step 2"
    )
