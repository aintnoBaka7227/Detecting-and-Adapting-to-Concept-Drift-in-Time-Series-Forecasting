"""Run ADWIN against the shared synthetic drift benchmark.

One record_run call is made per (drift_type, seed). Detection metrics are
computed only by the shared evaluation workflow inside record_run.
"""

from __future__ import annotations

import time

from drift_lab.config import SEEDS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.synthetic.generator import Kind, make_series
from experiments.run_harness import config_of, record_run

KINDS: tuple[Kind, ...] = ("none", "sudden", "gradual", "recurring")
N = 20_000
NOISE = 1.0


def split_id_for(changepoints: list[int]) -> str:
    """Identify the synthetic geometry used for a recorded run."""
    cp_tag = "-".join(str(cp) for cp in changepoints) or "none"
    return f"synth_n{N}_cp{cp_tag}"


def main() -> None:
    detector = ADWINDetector()

    for kind in KINDS:
        for seed in SEEDS:
            y, changepoints = make_series(
                kind=kind,
                n=N,
                noise=NOISE,
                seed=seed,
            )

            t0 = time.perf_counter()
            detected = detector.detect(y)
            elapsed = time.perf_counter() - t0

            record_run(
                method=detector.name,
                dataset=f"synthetic_{kind}",
                seed=seed,
                config=config_of(detector),
                wall_clock_s=elapsed,
                split_id=split_id_for(changepoints),
                detection=(detected, changepoints, len(y)),
            )

        print(f"{kind}: {len(SEEDS)} ADWIN runs done")


if __name__ == "__main__":
    main()
