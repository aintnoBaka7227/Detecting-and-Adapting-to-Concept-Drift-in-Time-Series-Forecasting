"""Step 4 / T1 — run the three detectors against the synthetic benchmark.

One `record_run` call per (detector, drift_type, seed); nothing is computed
here beyond `.detect()` — metrics come from `evaluation.evaluate_detections`
inside `record_run`.
"""

from __future__ import annotations

import time

from drift_lab.config import SEEDS
from drift_lab.detection.adwin import ADWINDetector
from drift_lab.detection.kswin import KSWINDetector
from drift_lab.detection.page_hinkley import PageHinkleyDetector
from drift_lab.synthetic.generator import Kind, make_series
from experiments.run_harness import config_of, record_run

KINDS: tuple[Kind, ...] = ("none", "sudden", "gradual", "recurring")
N = 20_000
NOISE = 1.0

# One instance per detector, reused across every (drift_type, seed): river
# detectors are rebuilt fresh inside .detect() each call (see
# detection/base.py::detect_with_river), so this is safe — and it keeps
# config_hash identical across all of a detector's runs. KSWIN's own `seed`
# (its internal reservoir sampling) is therefore fixed at its default, not
# tied to the experiment seed below, which only drives make_series.
DETECTORS = (ADWINDetector(), KSWINDetector(), PageHinkleyDetector())


def split_id_for(changepoints: list[int]) -> str:
    """Data-version tag: also encodes the changepoint positions, so if the
    generator's drift geometry ever changes (like it just did), the new
    rows land under a different split_id instead of silently mixing with
    the old ones under the same "synth_n20000"."""
    cp_tag = "-".join(str(cp) for cp in changepoints) or "none"
    return f"synth_n{N}_cp{cp_tag}"


def main() -> None:
    for kind in KINDS:
        for seed in SEEDS:
            y, changepoints = make_series(kind=kind, n=N, noise=NOISE, seed=seed)
            split_id = split_id_for(changepoints)
            for detector in DETECTORS:
                t0 = time.perf_counter()
                detected = detector.detect(y)
                record_run(
                    method=detector.name,
                    dataset=f"synthetic_{kind}",
                    seed=seed,
                    config=config_of(detector),
                    wall_clock_s=time.perf_counter() - t0,
                    split_id=split_id,
                    detection=(detected, changepoints, len(y)),
                )
        print(f"{kind}: {len(SEEDS)} seeds x {len(DETECTORS)} detectors done")


if __name__ == "__main__":
    main()
