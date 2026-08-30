# Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting
Design and implement an automated software pipeline that detects when concept drift occurs in a time series and adapts the underlying forecasting model in response, while also reporting how confident its predictions are and flagging cases where a human should review the output.

## Setup

```bash
pip install -e ".[dev]"
```

This installs the `drift_forecasting` package (from `src/`) in editable
mode, so `import drift_forecasting...` works from anywhere — notebooks,
tests, your own scripts — without path hacks.

## The pipeline, and why the structure looks like this

```
Data -> Detect -> Adapt -> Quantify (+ Escalate)
```

Three of these stages are fixed, frozen function contracts, agreed in
Sprint 1, so 4–5 people can build against them in parallel without
breaking each other's code:

```python
detector(stream) -> changepoint indices
adapter(changepoints, model, data) -> updated model
uq(point_forecast, calibration_residuals) -> lower bound, upper bound, escalate flag
```

Each contract has its own package under `src/drift_forecasting/`, and
each package's `base.py` is the contract — an abstract class nobody
edits casually. Full rationale (including why "Escalate" doesn't get its
own package) is in [`docs/interfaces.md`](docs/interfaces.md).

## Current project structure

```
.
├── src/drift_forecasting/        # the installable package — all pipeline code lives here
│   ├── config.py                  # frozen paths, regions, train/calibration/test split
│   │
│   ├── data/                      # Step 1 — load, clean, split AEMO demand
│   │   ├── loader.py               #   load_aemo_demand(region) -> DataFrame
│   │   ├── cleaning.py             #   DST gaps, duplicates, missing half-hours
│   │   └── splits.py               #   enforces config.SPLIT — the only place allowed to slice by date
│   │
│   ├── synthetic/                 # Step 2 — synthetic drift-benchmark generator
│   │   └── generator.py            #   make_series(kind, n, noise) -> (y, changepoints)
│   │
│   ├── forecasting/               # Step 3 — forecasting models (empty except the contract)
│   │   └── base.py                 #   Forecaster contract: fit(X, y), predict(X)
│   │
│   ├── detection/                 # Step 4 — drift detectors (empty except the contract)
│   │   └── base.py                 #   DriftDetector contract: detect(stream) -> indices
│   │
│   ├── adaptation/                # Step 5 — retraining policies (empty except the contract)
│   │   └── base.py                 #   Adapter contract: adapt(changepoints, model, data) -> model
│   │
│   └── uncertainty/               # Step 6 — prediction intervals + escalation
│       └── base.py                 #   UncertaintyQuantifier contract: quantify(...) -> UQResult
│
├── tests/                         # pytest suite, one file per package (mirrors src/ layout)
│   └── test_synthetic.py           #   currently the only test: no-drift series -> no changepoints
│
├── data/                          # gitignored on disk: raw/, processed/, synthetic/
├── reports/figures/                 # figures for the report; Step 1's 2019-vs-2021 figure goes here
├── notebooks/exploration/            # scratch EDA notebooks — not imported by anything, not production code
├── docs/interfaces.md                # the frozen contracts explained, plus a module-ownership table
└── pyproject.toml                    # package metadata + dependencies
```

`forecasting/`, `detection/`, `adaptation/` and `uncertainty/` each
contain **only** their `base.py` right now — that's deliberate. The
concrete implementations (an actual ADWIN detector, an actual retraining
policy, ...) are each teammate's own file, added in the sprint that
needs it. Nobody should pre-create those for someone else.

## Where do I put my code?

Find your task in the left column, then open the `base.py` named in the
right column first — it has a "How to implement" section in its
docstring with a worked example of exactly what your class should look
like.

| You're building... | File goes in | Subclass | Read first |
|---|---|---|---|
| A forecasting model (seasonal-naive, classical, GBM) | `forecasting/<your_model>.py` | `Forecaster` | `forecasting/base.py` |
| A drift detector (ADWIN, Page-Hinkley, KSWIN) | `detection/<your_detector>.py` | `DriftDetector` | `detection/base.py` |
| A retraining policy (one of the four Step 5 arms) | `adaptation/<your_arm>.py` | `Adapter` | `adaptation/base.py` |
| A UQ method (conformal, adaptive conformal) | `uncertainty/<your_method>.py` | `UncertaintyQuantifier` | `uncertainty/base.py` |
| The data loader / cleaning logic | edit `data/loader.py` / `data/cleaning.py` directly | — | Step 1 in the brief |
| The synthetic generator body | edit `synthetic/generator.py` directly | — | Step 2 in the brief, and branch `team1/synthetic-data` |
| A metric shared across steps (rolling MAE, coverage, detection delay, escalation sweep) | new `evaluation/` package — create it when Sprint 2/3 needs it | plain function, no base class | — |
| End-to-end wiring | new `pipeline.py` — create it in Sprint 5–6 | — | `docs/interfaces.md` |
| The prototype UI | new `app/streamlit_app.py` — create it in Sprint 7 | — | Step 7 in the brief |

A few rules that keep five people from stepping on each other:

- **One implementation per file.** One detector = one file. Don't add a
  second class to someone else's file, even a small one — open a new
  file next to it instead.
- **Never edit a `base.py` to fit your implementation.** If the frozen
  contract genuinely doesn't fit what you're building, that's a team
  conversation (see `docs/interfaces.md`), not a solo edit.
- **Name your test file after the package you're testing** —
  `tests/test_detection.py` for anything under `detection/`, etc.
- **Import via the package**, e.g. `from drift_forecasting.detection.base import DriftDetector`,
  never a relative or local path — that's what the editable install is for.

See [`docs/interfaces.md`](docs/interfaces.md) for the full contract
rationale and the module-ownership sign-up table.

#Dependencies
sklearn
statsmodels
