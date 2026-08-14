# Frozen interfaces

These three contracts are fixed for the semester (see the project briefing,
"The system" slide). Changing a signature after Checkpoint 1 breaks
whoever built against it — if a signature genuinely needs to change,
raise it with the whole team first, don't just edit it.

```python
detector(stream) -> changepoint indices
adapter(changepoints, model, data) -> updated model
uq(point_forecast, calibration_residuals) -> lower bound, upper bound, escalate flag
```

## Where each contract lives

| Contract | Abstract base class | File |
|---|---|---|
| `detector` | `DriftDetector` | [`src/drift_forecasting/detection/base.py`](../src/drift_forecasting/detection/base.py) |
| `adapter` | `Adapter` | [`src/drift_forecasting/adaptation/base.py`](../src/drift_forecasting/adaptation/base.py) |
| `uq` | `UncertaintyQuantifier` | [`src/drift_forecasting/uncertainty/base.py`](../src/drift_forecasting/uncertainty/base.py) |

A fourth contract, `Forecaster` (`fit`/`predict`), isn't on the slide but
is required by `adapter`'s signature — every model type has to expose the
same shape so an adapter can retrain any of them interchangeably. It
lives in [`src/drift_forecasting/forecasting/base.py`](../src/drift_forecasting/forecasting/base.py).

**Each of these four files has a "How to implement" section in its own
docstring** with a worked example of subclassing it — read the file
itself before writing a new implementation against it.

## Where escalation lives

The architecture slide shows five pipeline stages (Data, Detect, Adapt,
Quantify, Escalate) but only three frozen functions — because the
escalate flag is returned *by* `uq()`, not by a separate stage. So there
is no top-level `escalation/` package: the decision logic sits in
`uncertainty/`. The Step 6 threshold-sweep report (coverage vs. fraction
escalated) will be analysis code added later — it calls `quantify()` at
many thresholds, it doesn't change the interface.

## The split

`config.SPLIT` (train 2018–2019, calibration Jan–Feb 2020, test Mar 2020
onward) is the other thing frozen this sprint. Every module that needs a
date boundary imports it from `drift_forecasting.config` — nothing
hardcodes a date elsewhere.

## Module ownership (fill in at the Sprint 1 team meeting)

Only the interface (`base.py`) exists for these today — the concrete
implementations are each teammate's own file, added in the sprint listed.

| Area | Owner | Sprint(s) |
|---|---|---|
| `data/`, `config.py`, `synthetic/generator.py` | _TBD_ | 1–2 |
| `forecasting/` (implementations) | _TBD_ | 2 |
| `detection/` (implementations) | _TBD_ | 3 |
| `adaptation/` (implementations) | _TBD_ | 4 |
| `uncertainty/` (implementations) | _TBD_ | 5 |
| Integration: pipeline wiring, prototype UI, leakage audit sign-off | _TBD_ | 5–7 |

## Leakage rule

No `fit`, `quantify`, or split-boundary decision may use data from after
its own point in the stream. This is audited explicitly in Step 7, but
every interface above is documented with it in mind from the start.
