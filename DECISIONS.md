# Design decisions

This file records **why** the codebase is built the way it is — the
decisions made, the alternatives rejected, and the reasoning behind each,
organized per component. [`README.md`](README.md) teaches you how to
*use* what's here; this is the log of why it looks like this instead of
some other way, so nobody re-litigates a settled call or accidentally
undoes one while "cleaning up."

[`docs/refactor.md`](docs/refactor.md) is the original planning document
the refactor was built from — useful as history, but it still has a
couple of stale names (`_harness.py`, registries) that were superseded by
decisions below. This file is the current, corrected record; where the
two disagree, this one wins.

---

## Cross-cutting decisions (apply everywhere in `src/drift_lab/`)

**No registries.** Rejected explicitly, twice (once for models/detectors,
once again when it came up for adaptation arms). A `run_*.py` author
imports the class they want and instantiates it directly
(`XGBoostForecaster(max_depth=4)`, `ADWINDetector(delta=0.002)`) — there
is no `get_forecaster("xgboost")` lookup anywhere, and none should be
added. Reasoning: a registry is one more file everyone touches to add a
model, and it buys nothing a direct import doesn't already give you.

**One implementation per file.** One model, one detector, one file. Never
add a second class to someone else's file, even a small one.

**Two different underscore conventions, scoped differently — don't
conflate them:**
- **Instance attributes** (`self._history`, `self._model`, `self._nf`):
  underscore = private/fitted state, kept out of `config_hash`. This
  convention is load-bearing (`experiments/run_harness.py::config_of()`
  reads `vars(obj)` and keeps only the non-underscore keys) and applies
  everywhere, `src/drift_lab/` included.
- **File and function names** in `experiments/`: a leading-underscore
  convention (`_harness.py`, `_io.py`, `_label_regimes`) was tried and
  then explicitly rejected — renamed to `run_harness.py`, `results_io.py`,
  `label_regimes`. This was scoped to `experiments/` only; it was never
  proposed for `src/drift_lab/`, which didn't use that convention for
  files/functions to begin with.

**`src/drift_lab/` is implementation-only.** No working
`if __name__ == "__main__":` block, no file writes, the string
`"results/"` never appears in it. Anything that runs an experiment or
produces a table/figure lives in `experiments/`.

**Compose in memory within a stage.** A function passes its return value
to the next; nothing re-reads a file an earlier function in the same
stage just wrote. `aemo.loader.load(region)` returns the three splits
directly — there's no intermediate processed CSV. The **one** sanctioned
file hand-off in the whole pipeline is `results/runs.csv`: `run_*.py`
scripts append to it, `produce_*.py` scripts read it. Nothing else is
allowed to be a hand-off point.

**Comments are precise and minimal** — the non-obvious *why*, never the
*what*. If a comment just restates the code, it shouldn't exist.

**Package naming.** `drift_forecasting` was renamed to `drift_lab` (pip
name `drift-lab`) partway through the refactor, at the same time a
teammate independently renamed `data/` to `aemo/` in their own IDE session
— that concurrent rename was integrated into this one rather than
reverted, since both were moving in the same direction (clearer names).

---

## `config.py`

Single source of truth for anything that would otherwise be a magic
number duplicated across files:

- `REGIONS = ("SA1", "NSW1")` — both regions, always. Every figure/table
  in this codebase covers both; there is no "just do SA1 for now" mode.
- `SEEDS = (1, 2, 3, 4, 5)` — a team-agreed list so results are
  comparable across every table; not something an individual experiment
  script should override with its own seed list.
- `SPLIT` — train (2018–2019) / calibration (Jan–Feb 2020) / test (Mar
  2020 onward) — frozen for the semester. Every module that needs a date
  boundary imports it from here; nothing hardcodes a date elsewhere.
- `SEASON_LENGTH = 48` (half-hourly steps/day), `ROLLING_WINDOW_DAYS = 7`,
  `REGIME_DRIFT_MARGIN = 48 * 7` — the width of the "drift" regime band
  around a detected changepoint. This is a stated **assumption**, not a
  measured quantity — widen it if a detector's drift band turns out too
  narrow to show up in the rolling-MAE curve.
- `DOCUMENTED_EVENTS_CSV` — deliberately lives inside
  `src/drift_lab/aemo/`, not `notebooks/`, specifically to stop people
  from confusing it with scratch EDA output; it's pipeline input, not a
  notebook artifact.

---

## `aemo/` — AEMO demand data

- **One file, one function**: `loader.py` is read → clean → standardise →
  split, with no other module allowed to slice by date. `load(region)`
  returns `(train, calibration, test)` **in memory** — there is no
  processed CSV written anywhere. This was a deliberate consolidation:
  earlier versions of this pipeline had three separate copies of similar
  cleaning/splitting logic (`cleaning.py`, `splits.py`, a `csv_output/`
  directory of regenerable CSVs) that all got deleted in favor of this
  one file, because a change to a cleaning rule used to require touching
  three places to stay consistent.
- **`clean()` always prints its full audit, unconditionally** — there is
  no `verbose=False` flag. This was an explicit choice: the audit is
  cheap, and a silent-by-default cleaning step is exactly the kind of
  thing that hides a real data problem until someone happens to turn
  logging on.
- **`standardise()` resamples to a 30-minute grid and stops there** — raw
  MW/$ values are never altered or clipped, and gaps become `NaN` rows,
  never imputed. Imputation is a modeling decision, not a cleaning one;
  keeping it out of the loader means every downstream model sees the same
  honestly-gapped data and decides for itself how to handle a `NaN` lag.
- **`events.csv`** was converted from the team's `events.xlsx` verbatim
  (34 rows, tier + tier_reason columns) — not restyled, not re-derived,
  because it's a teammate-authored reference document, and the tiering
  criteria behind it (`docs/event_tiering_criteria.md`) is frozen before
  any detector-event matching is built against it.

---

## `synthetic/` — the drift benchmark

- **`make_series(kind, n, noise, seed)`** returns `(y, changepoints)` with
  no file output — it's regenerated on demand, never cached to disk,
  because caching it would risk a stale copy silently diverging from the
  generator's actual current logic.
- **`changepoints`' shape is `kind`-dependent by design**, matching
  exactly what `evaluation.evaluate_detections` expects for that
  `drift_type` — `none` → `[]`, `sudden` → one point `[cp]`, `recurring` →
  two points `[cp1, cp2]`, `gradual` → one **interval** `[start, end]`.
  The `gradual` shape was changed from an earlier single-point `[cp]`
  specifically to match `evaluate_detections`'s two-point interval
  requirement — the generator conforms to the evaluator's contract, not
  the other way around.
- **Changepoint position**: `sudden`/`gradual` inject their shift at
  `n // 4`, not `n // 2` — moved earlier by explicit request, so a
  detector has a long post-drift stretch to be evaluated against rather
  than a short tail. `recurring` was left at `n // 3` / `2n // 3`
  (roughly matching its "returns partway through" story) since there was
  no equivalent reason to move it.
- **`seed` controls only the noise draw**, never the changepoint
  positions — those are deterministic given `kind`/`n`. This is what lets
  `run_synthetic_detectors.py` build a `split_id` from the changepoints
  themselves rather than from the seed (see the harness section below).

---

## `forecasting/` — Forecaster implementations

- **`Forecaster.observe(y)`** exists specifically to replace a
  pre-refactor hack (`model._history = pd.concat([model._history, ...])`
  written directly from notebook code) with a real interface method.
  Default is a no-op; `SeasonalNaive` and `XGBoostForecaster` override it
  to extend their lag-lookup history across the train → calibration →
  test boundary without retraining. `DHRArima` deliberately keeps the
  no-op — its Fourier phase comes from the timestamp index itself
  (`X.index`), not from a lag buffer, so there's nothing to extend.
- **Frozen vs. walk-forward is a per-model choice, made by the `run_*.py`
  author, not by `record_run`.** `run_aemo_baselines.py::predict_input()`
  gives `seasonal_naive` the real test-period demand column (because
  lag-48 forecasting *is defined* as reading yesterday's real value) but
  gives `xgboost`/`dhr_arima` **no** target column at all — they're
  frozen and blind, rolling forward on nothing but their own past
  forecasts once past the calibration window. This asymmetry is
  deliberate: it's what makes the F1 "degradation curve" mean something
  — it's showing how badly a model degrades with zero feedback, which
  only works if the model genuinely gets zero feedback.

---

## `detection/` — DriftDetector implementations

- All three detectors (`adwin.py`, `kswin.py`, `page_hinkley.py`) are thin
  wrappers around a `river` detector, sharing one helper —
  `detect_with_river()` in `detection/base.py` — that feeds a stream
  through a fresh river detector one value at a time and collects flagged
  indices. Factored out specifically so the three wrappers can't drift
  apart in how they do this loop.
- **KSWIN's `seed` parameter is not the experiment seed.** It's KSWIN's
  own internal reservoir-sampling seed, deliberately left fixed at its
  river default (42) rather than tied to the outer experiment loop's seed
  in `run_synthetic_detectors.py` — the experiment seed only drives
  `make_series`'s noise draw. Conflating the two would make it impossible
  to tell whether a result changed because the *data* changed or because
  KSWIN's internal sampling changed.
- **Page-Hinkley's current defaults are a known, deliberately preserved
  starting point, not a bug to silently patch.** It fires ~311–312 false
  alarms per 10,000 observations on every drift type in the T1 table,
  including `"none"` — it reproduces the old notebook's exact
  hyperparameters faithfully, and is left that way so a teammate's
  research ticket has a real, measured starting point to improve on
  rather than a pre-tuned one.

---

## `adaptation/` and `uncertainty/` — scaffolded, not implemented

Both packages contain only their frozen `base.py` contract and a worked
example in its docstring. This is deliberate current-sprint scope, not an
oversight: forecasting + detection were built end to end first;
adaptation arms and UQ methods are each a teammate's own file, added in
the sprint that needs them. Nobody should pre-create those files for
someone else.

---

## `evaluation/` — the one place metrics are computed

- **The whole module was swapped in verbatim from a teammate's (Sahar's)
  implementation**, not written from scratch here, specifically so there
  is exactly one, team-agreed definition of every metric. The one
  deviation from the original file is a single `# noqa: TRY004` comment.
- **Enforced, not just documented**: `tests/test_evaluation.py::test_no_metric_code_outside_evaluation_module`
  greps `src/drift_lab/` and `experiments/` for `.rolling(`/`.ewm(` outside
  this one file and fails the build if it finds any. This exists because
  the failure mode it prevents — someone inlining a slightly-different
  rolling-MAE calculation in a `produce_*.py` script — is exactly the kind
  of thing that silently makes two figures disagree with each other.
- **`evaluate_detections` is `drift_type`-aware on purpose**: `none`
  treats every detection as a false alarm; `sudden`/`recurring` match
  point events within a tolerance window; `gradual` matches against a
  `[start, end]` interval instead, with delay measured from `start`. This
  mirrors the generator's own `changepoints` shape per `kind` (see
  `synthetic/` above) — the two were designed together, not independently.
- **AEMO detection evaluation (documented-event matching) is explicitly
  planned, not implemented.** Real AEMO data has no ground-truth
  changepoints — only documented events as *contextual* reference points.
  An unmatched AEMO detection must never be auto-classified as a false
  alarm the way a synthetic one is, because the documented-event list is
  not exhaustive ground truth. See "Deferred" below.

---

## The experiment harness and `results/runs.csv`

This is the part of the codebase every table and figure ultimately
depends on, so its design decisions get their own section rather than
being buried per-file.

### The core principle

`experiments/run_harness.py::record_run(...)` is the **only** function
allowed to append a row to `runs.csv`, and the **only** place a raw array
gets turned into a metric (via `evaluation/`, never inline). It does not
run anything itself — a `run_*.py` script fits/predicts or detects first,
then hands the resulting arrays to `record_run`. `produce_*.py` scripts
are the only readers, and they only ever `groupby` — no metric is
recomputed on the read side. This three-way split (run → record → produce)
is the "compose in memory, one sanctioned file hand-off" rule from the
top of this document, applied concretely.

### Where experiment output lives — and why nowhere else

Every experiment writes exactly three kinds of output, to exactly three
places, and nothing is allowed to land anywhere else:

```
results/
├── runs.csv                 the ledger — scalar metrics only, one row per
│                             (method × dataset × region × seed × metric × regime)
├── runs/<config_hash>/
│   ├── config.json          the actual hyperparameter values behind that hash
│   └── curve_<dataset>_<region>_<seed>.csv    the full rolling-MAE curve, one
│                             per forecast run — not squeezed into runs.csv
└── figures/                  every table/figure a produce_*.py script emits
```

- **`runs.csv` holds scalars only** (`mae`, `rolling_mae_7d_mean`,
  `rolling_mae_7d_max`, `detection_delay`, `false_alarms_per_10000`,
  `missed_detections`, `n_detections`) — a full time-indexed curve doesn't
  fit a long-format row, so it was never going to live here.
- **The curve dump is what a figure script actually reads for the line
  itself.** `produce_figure_f1.py` doesn't recompute a rolling MAE from raw
  forecasts — it reads the exact curve `run_aemo_baselines.py` already
  dumped via `record_run` → `results_io.dump_curve()`, keyed by
  `config_hash`/`dataset`/`region`/`seed`
  (`results_io.curve_path()` builds that path consistently on both
  the write and read side — see the `seed=None` bug above for why that
  consistency matters). `produce_table_t1.py`, by contrast, never touches
  a curve dump — a table of detection metrics is entirely a `groupby` of
  scalars already in `runs.csv`.
- **`results/figures/` is the only legal write target for a
  `produce_*.py` script.** Nothing it produces belongs in `data/`,
  `notebooks/`, the repo root, or a new folder invented per-script — use
  the `FIGURES_DIR` constant from `experiments/results_io.py`, don't
  hardcode a path.
- **No `run_*.py` script writes a file directly, ever.** It calls
  `record_run(...)`; `record_run` is the only thing that touches
  `results_io.append_runs` / `dump_config` / `dump_curve`. If a script is
  about to do its own `df.to_csv(...)` outside of that call, that's a
  sign the output belongs in `runs.csv` (or a curve dump) instead, not a
  new convention.
- **`runs.csv` is the only piece of `results/` tracked in git** — `runs/`
  and `figures/` are gitignored (see `.gitignore`) precisely because
  they're mechanically regenerable from `runs.csv` (or from re-running
  the `run_*.py` scripts). If you ever find yourself wanting to commit a
  figure or a curve CSV directly, that's a sign something upstream isn't
  actually reproducible yet — fix that instead.

This is also why the `seed=None` curve-path bug described below turned
out to be about this contract specifically — almost every "where did
this number come from" question in this codebase resolves to one of
these three files.

### Column-by-column reasoning

| Column | Why it's there / non-obvious rule |
|---|---|
| `seed` | `None` (written as `NaN`) for a genuinely deterministic method — **never a fabricated seed value** just to fill the column. A model whose training is genuinely stochastic should log its real seed instead — the column exists for that case too, not only the deterministic one. |
| `config_hash` | `sha1(json.dumps(config, sort_keys=True, default=str))[:12]`, where `config` comes from `config_of(obj)` — the object's public (non-underscore) attributes only. This is *why* the underscore-prefix instance-attribute convention above is load-bearing: fitted state leaking into `config` would make the same hyperparameters hash differently run to run. |
| `split_id` | Identifies the **data-partitioning scheme**, not "this batch of rows" (that's what `dataset` + `method` + `config_hash` already identify together). Required, with **no default** — an earlier version defaulted to `"frozen_v1"`, which silently mislabeled every synthetic run with an AEMO-shaped id. For synthetic detection, `split_id` embeds the actual changepoint positions (`synth_n{N}_cp{cp1-cp2-...}`) so a future change to the generator's drift geometry can't silently mix with old rows sharing the same generic id. |
| `regime` | See "The regime decision" below — this one took real back-and-forth to settle and is worth reading in full before changing it. |
| `n_retrains`, `train_samples`, `wall_clock_s` | Run bookkeeping the report needs alongside accuracy — a retrain, or a slow model, is not free, and the harness makes sure that cost is recorded next to the number it bought. |

### The regime decision

`regime` was the one column that took a real, documented back-and-forth
to settle, cross-checked against both the required-outputs brief and an
actual Slack conversation with a teammate on the other evaluation team.
The final rule:

- **Detection, synthetic** (ground truth known): always `regime="full"`.
  Splitting delay/false-alarms/missed by regime was judged not worth the
  added complexity, and `full` is already how the other team presents
  Table 1 — kept for consistency with their evaluation, not derived
  independently.
- **Detection, AEMO** (no ground truth): also `regime="full"`, logging a
  single `n_detections` count. This was **not** assumed lightly — an
  earlier claim that "AEMO has no known drift location" was corrected: the
  required-outputs brief does discuss AEMO drift (COVID, ~March 2020).
  The resolved position is subtler: AEMO's drift is genuinely **unknown a
  priori** — only a detector's own output can say where it might be, and
  that gets compared against documented events *after the fact* (T2, not
  yet built). Regimes defined *relative to a detector's own changepoints*
  (pre-drift before it, drift in a window around it, post-drift after)
  were considered, but for now this is a **placeholder**, explicitly
  revisitable without a schema change once the documented-event matcher
  exists.
- **Forecast rows**: `record_run(..., changepoints=[...])` splits metrics
  into `pre-drift`/`drift`/`post-drift`; omitting `changepoints` gives one
  `regime="full"` set. `changepoints` here is **never ground truth for
  AEMO** — it's either the synthetic generator's known changepoints, or a
  detector's own output, passed in by whichever `run_*.py` author is
  calling `record_run`. The regime boundary width is
  `config.REGIME_DRIFT_MARGIN` (7 days) — a stated assumption from the
  same Slack conversation, adjustable if a detector's drift band turns out
  too narrow to see in the rolling-MAE curve.

### A bug worth remembering (fixed, but easy to reintroduce)

- **`curve_path(..., seed=None)` used to interpolate Python's literal
  `None`** into the filename (`"...None.csv"`). After a `runs.csv`
  round-trip, `pd.read_csv` turns an empty seed cell into `NaN`, and
  `str(nan)` is `"nan"`, not `"None"` — the write path and the read path
  would drift apart and the curve file would become unfindable. Fixed by
  an explicit `seed_token = "none" if seed is None else str(seed)`, used
  identically on both the write (`dump_curve`) and read
  (`produce_figure_f1.py`) sides. If you touch `curve_path`, keep the
  token symmetric on both ends.
- **`runs.csv` is append-only, on purpose** — re-running a `run_*.py`
  script adds new rows rather than overwriting old ones, including after
  an interrupted run that only got partway through its loop.
  `produce_*.py` scripts are expected to take the **latest** row per
  `(method, region)` when duplicates exist (see
  `produce_figure_f1.py::load_model_curves()`'s `rows.iloc[-1]`) rather
  than assuming one row per method ever exists.

---

## Testing conventions

- **One test file per package/module it covers** (`test_detectors.py`,
  `test_dhr_arima.py`, `test_evaluation.py`, ...), mirroring `src/`'s
  layout.
- **The grep-guard** (`test_no_metric_code_outside_evaluation_module`) is
  the one test allowed to fail for a reason other than "you broke this
  specific thing" — it fails when metric logic leaks outside
  `evaluation.py`, anywhere in the tree.
- **`RUN_BASELINE_REPRO=1`** gates tests that reproduce a committed
  baseline CSV against real AEMO data (`test_xgboost_forecaster.py`,
  `test_dhr_arima.py`) — skipped by default because they take minutes.
  These are pinned to the model's *current* exact defaults; if a
  teammate's research changes those defaults, the test is **expected** to
  start failing — that's a signal to bring to the team about whether to
  accept the new defaults as the new frozen baseline (and regenerate the
  reference CSV), not something to quietly loosen.

---

## Deferred — needs the team before building

- **T2 documented-event matcher.** `aemo/events.csv` (Tier 1: 5 primary
  AEMO/AER events; Tier 2: the rest — see
  `docs/event_tiering_criteria.md`, frozen before matching) exists, but
  nothing scores a detector's AEMO output against it yet. Needs a
  `match_detections_to_events(...)`-shaped function: Tier 1 first, then
  Tier 2, else `unmatched` — **never** auto-classified as a false alarm,
  since the documented list isn't exhaustive ground truth. Where it lives
  (presumably `evaluation.py`) and the temporal matching rule are open.
- **`label_regimes` ownership.** Currently lives in
  `experiments/run_harness.py`. Whether the shared `evaluation.py` module
  should own it instead is open — both this team's forecast regimes and
  the detection team's event-matching need to carve the AEMO timeline
  consistently, which argues for one shared owner.
- **Recovery regime.** Between two changepoints, error can fall back to
  the pre-drift level (most visible for `recurring` drift); that segment
  is currently still labeled `post-drift`. If it needs its own label,
  that's a change inside `label_regimes`, not a schema change.
