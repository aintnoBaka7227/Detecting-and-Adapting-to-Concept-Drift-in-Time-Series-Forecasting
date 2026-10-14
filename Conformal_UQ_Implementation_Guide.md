# Fixed Split Conformal and AgACI: Development Guide

## Goal

Compare **fixed split conformal** with **AgACI using rolling calibration**, on the same existing point forecasts. Apply both to every forecaster included in the final experiments. The uncertainty layer must not change Arm A–D retraining policies or train another forecasting model for each learning rate.

AgACI has two layers of adaptation: each ACI candidate changes its internal miscoverage setting, and two online aggregators change how much each candidate contributes to the final interval.

## Step 1 — Check the interface and divide responsibilities

Inspect the existing `UncertaintyQuantifier` interface before writing classes. Preserve its required method signatures and return structure. The names below are proposed, not verified repository paths.

| Component | What to develop |
| --- | --- |
| `fixed_split_conformal.py` | The fixed-radius baseline. |
| `agaci.py` | ACI candidates (one learning rate and one internal level each), shared error buffer, endpoint aggregation and issued records. |
| BOA backend/helper | The paper's Bernstein Online Aggregation with the gradient trick. |
| Experiment harness | Forecast issuance, outcome delivery, chronological execution and logging. |
| `evaluation.py` | Metrics only; no adaptive state updates. |

The fixed and AgACI classes should inherit from the interface. Internal candidates and aggregators can be helper objects. If the interface lacks feedback support, provide a subclass observation method called by the harness; coordinate any change to a frozen base interface.

## Step 2 — Prepare calibration errors

Use errors from forecasts originally issued without knowing their target outcomes:

```text
score = abs(actual - issued_point_forecast)
```

Validate that scores are nonempty, finite and one-dimensional. Preserve their timestamps and chronological order. Match calibration forecasting conditions to the evaluated forecast horizon.

For the existing split, fit on 2018–2019, calibrate on January–February 2020, and begin test replay afterwards. If February is also reported as the stable reference, label its results **calibration-period diagnostics**, because those observations helped set the intervals.

## Step 3 — Keep the fixed baseline

For nominal 90% coverage, use `target_alpha = 0.10`. With `n` calibration scores:

```python
rank = ceil((n + 1) * (1 - target_alpha))
radius = kth_smallest_score(rank)  # rank is one-based
lower = point_forecast - radius
upper = point_forecast + radius
```

If `rank > n`, return an infinite radius or reject insufficient calibration clearly; do not silently cap the rank. Calibrate once and keep this radius unchanged throughout testing.

The standard coverage guarantee depends on assumptions such as exchangeability; it does not automatically apply to drifting time series.

## Step 4 — Configure AgACI

Create independent state for every **forecaster × region × arm × seed × configuration**, including detector settings when relevant.

| Configuration/state | Instruction |
| --- | --- |
| Target miscoverage | Keep `target_alpha = 0.10` fixed. |
| Learning-rate grid | Predefine several rates; maintain one candidate per rate. |
| Candidate levels | Initialise each `alpha_k = target_alpha`. |
| Shared error buffer | Seed with the latest `min(W, n)` calibration scores. |
| Endpoint aggregators | Maintain separate lower and upper BOA states. |
| Replacement thresholds | Define finite `M_lower` and `M_upper` using historical information. |
| Pending forecasts | Store original candidate bounds, final bounds and issuance metadata. |
| Issuance | One interval per half-hour for every forecaster; feedback is applied after each outcome. |

The paper examines rates including `0, 0.0001, 0.0004, 0.0007, 0.001, 0.004, 0.007, 0.01, 0.04, 0.07`. This is a published experimental grid, not an optimal AEMO setting. A zero-rate candidate keeps its level fixed, but its radius can still change with a rolling buffer.

The previous guide's **W = 336 scores (7 days)** remains provisional. Choose W through chronological historical validation or document it as a predetermined project choice. Do not tune on final test results. AgACI adapts candidate influence; it does not select W automatically.

## Step 5 — Develop one ACI candidate

At issuance, use the candidate's current internal level to select a calibration-error quantile:

```text
radius_k = conformal_quantile(shared_scores, 1 - alpha_k)
candidate_lower_k = point_forecast - radius_k
candidate_upper_k = point_forecast + radius_k
```

After the actual arrives, evaluate the candidate's **original interval**:

```python
miss_k = int(actual < original_lower_k or actual > original_upper_k)
alpha_k += gamma_k * (target_alpha - miss_k)
```

A miss lowers `alpha_k`, which selects a higher error percentile and usually widens future intervals. A covered outcome raises it slightly. Equality with an endpoint counts as covered.

Document the quantile convention and extreme-level behaviour from the selected ACI reference. Do not silently clip levels. In particular, a zero-radius interval and an empty prediction set are different conventions. Keep original candidate sets available for feedback even if their endpoints need replacement for aggregation.

**Check:** misses lower the level, coverage raises it, and prediction calls do not update candidate state.

## Step 6 — Implement and verify BOA

Follow **Zaffran et al. (2022), Section 4, Algorithm 1 and Appendix C.1**. Their chosen implementation uses BOA with the gradient trick through R's `opera` package.

1. Retrieve the authors' AgACI scripts and identify the actual `opera` settings, initialisation and update calls.
2. Use that R backend directly, or port its exact updates to Python.
3. Run a small chronological sequence of candidate endpoints and actuals through the reference and your implementation.
4. Compare lower/upper predictions and weights after each update, within a stated numerical tolerance.
5. Record the reference revision and package version.

Do not substitute a generic exponential-weight rule and call it the paper's BOA. This guide describes the development workflow; it does not provide a verified numerical BOA port.

For the gradient trick, use the aggregator's prescribed loss-gradient update, not just raw candidate losses. The two endpoint scoring levels stay fixed at **0.05 and 0.95** for a 90% target.

## Step 7 — Issue the final AgACI interval

1. Generate every candidate's original interval using currently available scores.
2. Replace non-finite candidate endpoints with the specified finite thresholds before aggregation, following Algorithm 1. Preserve both original and replacement endpoints.
3. Use the current normalised lower and upper weights:

```text
final_lower = sum(lower_weight_k * aggregation_lower_k)
final_upper = sum(upper_weight_k * aggregation_upper_k)
```

4. Save the forecast ID, issuance/target times, point forecast, original candidate bounds, replacement bounds, weights and final interval.

Lower and upper weights are separate, so the final interval can be asymmetric. Check that final bounds are finite and ordered. Flag crossing bounds and resolve the policy explicitly; do not silently sort them.

## Step 8 — Process feedback once

When the outcome becomes available:

1. Retrieve the saved issuance record.
2. Update each ACI candidate from its own original interval miss.
3. Update lower BOA using saved replacement lower endpoints at quantile 0.05, and upper BOA using saved replacement upper endpoints at 0.95.
4. Append `abs(actual - issued_point_forecast)` to the shared buffer **once**, then discard the oldest score if full.
5. Mark the feedback as processed and log updated levels and weights.

All updates affect future issuances. Do not recompute an old interval after seeing its outcome, and do not update every candidate from the final aggregated interval's miss.

## Step 9 — Integrate chronological replay

```text
Initialise fixed conformal and AgACI from calibration errors.

For each half-hour, in chronological order:
    produce the fixed bounds
    produce candidate bounds and the aggregate AgACI bounds for that half-hour
    save the original issuance record

When that half-hour's outcome arrives, before the next issuance:
    retrieve the corresponding issuance record
    update candidate levels and endpoint aggregators
    update the shared error buffer once
```

Issue one interval per half-hour for every forecaster (NHITS, XGBoost and DHR-ARIMA), regardless of when the point forecast was produced. Build each interval from the state available just before its target time, that is, from every outcome up to the previous half-hour, then apply its outcome as feedback before the next interval is issued. The point forecast itself is not revised: for NHITS it is still the value from the 48-step block, and for XGBoost and DHR-ARIMA the value from the last fit. An outcome must never change an interval that was already issued. Record the issuance step (1) in the run configuration.

For overlapping forecasts, select one evaluation horizon or maintain separate horizon states. Do not insert one target outcome repeatedly into the same state accidentally. Retain UQ state across forecaster retraining by default; any reset policy must be explicit and use only available outcomes.

Saved forecasts can be replayed without retraining when issuance times and outcome availability are recoverable.

## Step 10 — Evaluate and produce outputs

Use identical point forecasts and regime windows for fixed conformal and AgACI.

| Output | What to calculate/report |
| --- | --- |
| T3 | Endpoint pinball loss at 0.05/0.95; identify the UQ method supplying the bounds. Point-forecast MAE stays unchanged. |
| T5 | Coverage, mean width, normalised width and worst rolling 24-hour coverage, per UQ method and regime. |
| F4 | Rolling coverage for both methods, the 0.90 target and event markers. |
| T6/F5 | Width-based review thresholds, review rate and coverage among retained forecasts. |

For normalised width, record the agreed denominator explicitly. If using each regime's mean actual demand, calculate `mean(upper - lower) / mean(actual)` on the same evaluated observations. This is a retrospective metric, not an input to interval generation.

For worst 24-hour coverage, use 48 consecutive half-hours on a complete grid. A proposed regime rule requires all 48 points to belong to that regime; report N/A if no window qualifies. Never concatenate separate events before rolling. Keep this metric separate from a seven-day rolling-coverage plot.

Report candidate infinite-interval frequency and endpoint replacements separately from final AgACI metrics. Missing or reversed bounds are invalid, not intentional infinite intervals. Inspect the actual evaluator before assuming it supports infinity. For escalation, report when no forecasts remain or target coverage is unreachable.

Do not claim 90% coverage in every event or day. The rolling buffer, finite thresholds and delayed feedback must be documented as part of this project's implementation.

## Completion checklist

- [ ] Fixed conformal is calibrated once and remains fixed.
- [ ] Candidates use separate levels and a shared chronological error buffer.
- [ ] Lower and upper BOA implementations match the chosen reference.
- [ ] Original and replacement candidate endpoints are saved separately.
- [ ] Predictions do not update adaptive state; feedback is processed once.
- [ ] Half-hourly issuance and overlapping horizons respect outcome availability.
- [ ] Grid, buffer, quantile convention, thresholds and feedback policy are recorded.
- [ ] Fixed and AgACI use the same point forecasts and evaluation observations.
- [ ] T3/T5/F4 and escalation outputs are generated from saved intervals.

## Sources and scope

- [Zaffran et al. (2022), Adaptive Conformal Predictions for Time Series](https://proceedings.mlr.press/v162/zaffran22a/zaffran22a.pdf): Section 4/Algorithm 1 for endpoint aggregation; Appendix C.1 for BOA; Figure 4 for experimental learning rates.
- [Authors' reference repository](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries): inspect AgACI scripts and `opera` settings before implementing the backend.
- [Gibbs & Candès (2021), Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html): candidate ACI feedback update.

Describe the result as **AgACI with rolling calibration from originally issued forecast errors**. It follows the published aggregation framework while retaining existing forecasters and documenting project-specific calibration and feedback choices. AgACI replaces single-rate ACI as the proposed adaptive comparison; fixed split conformal remains the baseline.
