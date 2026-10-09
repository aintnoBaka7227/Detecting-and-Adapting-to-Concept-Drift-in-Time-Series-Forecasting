# Fixed Split Conformal and Adaptive Conformal Inference: Implementation Guide

## Purpose

Implement two uncertainty methods in separate files. Both must inherit from the existing `UncertaintyQuantifier` interface. Apply them to every forecaster included in the final experiments, not only NHITS. Keep the existing forecasters and Arm A–D retraining policies unchanged by the uncertainty layer.

| Proposed file | Proposed class | Purpose |
| --- | --- | --- |
| `fixed_split_conformal.py` | `FixedSplitConformal` | Generate intervals using a radius calibrated once. |
| `adaptive_conformal.py` | `AdaptiveConformalInference` | Generate intervals using a rolling error buffer and a quantile level updated from coverage feedback. |

These filenames and class names are proposals. The interface definition has not been inspected, so import paths, constructor arguments and method signatures must be adapted to the actual codebase.

## 1. Responsibilities

| Component | Responsibility |
| --- | --- |
| The two uncertainty files | Calibration, interval generation, method state and optional escalation flags. |
| Experiment harness | Forecast issuance, delivery of observed outcomes, chronological execution and logging. |
| `evaluation.py` | Coverage, width, pinball loss, rolling coverage and escalation metrics. |
| Table/figure scripts | Generate T3–T6 and F3–F5 from saved results. |

Do not put ACI state updates inside evaluation functions. Neither uncertainty class should train or retrain the forecaster.

## 2. Instructions for `fixed_split_conformal.py`

1. Import `UncertaintyQuantifier` and inherit from it. Implement every required abstract method using the existing signatures.
2. Accept `alpha=0.10` as the default for a nominal 90% prediction interval.
3. Receive chronological calibration residuals calculated from forecasts issued without access to their target outcomes. Convert them to absolute scores:

   ```text
   score_i = abs(actual_i - issued_forecast_i)
   ```

4. Validate that residuals are nonempty, one-dimensional and finite, and that `0 < alpha < 1`.
5. Calculate the finite-sample conformal rank:

   ```python
   rank = math.ceil((n + 1) * (1 - alpha))
   ```

6. Select the rank-th smallest absolute residual using `np.partition(scores, rank - 1)[rank - 1]`.
7. If `rank > n`, handle insufficient calibration explicitly. An infinite radius is the strict conformal convention; alternatively, reject insufficient calibration with a clear error. Do not silently cap the rank and claim the original guarantee.
8. Store the radius after calibration. Do not update it from test outcomes.
9. Generate bounds for each supplied point forecast:

   ```python
   lower = forecasts - radius
   upper = forecasts + radius
   ```

10. If the interface requires escalation flags, support an explicit width threshold, or return all-false flags when escalation is disabled. A constant radius gives constant width, so a width-only rule selects all or none of the forecasts within that instance.
11. Expose configuration and radius for logging. Reject prediction calls before calibration.

### Verification

- Check the radius against a manually calculated order statistic.
- Check symmetric bounds and constant width across repeated calls.
- Check invalid and insufficient calibration inputs.
- Confirm test observations do not change the radius.

### Interpretation

The method uses one historical error distribution throughout testing. It supplies the initial uncertainty baseline. Standard split-conformal coverage results depend on assumptions such as exchangeability; they do not automatically apply to drifting time series [1].

## 3. Instructions for `adaptive_conformal.py`

### Initial configuration

Use ACI with a rolling buffer of absolute errors from originally issued forecasts. This replaces the earlier fixed-score ACI proposal. The fixed split-conformal baseline still retains its original calibration scores and radius.

Two components adapt: the buffer represents recent forecasting errors, while ACI adjusts the selected quantile level using interval misses. ACI's published update can wrap a conformal procedure [2, 3]. A rolling issued-error buffer is the proposed calibration configuration for this project; it is not an exact reproduction of every model-fitting and calibration choice in those papers. It is not full EnbPI.

Choose and log a positive integer `window_size` W and learning rate `gamma` using historical validation, then freeze them. No particular W is prescribed here. The previously discussed 14-day window was illustrative. On a complete half-hourly single-horizon stream, W = 48 × days; this conversion does not apply unchanged to sparse or horizon-specific feedback.

### Implementation steps

1. Import and inherit from the same `UncertaintyQuantifier` interface.
2. Accept `target_alpha=0.10`, configurable `gamma`, a positive integer `window_size`, and calibration residuals through the appropriate interface methods.
3. Validate residuals and parameters. Require a finite positive `gamma` for an adaptive run.
4. Seed a bounded chronological buffer with the latest min(W, n) absolute calibration scores, for example using `deque(scores[-W:], maxlen=W)`. Preserve chronological order, not sorted order. Validate that initial calibration is sufficient for the requested quantile or handle an infinite radius explicitly.
5. Initialise `alpha_current = target_alpha`.
6. At forecast issuance, compute the radius using `alpha_current`, rather than `target_alpha`:

   ```text
   radius_t = corrected conformal quantile at level 1 - alpha_current
   ```

7. Return bounds centred on the supplied point forecast:

   ```python
   lower = forecasts - radius_t
   upper = forecasts + radius_t
   ```

8. Keep prediction calls free of feedback updates. Repeated calls without new observations must not change `alpha_current`.
9. Add an observation/update method consistent with the existing interface. It must receive actual outcomes, originally issued point forecasts and originally issued bounds, or resolve these from saved forecast identifiers.
10. Determine whether each issued interval missed:

    ```python
    miss = int(actual < issued_lower or actual > issued_upper)
    ```

    Equality with either endpoint counts as covered.

11. After checking the original interval, append `abs(actual - issued_point_forecast)` to the buffer. Discard the oldest error when full. Also apply the published ACI feedback update:

    ```python
    alpha_current += gamma * (target_alpha - miss)
    ```

    Both updates affect future interval issuances only. Never recalculate an already-issued interval before evaluating its miss.

12. Handle extreme levels explicitly:

    | Condition | Published convention / required handling |
    | --- | --- |
    | `alpha_current <= 0` | Infinite interval. |
    | `alpha_current >= 1` | Zero-radius interval. |
    | `0 < alpha_current < 1`, but corrected rank exceeds score count | Infinite radius under strict finite-sample handling. |

    Do not silently clip levels or widths. Clipping creates a modified method and can change its error-control properties. Validate point forecasts as finite values even when bounds can be infinite.

13. Record `alpha_current`, radius, buffer size, issued bounds, miss indicators and update timestamps for diagnosis.
14. Support the same optional escalation-threshold configuration as the fixed method.

### Why the update works

For a 90% target and illustrative `gamma=0.01`:

| Observation | Update from `alpha_current=0.10` | Next quantile level |
| --- | --- | --- |
| Outside the issued interval | `0.10 + 0.01 * (0.10 - 1) = 0.091` | 0.909 |
| Inside the issued interval | `0.10 + 0.01 * (0.10 - 0) = 0.101` | 0.899 |

A miss lowers the internal miscoverage level, selecting a higher quantile and potentially a wider radius. Coverage moves it in the opposite direction. Empirical quantiles are discrete, so every update need not change the radius.

`gamma=0.01` is an explanatory example, not a selected project parameter. Choose the actual learning rate using historical validation.

### Verification

- A miss decreases `alpha_current`; coverage increases it.
- Prediction alone does not update state.
- Feedback evaluates original bounds, not bounds recalculated after the outcome.
- Each outcome updates state once for the forecast being evaluated.
- New errors enter the buffer, the oldest leave when full, and each stored error remains based on its original issued forecast.
- Prediction calls mutate neither the buffer nor the quantile level.
- Extreme levels and finite-sample rank overflow follow documented conventions.
- Reinitialising a run resets all state.

ACI aims to control long-run miscoverage under its formulation. Do not claim 90% coverage in every event or rolling 24-hour window, particularly when applying a different delayed-feedback or block-update schedule [2, 3].

## 4. Interface compatibility

Before implementation, inspect the actual base interface and existing uncertainty implementation.

- Preserve required method names, signatures and return structure.
- If the interface returns `(lower, upper, escalate)`, both classes must do so.
- If it supplies calibration residuals on every prediction call, define an explicit lifecycle so repeated calls do not accidentally reset ACI.
- Initialise calibration once per forecaster/region/arm/seed/configuration, and separately per horizon when using horizon-specific streams.
- If the base interface lacks an observation method, add an explicit subclass method that the harness can call. Do not modify a frozen base interface without coordinating the change.
- Avoid shared mutable state between instances.

## 5. Harness integration instructions

1. Instantiate separate fixed and ACI objects for each forecaster × region × arm × seed × experiment configuration. Include the detector configuration for Arms C/D when comparing several triggers. Never pool mutable UQ state across different runs.
2. Supply the pair with the same appropriate calibration residual stream. Fixed conformal uses the full chosen calibration pool; ACI seeds its buffer from the latest W available scores. Document that difference. Match calibration forecasting conditions to the evaluated horizon as closely as possible.
3. Supply identical point forecasts to the two methods.
4. Preserve Arm A–D retraining policies. Uncertainty updates must not trigger additional forecaster fitting unless a separate policy explicitly requires it.
5. Save forecast identifiers, issuance times, target times, point forecasts and originally issued bounds.
6. Respect each model's actual issuance schedule. For a single forecast, use information available at its issuance; for any multi-step block, construct every bound from information available when the block was issued. The 48-step block is an NHITS example, not a universal rule. Outcomes from a block must not influence its already-issued intervals.
7. Deliver ACI feedback only as the actual outcomes become available. Process it chronologically and once per evaluated forecast.
8. If all outcomes in a block arrive before the next issuance, apply their feedback before the next block. Document this delayed-feedback schedule; it is not identical to immediate one-step feedback.
9. For overlapping forecasts, select a consistent evaluation horizon or maintain horizon-specific streams. Do not accidentally count several forecasts for one target as independent feedback for one state.
10. Select `gamma` and W using historical validation and freeze them before final test evaluation. Use the same parameter-selection protocol across models; if values differ, document the choices.
11. Keep the same uncertainty procedure across teams for the final trigger comparison.
12. Saved forecasts can be replayed without retraining if their chronology and information-availability boundaries are preserved.

### Applying the methods across forecasters

Use the same two UQ classes for seasonal naive, DHR–ARIMA, XGBoost, NHITS, or other included models. The classes operate on forecasts and outcomes, not on model-specific training code.

| Shared across models | Separate per experiment stream |
| --- | --- |
| Class implementations and nominal 90% target | Calibration residuals and fixed radius |
| Evaluation formulas and regime protocol | Rolling error buffer and ACI level |
| Parameter-selection procedure | Forecast horizon and issuance schedule |
| Logging schema | Feedback timing and pending issued forecasts |

Use independent state per horizon when evaluating multiple lead times, or explicitly document and assess a pooled-horizon approach. Do not insert the same target outcome repeatedly into one shared state merely because several overlapping forecasts exist.

At a forecaster retrain, retain the rolling UQ state by default: new issued-forecast errors gradually replace old-model errors. An automatic reset is a different policy and must be explicitly defined and evaluated; never seed a reset using outcomes that have not yet arrived.

### Chronological replay outline

```text
for each experiment stream:
    initialise fixed method and rolling ACI from calibration errors
    process issuance and outcome events in availability order

    on forecast issuance:
        generate fixed bounds
        generate ACI bounds from current buffer and alpha_current
        save point forecast, original bounds, issuance and target times

    on actual outcome arrival:
        retrieve the forecast and ACI bounds issued for that target/horizon
        calculate miss against the original ACI bounds
        append absolute issued-forecast error to the rolling buffer
        update alpha_current using the published ACI rule
        mark this feedback event as processed
```

Order events with equal timestamps according to the real information-availability convention. Generating the entire test forecast file offline does not imply that all future outcomes were available at the first forecast issuance.

## 6. Evaluation and required outputs

### Pinball loss for T3

Use the bounds supplied by each UQ method. Keep the nominal evaluation level fixed:

```python
calculate_interval_pinball_loss(
    y_true,
    lower_bound,
    upper_bound,
    alpha=0.10,
)
```

This evaluates the lower endpoint at 0.05 and upper endpoint at 0.95. Do not pass ACI's changing internal `alpha_current` as the scoring level. State which UQ method supplies T3's bounds; report separate results if comparing both.

Adding ACI does not change MAE if point forecasts are unchanged. Pinball loss can change because interval endpoints change.

### T5 and F4

For fixed conformal and ACI, calculate coverage, mean width, normalised width and worst 24-hour coverage by regime. Plot rolling coverage for both methods with the 0.90 target and event markers. Keep the seven-day plot window separate from the 24-hour worst-coverage metric.

Handle `unassigned` timestamps explicitly and consistently. Retain the full chronological stream for rolling plots.

### Infinite intervals

The currently inspected uncertainty validator rejects non-finite bounds. Update the evaluation policy before using strict ACI conventions:

- Valid infinite intervals can be evaluated for coverage.
- Their width is infinite; do not replace it with a finite cap silently.
- Pinball loss may be infinite for infinite endpoints. Handle this explicitly to avoid accidental `NaN` from arithmetic such as infinity multiplied by zero.
- Report the frequency of infinite intervals alongside the required metrics.
- Keep missing values and invalid/reversed bounds distinct from intentionally infinite bounds.

### T6 and F5

Implement a threshold rule using information available at issuance, such as interval width. Sweep thresholds and report review rate and coverage on retained forecasts. Handle the case where no forecasts remain. Mark target coverage as unreachable if no tested operating point achieves it.

An all-false escalation array is a disabled-rule placeholder, not a completed escalation implementation.

## 7. Completion checklist

- [ ] Both files inherit from the actual uncertainty interface.
- [ ] Fixed calibration is performed once and stays fixed.
- [ ] ACI updates its internal quantile level from issued-interval feedback.
- [ ] ACI updates its rolling buffer only from available issued-forecast errors.
- [ ] Buffer length, learning rate and extreme-level policies are documented.
- [ ] Every included forecaster is supported by the same UQ classes.
- [ ] Each forecaster/region/arm/seed/configuration has independent state.
- [ ] Multi-horizon and overlapping-forecast feedback policies are explicit.
- [ ] Forecast blocks respect outcome availability.
- [ ] Learning-rate and buffer-length selection avoid final-test tuning.
- [ ] Evaluation handles infinite intervals explicitly.
- [ ] Fixed and ACI use identical point forecasts.
- [ ] Metrics and curve records are logged for reproducibility.
- [ ] T5/F4 compare both methods; T6/F5 evaluate escalation.
- [ ] Leakage audit verifies issuance and update timing.

## 8. Research references

1. Angelopoulos, A. N., & Bates, S. (2021). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511. Background for split conformal prediction, absolute-residual calibration and coverage assumptions. https://arxiv.org/abs/2107.07511
2. Gibbs, I., & Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift*. Advances in Neural Information Processing Systems, 34, 1660–1672. Original ACI feedback rule. https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html
3. Zaffran, M., Féron, O., Goude, Y., Josse, J., & Dieuleveut, A. (2022). *Adaptive Conformal Predictions for Time Series*. Proceedings of ICML, PMLR 162, 25834–25866. Section 2 describes the ACI update and extreme-level conventions; later sections compare methods. https://proceedings.mlr.press/v162/zaffran22a.html
4. Xu, C., & Xie, Y. (2021). *Conformal prediction interval for dynamic time-series*. Proceedings of ICML, PMLR 139, 11559–11569. Related EnbPI approach; neither of the two proposed classes implements full EnbPI. https://proceedings.mlr.press/v139/xu21h.html

## 9. Project requirement sources

- `drift-project-briefing (1).pdf`: Step 6 requires initial intervals, adaptive conformal evaluation and escalation.
- `sprint-plan-forcasting.pdf`: Sprint 4 begins intervals; Sprint 5 completes uncertainty, escalation, integration and the leakage audit.
- `required-outputs_Drift.pdf`: T3 pinball loss, T5 uncertainty metrics, T6 operating points, F4 coverage trajectory and F5 escalation curve.

The supervisor requires an adaptive conformal method but does not name ACI specifically. ACI is the proposed research-backed choice for the existing forecasting setup.
