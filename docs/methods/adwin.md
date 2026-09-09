# ADWIN

## Project role

ADWIN (Adaptive Windowing) is one of the online drift detectors evaluated in
this project. The implementation wraps `river.drift.ADWIN` behind the frozen
project `DriftDetector` interface.

Implementation:

`src/drift_lab/detection/adwin.py`

The required public contract is:

    name = "adwin"
    detect(stream) -> list[int]

The returned values are positional indices of observations at which drift was
flagged.

## Detection mechanism

ADWIN maintains a variable-length window of recent observations. Conceptually,
the window is divided into an older sub-window W0 and a newer sub-window W1.
ADWIN tests whether their statistics remain compatible with a stable stream.

A change is detected when the difference between the sub-window means exceeds
a statistically derived cut threshold:

    |mean(W0) - mean(W1)| > epsilon

The threshold is data-dependent rather than a fixed difference chosen by the
user. The significance parameter `delta` influences the statistical evidence
required for the change decision.

When change is detected, obsolete older observations are removed so that the
adaptive window represents the more recent concept.

River 0.25.0 implements the compressed ADWIN2 form of the algorithm using
buckets rather than retaining an unrestricted raw history.

## Current implementation

The current project wrapper preserves:

    name = "adwin"

and:

    detect(stream) -> list[int]

The wrapper delegates sequential processing to the shared
`detect_with_river()` helper. Each observation is processed in order and the
positional index is recorded whenever River reports drift.

A fresh River ADWIN instance is created for every call to `detect()`, so
runtime state does not persist between independent experiment runs.

## Configuration

The canonical baseline configuration is:

    delta = 0.002

`delta` is a public instance attribute and is therefore captured by
`config_of()` as:

    {"delta": 0.002}

This configuration contributes to the experiment `config_hash`.

River 0.25.0 also provides the defaults:

- `delta = 0.002`
- `clock = 32`
- `max_buckets = 5`
- `min_window_length = 5`
- `grace_period = 10`

The project wrapper currently exposes only `delta`. The remaining River
parameters retain their library defaults because no recorded project evidence
currently justifies changing or exposing them.

## Synthetic evaluation

ADWIN is evaluated against the canonical synthetic generator:

    make_series(kind, n, noise, seed)

The required experiment uses:

- `none`
- `sudden`
- `gradual`
- `recurring`
- `n = 20,000`
- `noise = 1.0`
- seeds `1, 2, 3, 4, 5`

The generator returns both the series and its ground-truth changepoints.
Those changepoints are used directly by the runner rather than hard-coded.

The experiment pipeline is:

    make_series()
        -> ADWINDetector.detect()
        -> config_of()
        -> record_run()
        -> results/runs.csv

Detection metrics are owned by the shared evaluation workflow.
The ADWIN runner must not recompute them independently.

## Expected behaviour

### No drift

Every detection in the no-drift stream is treated as a false alarm.

### Sudden drift

ADWIN should respond after enough post-change evidence accumulates for the
adaptive-window comparison to indicate a statistically significant change.

### Gradual drift

Gradual changes can take longer to detect because the transition develops
progressively rather than at a single abrupt point.

### Recurring drift

The recurring benchmark contains multiple true transitions, so ADWIN is
evaluated on its ability to detect more than one regime change.

## Runtime and configuration trade-offs

ADWIN is intended for online stream processing. River implements ADWIN using
compressed buckets, reducing the need to retain the complete observation
history.

The `clock` parameter controls how frequently change checks are performed.
The project keeps River's default value of 32.

Configuration affects the balance between responsiveness and false alarms.
A detector that reacts more readily may reduce detection delay while producing
more false alarms. A more conservative configuration may reduce false alarms
while increasing delay or missed detections.

The canonical baseline remains:

    delta = 0.002

Any configuration change must be justified using results recorded through the
shared experiment workflow rather than a single preliminary observation.

## AEMO considerations and limitations

The real-data stage uses half-hourly AEMO electricity-demand data for SA1 and
NSW1.

Electricity demand contains daily, weekly and seasonal structure. A
distributional change detected in raw demand therefore does not automatically
mean that forecasting accuracy has deteriorated.

Raw-demand monitoring can identify changes in the observed demand stream,
while forecast-residual monitoring can provide complementary evidence about
changes in predictive performance.

Unlike the synthetic benchmark, the AEMO data does not provide complete
ground-truth drift labels. Unmatched detections on AEMO data must therefore not
automatically be classified as false alarms.

ADWIN also has several practical limitations:

- it identifies statistical change but not the cause of that change;
- detection delay depends on the strength and form of the drift;
- gradual drift may require more observations before sufficient evidence is
  accumulated;
- seasonal demand behaviour may resemble distributional change;
- raw-demand drift does not necessarily imply forecast-model degradation.

## References

Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with
adaptive windowing. Proceedings of the 2007 SIAM International Conference on
Data Mining, 443-448. https://doi.org/10.1137/1.9781611972771.42

Oliveira, M. B., & Silva, L. A. (2026). Two-phase drift monitoring and
automatic retraining (TPDM-AR) for electric load forecasting. MethodsX, 16,
103966. https://doi.org/10.1016/j.mex.2026.103966

Rodríguez, F., Lobo, J. L., Laña, I., Álvarez, V., & Perea, E. (2026).
Intra-hour interval forecasting combined with concept drift adaptation for
electric energy demand at residential level. Knowledge-Based Systems, 116956.
https://doi.org/10.1016/j.knosys.2026.116956
