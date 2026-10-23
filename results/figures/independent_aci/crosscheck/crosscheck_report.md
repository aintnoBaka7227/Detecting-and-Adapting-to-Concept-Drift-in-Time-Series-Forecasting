# Independent Fixed Split / ACI versus AgACI

## Experimental alignment

- Independent runs: A_never_retrain; ACI updates after 48 issued half-hour forecasts; rolling score buffer 336; gamma tuned on historical calibration.
- Figure 5 uses retrospective global ranking by interval width, **not** a deployable online threshold sweep.
- Leader results may use different adaptation arms, detector triggers, feedback, escalation policies, seeds and/or AgACI implementations.
- Do not interpret differences as a causal effect of the uncertainty method until those configurations are aligned.

## Source CSV schemas

- `C:\Users\25472\Documents\Work\Capstone\Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting\leader_figures\results\tables\table5\adwin\table_t5_coverage_and_width_adwin.csv`: ['model', 'arm', 'region', 'UQ method', 'regime', 'coverage', 'mean width (MW)', 'normalised width', 'worst 24h coverage', 'complete 24h windows', 'seeds'] (144 rows)
- `C:\Users\25472\Documents\Work\Capstone\Detecting-and-Adapting-to-Concept-Drift-in-Time-Series-Forecasting\leader_figures\results\tables\table5\page_hinkley\table_t5_coverage_and_width_page_hinkley.csv`: ['model', 'arm', 'region', 'UQ method', 'regime', 'coverage', 'mean width (MW)', 'normalised width', 'worst 24h coverage', 'complete 24h windows', 'seeds'] (144 rows)


## Parsed numerical comparison

- AgACI rows with numeric coverage: 144 / 144
- AgACI rows with numeric mean width: 144 / 144
- Same adaptation-arm comparison rows: 60
- Each AgACI value is the reported point estimate (the ± uncertainty is retained in the source CSV).
- The leader Table 5 contains regime-level and possibly multi-seed aggregates; independent results cover the full test series and one seed. **Do not compare coverage or width deltas as like-for-like performance.**
- For a valid controlled comparison, use matching full-test or matching regime windows, seeds, calibration periods, detector settings and adaptation arms.

## Descriptive comparison (not controlled)

240 matched model-region-method-detector rows exported to `descriptive_comparison_not_controlled.csv`.
Multiple detector and/or arm rows can appear for the same model-region.
Check detector, adaptation arm, seed, calibration and feedback before making any performance claim.

