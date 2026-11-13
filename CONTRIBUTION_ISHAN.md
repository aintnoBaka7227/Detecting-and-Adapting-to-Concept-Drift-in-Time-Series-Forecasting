# Contribution: Ishan

Branch: `docs/S6-contribution-ishan`

Format: branch -> what I built there -> project stage. Only work that was merged into `main` is listed, with the pull request. Line counts are from `git blame` on the final `main`.

## Merged into main

| PR | Branch | What I built | Stage |
|---|---|---|---|
| #2 to #5 | `INS909-patch-1` to `INS909-patch-4` | First upload of the synthetic data generator and its README (patches 3 and 4 only moved a README file) | Synthetic data |
| #7, #8, #9 | `s1-synthetic-data-1` to `3` | Reworked the synthetic generator into `generator.py` with a README | Synthetic data |
| #12 | `s1-generator-batch-outputs` | Batch outputs of the generator: the synthetic series CSVs (sudden, gradual, recurring and no-drift, seeds 1 to 5) and their plots, in `src/drift_forecasting/data/` | Synthetic data |
| #13 | `S2-classical-model` | The original DHR+ARIMA classical baseline on the synthetic and AEMO data, with rolling-MAE outputs (now under `data/archive/baseline/dhr_arima/`), and a generator update. Phong later refactored it into `src/drift_lab/forecasting/dhr_arima.py` on 5 Sep so every baseline shares one interface | Forecasting baseline |
| #37 | `S4-pipeline-integration-and-streamlit` | First version of the pipeline runner `experiments/run_pipeline.py`, the Streamlit results app `app/streamlit_app.py`, and `tests/test_run_pipeline.py` | Pipeline integration |
| #41 | `s5-pipeline-and-stramlit` | Reworked runner (stage selection, dry-run, filters by model, region, seed and detector, clear failure messages) and Streamlit app (one page per stage, method text on the stage page, Comparison tab), with updated tests | Pipeline integration |

In the final `main`, the runner `experiments/run_pipeline.py` is 398 of 414 lines mine, `tests/test_run_pipeline.py` is all 116 lines mine, and `app/streamlit_app.py` is 543 of 961 lines mine (the rest is Phong's).
