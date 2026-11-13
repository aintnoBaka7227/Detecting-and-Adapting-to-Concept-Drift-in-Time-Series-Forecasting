# Contribution – Bingqi Yan

## Merged into final main

### `feature/f2-synthetic-visual-companion`
**Stage:** Detection / Synthetic benchmark visualisation

Implemented the synthetic drift visual companion used to present detector behaviour on the synthetic benchmark. The work included aligning the synthetic figures with the frozen detector results and briefing format, validating the synthetic split provenance, separating cross-platform result file locking, and adding split-aware synthetic runs and figures.

**Merged into main via Pull Request #24.**

Main commits include:
- `3baf265` – feat(f2): add synthetic drift visual companion
- `ccc291b` – fix(f2): align synthetic visual companion with briefing format
- `bdbd988` – refactor(results): separate cross-platform file locking
- `6d2d739` – fix(f2): align synthetic figure with frozen detector results
- `ffd7985` – fix(f2): validate frozen synthetic split provenance
- `cf559e4` – Add F2 synthetic figures and split-aware synthetic runs

A later update to show all five seeds in the synthetic figures was completed on the branch but was not merged into final main.

---

## Completed work not merged into final main

### `feature/fine-tune-drift-detectors`
**Stage:** Detection

Independently fine-tuned and validated drift-detector settings on the synthetic benchmark. Produced post-tuning detector comparison results, validation outputs, and synthetic detection figures.

**Pull Request #33.**
**Status:** Completed, but not merged into final main.

---

### `feature/arm-c-full-history`
**Stage:** Adaptation

Implemented Arm C full-history retraining and Arm D recent-window retraining, integrated the adapters with the AEMO adaptation runner, and produced validation and evaluation outputs for Table 3, Table 4, and Figure 3.

**Pull Request #35.**
**Status:** Completed, but not merged into final main.

---

### `feature/S4-bingqi`
**Stage:** Adaptation / Forecasting

Extended the Sprint 4 adaptation work by adding Bingqi implementations of DHR-ARIMA and NHITS, updating regime evaluation, running the adaptation experiments, and regenerating Table 3, Table 4, and Figure 3 results.

**Status:** Completed, but not merged into final main.

---

### `feature/t7-f6-bingqi-final`
**Stage:** Cross-team trigger comparison

Produced an independent Table 7 and Figure 6 result set for the Team 41 and Team 42 trigger comparison.

The **Figure 6** work from this branch was later referenced and used in the final project comparison, although the branch itself was not merged into final main.

The **Table 7** work was completed on this branch but was not used in the final project and was not merged into final main.

Files produced on the branch include:
- `make_t7_png_custom.py`
- `make_f6_png_custom.py`
- `results/tables/table7/T7_custom.csv`
- `results/tables/table7/T7_custom_clean.png`
- `results/figures/figure6/bingqi/F6_custom_final.png`
- `results/figures/figure6/bingqi/F6_disagreement_summary.csv`

**Status:** Completed; Figure 6 was referenced in the final project, while Table 7 was not used. The branch was not merged into final main.