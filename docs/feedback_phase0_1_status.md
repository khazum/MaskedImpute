# Phase 0-1 Status

This document records the Phase 0 audit and Phase 1 manuscript cleanup state for the feedback implementation plan in `docs/feedback_implementation_plan.md`.

## Phase 0 Audit

### Active Implementation Path

- Public method name: `MaskImpute`.
- Historical result key: `balanced_mse`.
- Active implementation file: `masked_imputation26.py`.
- Dispatcher: `run_imputation.py` maps `balanced_mse`, `maskimpute`, `maskedimpute`, `masked_impute`, and `masked_imputation26` to the same implementation.
- Parallel runner: `run_parallel_imputations_dgx.sh` still uses `balanced_mse` as the on-disk method key, but displays it as `MaskImpute`.

### Existing Synthetic Result Artifacts

Current result files are present for the synthetic benchmark protocol:

| Result root | Method | Test files | Scaling files |
|---|---:|---:|---:|
| `results_imputation_py` | `balanced_mse` / MaskImpute | 13 | 10 |
| `results_imputation_py` | `dca` | 13 | 10 |
| `results_imputation_py` | `magic` | 13 | 10 |
| `results_imputation_py` | `autoclass` | 13 | 10 |
| `results_imputation_r` | `baseline` | 13 | 10 |
| `results_imputation_r` | `ccimpute` | 13 | 10 |
| `results_imputation_r` | `saver` | 13 | 10 |

The current generated benchmark summary in `paper/generated/benchmark_summary.json` reports MaskImpute mean test-split metrics over 13 scenarios:

| Metric | Value |
|---|---:|
| MSE | 0.2264506 |
| Dropout-MSE | 0.3080080 |
| Biozero-MSE | 0.0117362 |
| MAE | 0.3503505 |
| gNRMSE | 1.1661451 |

These values were produced before the requested final rerun after the biozero posterior review item. Treat them as current artifacts, not final post-feedback results.

### Existing Paper Artifacts

- Main manuscript: `paper/main.tex`.
- Current PDF: `paper/main.pdf`.
- NeurIPS style file present: `paper/neurips_2026.sty`.
- Checklist present: `paper/checklist.tex`.
- Generated tables present in `paper/generated/`.
- Figures present in `paper/figures/`.

### Phase 0 Findings

| Area | Finding | Status |
|---|---|---|
| Template | `paper/main.tex` uses `neurips_2026`; no UAI style is loaded in the active paper. | OK |
| Naming | Active code uses `MaskImpute`, but active paper/generated tables still used `MaskClass`. | Fixed in Phase 1 |
| Unsupported placeholders | Real-data, scVI, ALRA, downstream, ablation, and sensitivity sections had TODO placeholders. | Partially fixed in Phase 1; experiments remain Phase 3-6 |
| Clustering evaluation | Paper contained synthetic and real clustering evaluation placeholders. | Removed from active methods/results in Phase 1 |
| Biozero posterior | Code and paper now use the corrected posterior form; Phase 2 rerun results are recorded in `docs/feedback_phase2_results.md`. | Done Phase 2 |
| Standard deviations | Active generated tables now omit placeholder standard deviations; repeat-level statistics still need regeneration. | Pending Phase 3 |
| Modern baselines | scVI and ALRA rows existed as placeholders only. | Removed from active tables; pending Phase 4 |
| Real-data experiments | Placeholder section existed without computed data. | Removed from active results; pending Phase 5 |
| Ablations/sensitivity | Placeholder sections existed without computed data. | Removed from active paper text; pending Phase 6 |
| Runtime scaling | Scaling artifacts exist and are isolated to the runtime appendix. | Kept; no scaling rerun in Phase 1 |

## Feedback Item Tracking

| Feedback item | Phase | Current state |
|---|---:|---|
| Convert to NeurIPS template and remove UAI footer | 1 | Active paper uses NeurIPS style; local compile could not run because no TeX engine is installed |
| Remove de-anonymizing language | 1 | No active "our previous work" phrasing found in `paper/main.tex`; related-method citations remain third-person |
| Rename method consistently to MaskImpute | 1 | Done for active paper and generated labels |
| Remove unsupported real-data claims | 1 | Done from abstract, contributions, and active results |
| Remove clustering evaluation methods/results | 1 | Done from active methods/results |
| Fix title/abstract overclaims | 1 | Done; title no longer uses "accurate" or "scalable" |
| Fix biozero posterior math/code | 2 | Done; corrected helper and sanity check added |
| Rerun corrected synthetic benchmark | 3 | MaskImpute-only Phase 2 rerun done; full benchmark rerun pending |
| Add standard deviations/significance | 3 | Pending |
| Add per-scenario result verification | 3 | Existing per-scenario tables present; need post-rerun verification |
| Add scVI and ALRA | 4 | Pending |
| Add real-data experiments | 5 | Pending |
| Add ablations and sensitivity | 6 | Pending |
| Final paper integration and page-budget check | 7 | Pending |

## Phase 1 Cleanup Applied

- Renamed the active manuscript method from `MaskClass` to `MaskImpute`.
- Removed unsupported real-data benchmark claims from the abstract and contribution list.
- Removed clustering, trajectory, real-data correlation, and real-data DE evaluation paragraphs from the active methods section.
- Removed the placeholder real-data benchmark section and placeholder downstream task table from active results/appendix text.
- Removed scVI and ALRA placeholder rows from active generated benchmark tables until those baselines are computed.
- Removed placeholder standard-deviation markers from active generated tables until repeat-level statistics are regenerated.
- Kept runtime scaling isolated to the appendix because scaling artifacts exist and the user previously requested not to update scaling results yet.

## Remaining Blockers Before Paper-Ready Results

1. Verify and, if needed, correct `masked_imputation26.py` for the reviewed biozero posterior formula.
2. Rerun MaskImpute on the synthetic test split after the posterior check.
3. Rerun all retained synthetic baselines with standard deviations and marker-gene metrics.
4. Add scVI and ALRA or downgrade baseline-coverage claims.
5. Generate ablation and sensitivity results.
6. Add real-data experiments only after metrics are computed.
