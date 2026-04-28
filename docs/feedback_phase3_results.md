# Phase 3 Results Update

## Scope

Phase 3 refreshes the synthetic benchmark evidence in `paper/` using the corrected and retuned MaskImpute v26 configuration.

## Inputs

- MaskImpute test split: `results_imputation_py/balanced_mse/test_phase2_retuned`
- Existing retained baselines: `results_imputation_py/{dca,magic,autoclass}/test` and `results_imputation_r/{baseline,ccimpute,saver}/test`
- Asset generator: `scripts/generate_phase3_benchmark_assets.py`

## Generated Assets

- Tables: `paper/generated/benchmark_core_table.tex`, `paper/generated/benchmark_marker_table.tex`, `paper/generated/benchmark_wins_table.tex`, `paper/generated/benchmark_significance_table.tex`
- Per-scenario tables: `paper/generated/benchmark_*_by_scenario.tex`
- Figure: `paper/figures/mse_5000_bars.pdf` and `paper/figures/mse_5000_bars.png`
- Summary: `paper/generated/benchmark_summary.json`
- Markdown summary: `results_imputation.md`

## Key Metrics

Mean over 13 synthetic test scenarios, with 5 repeats per scenario:

| Method | MSE | Dropout-MSE | Biozero-MSE | MAE | gNRMSE |
|---|---:|---:|---:|---:|---:|
| MaskImpute | 0.2242 | 0.3053 | 0.0133 | 0.3485 | 1.1592 |
| DCA | 0.3598 | 0.4652 | 0.0164 | 0.4504 | 1.3867 |

Relative to DCA, MaskImpute is lower by 37.7% for MSE, 34.4% for Dropout-MSE, 22.6% for MAE, 16.4% for gNRMSE, and 38.0% for Marker-MSE.

## Statistical Interpretation

- MaskImpute wins scenario-level MSE, Dropout-MSE, MAE, gNRMSE, and marker-gene metrics in 12/13 scenarios; the identity baseline wins the no-drop scenario.
- MaskImpute has lower mean Biozero-MSE than DCA (0.0133 vs. 0.0164), but this is not a uniform scenario-level advantage; the one-sided Wilcoxon test for MaskImpute < DCA has p=0.9915.
- Biozero-MSE is lowest for identity-like methods because they preserve observed biological zeros exactly; this should be described explicitly rather than framed as a failure of MSE non-negativity or error-bar plotting.

## Paper Updates

Updated `paper/main.tex` to:

- Report the retuned Phase 3 MaskImpute results.
- Use regenerated core and marker tables.
- Add scenario-level win counts and paired Wilcoxon tests against DCA.
- Clarify that error bars in the aggregate figure are standard deviations across scenarios and are clipped at zero for non-negative metrics.
- Avoid claiming uniform Biozero-MSE superiority over DCA.

## Remaining Follow-Up

- scVI and ALRA were added in Phase 4; see `docs/feedback_phase4_results.md`.
- Runtime scaling was not regenerated in this phase, per instruction.
