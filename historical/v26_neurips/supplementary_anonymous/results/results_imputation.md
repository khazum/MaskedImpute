# Phase 4 Synthetic Imputation Benchmark

This summary reflects the Phase 4 regenerated test-split benchmark assets used by `paper/main.tex`, including scVI and ALRA.

## Sources

- MaskImpute: `results_imputation_py/balanced_mse/test_phase2_retuned`
- Python baselines: `results_imputation_py/{dca,scvi,alra,magic,autoclass}/test`
- R baselines: `results_imputation_r/{baseline,ccimpute,saver}/test`
- Generated summary: `paper/generated/benchmark_summary.json`
- Generated paper assets: `paper/generated/*.tex`, `paper/figures/mse_5000_bars.pdf`

## Coverage

- Methods: 9 (`MaskImpute`, `DCA`, `scVI`, `ALRA`, `MAGIC`, `AutoClass`, `ccImpute`, `SAVER`, `Baseline`)
- Synthetic test scenarios: 13
- Repeats: 5 random seeds per scenario
- Result status: all retained methods have complete test-split outputs; no error rows were included.

## Overall Mean Performance

Lower is better for all metrics in this table.

| Method | MSE | Dropout-MSE | Biozero-MSE | MAE | Dropout-MAE | Biozero-MAE | gNRMSE | Marker-MSE | Runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MaskImpute | 0.2242 | 0.3053 | 0.0133 | 0.3485 | 0.4368 | 0.0379 | 1.1592 | 0.2235 | 136.4576 |
| DCA | 0.3598 | 0.4652 | 0.0164 | 0.4504 | 0.5544 | 0.0527 | 1.3867 | 0.3603 | 114.9107 |
| scVI | 0.3426 | 0.4511 | 0.0179 | 0.4355 | 0.5372 | 0.0538 | 1.3502 | 0.3468 | 48.4685 |
| ALRA | 0.6917 | 0.7912 | 0.0103 | 0.6498 | 0.7490 | 0.0421 | 1.8498 | 0.6970 | 0.2556 |
| MAGIC | 0.6411 | 0.7796 | 0.0104 | 0.6256 | 0.7465 | 0.0428 | 1.7747 | 0.6440 | 0.8845 |
| AutoClass | 0.7281 | 0.8621 | 0.0098 | 0.6721 | 0.7833 | 0.0380 | 1.9457 | 0.6662 | 101.9183 |
| ccImpute | 0.6983 | 1.4643 | 0.0095 | 0.5068 | 0.9644 | 0.0144 | 1.6406 | 0.7454 | 26.9013 |
| SAVER | 1.2208 | 2.6786 | 0.0020 | 0.6745 | 1.3590 | 0.0148 | 2.4384 | 1.2286 | 91.4910 |
| Baseline | 1.5236 | 3.3665 | 0.0000 | 0.7343 | 1.4959 | 0.0000 | 2.7044 | 1.5151 | 5.5554 |

## Scenario-Level Wins

- MSE: MaskImpute: 12/13, Baseline: 1/13
- Dropout-MSE: MaskImpute: 12/12
- Biozero-MSE: Baseline: 13/13
- MAE: MaskImpute: 12/13, Baseline: 1/13
- gNRMSE: MaskImpute: 12/13, Baseline: 1/13
- Marker-MSE: MaskImpute: 12/13, Baseline: 1/13
- Marker-MAE: MaskImpute: 12/13, Baseline: 1/13
- Marker-gNRMSE: MaskImpute: 12/13, Baseline: 1/13
- Runtime (s): ALRA: 13/13

## Comparison to DCA and scVI

- vs DCA: MaskImpute is 37.7% lower for MSE, 34.4% lower for Dropout-MSE, 22.6% lower for MAE, and 16.4% lower for gNRMSE.
- vs scVI: MaskImpute is 34.6% lower for MSE, 32.3% lower for Dropout-MSE, 20.0% lower for MAE, and 14.2% lower for gNRMSE.
- Marker-MSE: MaskImpute is 38.0% lower than DCA and 35.5% lower than scVI.
- Biozero-MSE: MaskImpute mean is lower than DCA (0.0133 vs. 0.0164), and lower than scVI (0.0133 vs. 0.0179). This is not a uniform per-scenario advantage over DCA; the one-sided Wilcoxon test for MaskImpute < DCA has p=0.9915.

## Interpretation

- MaskImpute remains the strongest method for full-matrix reconstruction and dropout recovery after adding scVI and ALRA.
- scVI is the closest neural baseline by MSE and Dropout-MSE, but remains worse than MaskImpute on the headline reconstruction metrics.
- ALRA is fast and zero-conservative, but has substantially higher full-matrix and dropout error in this benchmark.
- Biozero-focused metrics remain lowest for identity-like methods because they preserve observed biological zeros exactly; Baseline wins Biozero-MSE and Biozero-MAE on 13/13 scenarios.
- The paper should claim lower mean Biozero-MSE than DCA/scVI, not uniform scenario-level superiority over DCA on biozero entries.

