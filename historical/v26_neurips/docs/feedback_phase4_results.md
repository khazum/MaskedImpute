# Phase 4 Results Update

## Scope

Phase 4 adds modern baselines to the synthetic test benchmark and regenerates the paper assets with those methods included.

## Added Baselines

- `scVI`: Python baseline run through `run_imputation.py` with `scvi-tools==1.1.6` in `.venv_scvi`.
- `ALRA`: Python implementation of adaptive low-rank reconstruction added to `run_imputation.py` because the R ALRA/Seurat package stack was not available in `r45_bio`.

Both baselines use fixed settings across all scenarios and evaluate on the same log2(1+CP10k) target scale as the rest of the benchmark.

## Execution

- Runner: `scripts/run_phase4_scvi_alra.sh`
- scVI GPUs: `0,1,2,3`
- Repeats: 5 per scenario
- Scenarios: 13 synthetic test scenarios
- Logs: `logs_parallel_runs/phase4_scvi_alra/`
- Outputs:
  - `results_imputation_py/scvi/test/*/scvi_mse_table.tsv`
  - `results_imputation_py/alra/test/*/alra_mse_table.tsv`

## Generated Assets

Regenerated with `scripts/generate_phase3_benchmark_assets.py`:

- `paper/generated/benchmark_core_table.tex`
- `paper/generated/benchmark_marker_table.tex`
- `paper/generated/benchmark_wins_table.tex`
- `paper/generated/benchmark_significance_table.tex`
- `paper/generated/benchmark_summary.json`
- `paper/figures/mse_5000_bars.pdf`
- `paper/figures/mse_5000_bars.png`
- `results_imputation.md`

## Key Metrics

Mean over 13 synthetic test scenarios, with 5 repeats per scenario:

| Method | MSE | Dropout-MSE | Biozero-MSE | MAE | gNRMSE |
|---|---:|---:|---:|---:|---:|
| MaskImpute | 0.2242 | 0.3053 | 0.0133 | 0.3485 | 1.1592 |
| DCA | 0.3598 | 0.4652 | 0.0164 | 0.4504 | 1.3867 |
| scVI | 0.3426 | 0.4511 | 0.0179 | 0.4355 | 1.3502 |
| ALRA | 0.6917 | 0.7912 | 0.0103 | 0.6498 | 1.8498 |

Relative to scVI, MaskImpute is lower by 34.6% for MSE, 32.3% for Dropout-MSE, 20.0% for MAE, 14.2% for gNRMSE, and 35.5% for Marker-MSE.

## Interpretation

- MaskImpute remains the best retained method on MSE, Dropout-MSE, MAE, gNRMSE, and marker-gene metrics in 12/13 scenarios.
- scVI is the closest neural baseline, but remains behind MaskImpute on the headline reconstruction metrics.
- ALRA is fastest in the current benchmark and relatively conservative on biozero entries, but has higher full-matrix and dropout error.
- Biozero-MSE should still be described carefully: Baseline wins by preserving observed biological zeros exactly, while MaskImpute has lower mean Biozero-MSE than DCA and scVI.
