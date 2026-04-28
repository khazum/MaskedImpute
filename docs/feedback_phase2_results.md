# Phase 2 Results: Biozero Posterior Correction

Phase 2 verified and corrected the biozero posterior used by MaskImpute.

## Code Changes

- Updated `predict_dropouts_new.py`.
- Added `biozero_posterior_from_nb_dropout(p0_nb, p_drop)`.
- Replaced the previous survival-weighted numerator
  `(1 - p_drop) * p0_nb` with the corrected latent-zero numerator `p0_nb`.
- Added `tests/test_biozero_prior.py`, a lightweight sanity check runnable with
  plain Python because `pytest` is not installed in `magic311`.

The corrected posterior is:

```text
P(T = 0 | Y = 0) = p0_nb / [p_drop + (1 - p_drop) p0_nb]
```

where `T` is the latent pre-dropout count and `Y` is the observed count.

## Verification Commands

```bash
conda run -n magic311 python tests/test_biozero_prior.py
conda run -n magic311 python -m py_compile predict_dropouts_new.py masked_imputation26.py run_imputation.py tests/test_biozero_prior.py
conda run -n magic311 python -c "import run_imputation; print(run_imputation.parse_methods('maskimpute,balanced_mse,masked_imputation26'))"
```

All checks passed.

## Rerun Scope

MaskImpute was rerun on the synthetic test split only:

- Input: `synthetic_datasets/simulated_data/test`
- Method: `balanced_mse` / `MaskImpute`
- Repeats: 5
- Output: `results_imputation_py/balanced_mse/test_phase2_prior_fix`
- Logs: `logs_parallel_runs/maskimpute_phase2_prior_fix`

The first launch used GPUs 0-7. GPUs 4-7 were occupied by external VLLM
processes, causing two initial out-of-memory result rows. Those failed scenarios
were rerun on GPUs 0-1 after the first pass completed. The final Phase 2 result
directory has all 13 scenarios and no error rows.

## Average Metric Changes

Comparison is against the previous MaskImpute artifacts in
`results_imputation_py/balanced_mse/test`.

| Metric | Previous | Phase 2 corrected | Delta | Percent change |
|---|---:|---:|---:|---:|
| MSE | 0.2264506 | 0.2520909 | +0.0256403 | +11.32% |
| Dropout-MSE | 0.3080080 | 0.3520509 | +0.0440429 | +14.30% |
| Biozero-MSE | 0.0117362 | 0.0085033 | -0.0032329 | -27.55% |
| MAE | 0.3503505 | 0.3677790 | +0.0174286 | +4.97% |
| Dropout-MAE | 0.4393145 | 0.4708325 | +0.0315180 | +7.17% |
| Biozero-MAE | 0.0330600 | 0.0258528 | -0.0072072 | -21.80% |
| gNRMSE | 1.1661451 | 1.2069788 | +0.0408337 | +3.50% |
| Marker-MSE | 0.2262978 | 0.2512821 | +0.0249843 | +11.04% |
| Runtime (s) | 135.7085 | 138.4192 | +2.7107 | +2.00% |

Machine-readable summary:

- `results_imputation_py/balanced_mse/test_phase2_prior_fix/summary.tsv`
- `results_imputation_py/balanced_mse/test_phase2_prior_fix/summary.json`

## Interpretation

The corrected posterior increases the estimated probability that an observed
zero is a biological zero because latent zero counts imply observed zeros
regardless of dropout. This makes MaskImpute more conservative at observed
zeros. The expected tradeoff appears in the rerun:

- Biozero error improves substantially.
- Dropout recovery worsens.
- Overall MSE worsens because the benchmark contains many dropout-like entries
  where stronger zero preservation can under-impute true nonzero signal.

## Phase 3 Implication

The corrected prior should be treated as the valid method going forward, but the
fixed hyperparameters were tuned under the older posterior. Phase 3 should
retune only a small number of defensible parameters under the corrected prior,
especially:

- `bio_reg_weight`
- `ZERO_SHRINK_STRENGTH`
- `p_zero`
- `noise_max`

The objective should be to recover the lost overall/dropout MSE while preserving
the improved Biozero-MSE.

