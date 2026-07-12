# Phase 2 Retune: Corrected-Prior MaskImpute

This retune was performed before Phase 3 so the full benchmark rerun uses a
valid corrected-prior MaskImpute configuration.

## Tuning Protocol

- Tuning split: `synthetic_datasets/simulated_data/tune`
- Validation split: `synthetic_datasets/simulated_data/test`
- Active method key: `balanced_mse`
- Public method name: `MaskImpute`
- Corrected biozero posterior: enabled through `predict_dropouts_new.py`
- GPUs used: 0-3 only; GPUs 4-7 were occupied by external VLLM processes.

Tuning was performed on the tuning split. The test split was used once for the
selected configuration validation.

## Search Scope

The search intentionally stayed small and publishable. The main tradeoff after
correcting the biozero posterior was controlled by the strength of zero
regularization. Large architecture or dataset-specific changes were not used.

Parameters searched:

- `bio_reg_weight`
- `loss_bio_weight`
- `p_zero`
- `zero_shrink_strength`
- `p_nz` in one coarse check
- `observed_recon_weight` in one coarse check

Parameters kept unchanged in the selected configuration:

- architecture: hidden `[128, 64]`, bottleneck `16`
- epochs: `400`
- learning rate: `1e-4`
- `p_zero`: `0.01`
- `p_nz`: `0.05`
- `noise_max`: `0.02`
- `loss_nz_weight`: `2.0`
- `zero_shrink_strength`: `0.3`
- `observed_recon_weight`: `0.75`

## Selected Configuration

Stored at `configs/maskimpute_phase2_retuned.json`.

```json
{
  "bio_reg_weight": 1.2,
  "epochs": 400,
  "loss_bio_weight": 1.0,
  "loss_nz_weight": 2.0,
  "lr": 0.0001,
  "noise_max": 0.02,
  "observed_recon_weight": 0.75,
  "p_nz": 0.05,
  "p_zero": 0.01,
  "zero_shrink_strength": 0.3
}
```

This was selected because it recovered almost all of the lost overall/dropout
MSE while keeping average Biozero-MSE below DCA's current benchmark value
(`0.0164`).

## Tuning-Split Finalist Results

Finalists were evaluated on the tuning split with 3 repeats.

| Config | MSE | Dropout-MSE | Biozero-MSE | MAE | gNRMSE |
|---|---:|---:|---:|---:|---:|
| selected: `bio_reg_weight=1.2`, `loss_bio_weight=1.0` | 0.233282 | 0.309888 | 0.015582 | 0.359567 | 1.1720 |
| `bio_reg_weight=1.2`, `p_zero=0.005` | 0.233343 | 0.309895 | 0.015583 | 0.359622 | 1.1721 |
| `bio_reg_weight=1.25`, `loss_bio_weight=1.0` | 0.235065 | 0.312743 | 0.015105 | 0.360835 | 1.1752 |
| `bio_reg_weight=1.3`, `p_zero=0.005` | 0.237045 | 0.315813 | 0.014621 | 0.362250 | 1.1787 |
| corrected default | 0.261766 | 0.354756 | 0.009957 | 0.379330 | 1.2198 |

The `p_zero=0.005` finalist was effectively tied, but the selected configuration
keeps the masking rate unchanged and changes only the zero-loss weighting.

## Held-Out Test Validation

Output directory:

- `results_imputation_py/balanced_mse/test_phase2_retuned`

Comparison against previous artifacts:

| Metric | Pre-fix old config | Corrected default | Retuned corrected | Vs corrected default | Vs pre-fix |
|---|---:|---:|---:|---:|---:|
| MSE | 0.2264506 | 0.2520909 | 0.2241589 | -11.08% | -1.01% |
| Dropout-MSE | 0.3080080 | 0.3520509 | 0.3053305 | -13.27% | -0.87% |
| Biozero-MSE | 0.0117362 | 0.0085033 | 0.0133131 | +56.56% | +13.44% |
| MAE | 0.3503505 | 0.3677790 | 0.3484975 | -5.24% | -0.53% |
| Biozero-MAE | 0.0330600 | 0.0258528 | 0.0378908 | +46.56% | +14.61% |
| gNRMSE | 1.1661451 | 1.2069788 | 1.1591544 | -3.96% | -0.60% |
| Marker-MSE | 0.2262978 | 0.2512821 | 0.2234953 | -11.06% | -1.24% |

Machine-readable comparison:

- `results_imputation_py/balanced_mse/test_phase2_retuned/comparison_summary.tsv`
- `results_imputation_py/balanced_mse/test_phase2_retuned/comparison_summary.json`

## Code Updates

- `masked_imputation26.py`
  - `loss_bio_weight`: `2.0 -> 1.0`
  - `bio_reg_weight`: `2.0 -> 1.2`
- `run_imputation.py`
  - no longer hardcodes `bio_reg_weight=2.0`; it uses the implementation default.
- `paper/main.tex`
  - method parameters updated to the retuned values.

## Phase 3 Decision

Proceed to Phase 3 using `configs/maskimpute_phase2_retuned.json` / the updated
`masked_imputation26.py` defaults. The retuned corrected-prior configuration is
better than both the corrected default and the previous pre-fix configuration on
overall MSE, dropout-MSE, MAE, gNRMSE, and marker-MSE, while remaining below DCA
on Biozero-MSE.

