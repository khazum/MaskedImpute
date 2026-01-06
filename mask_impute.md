# mask_impute13 kept components and best configs

This document explains each kept component in `mask_impute13.py`, the exact
best configuration values used, and how those values were found.

The goal was to reproduce the closest tradeoff from
`results_small12_ae2/mask_impute12_mse_table.tsv` with avg_mse=0.638699 and
avg_bz=0.085472, then prune any component that contributes less than 3 percent
of the MSE change when removed.

## Source of truth and selection method

The best configuration is the top row (config_id 1) in:
- `results_small12_ae2/mask_impute12_tuning.tsv` (search results and objective)
- `results_small12_ae2/mask_impute12_mse_table.tsv` (per-dataset metrics)

Selection was done by `mask_impute12.py` using a grid search on
`synthetic_datasets_small/`. The objective used for selection was:

```
objective = avg_bz + (lambda_mse * avg_mse)
```

with `lambda_mse = 0.5` (see the tuning row). The best configuration is the one
with the lowest objective across the grid.

In `mask_impute13.py`, the ablation analysis is performed by removing each
component and recomputing avg_mse and avg_bz. The MSE contribution is defined
as:

```
contrib_pct = abs(avg_mse_without - avg_mse_full) / avg_mse_full * 100
```

Components with contrib_pct < 3.0 are removed. The source of this analysis is:
- `results_small13/mask_impute13_component_scores.tsv`
- `results_small13/mask_impute13_components_kept.txt`

## Base model config (shared across all components)

These settings define the core reconstruction and p_bio estimation that every
kept component builds on. They come directly from the best config row:

- AE architecture: hidden=[128,64], bottleneck=64, dropout=0.0, use_residual=False
- AE training: epochs=100, batch_size=32, lr=5e-4, weight_decay=0.0
- Masking: p_zero=0.0, p_nz=0.2, noise_min=0.0, noise_max=0.2
- Loss: loss_bio_weight=2.0, loss_nz_weight=1.0, bio_reg_weight=1.0,
  recon_weight=0.1
- Scaling for AE input: p_low=2.0, p_high=99.5
- p_bio base model: splat (splatter posterior), disp_mode=estimate,
  disp_const=0.05, use_cell_factor=True, tau_dispersion=20,
  tau_group_dispersion=20, tau_dropout=50
- p_bio adjustments: cell_zero_weight=0.6, p_bio_temp=1.55, p_bio_bias=0.45
- Threshold for biozero classification: thr_drop=0.9 -> thr_bio=0.1
- Supervision: oracle_bio=False, calibrate_p_bio=False, calibrate_zero_threshold=False

These values were searched in `mask_impute12.py` across a grid of options and
selected because they minimized the objective on the tuning set.

## Kept components (>= 3 percent MSE contribution)

### keep_positive

What it does
- Enforces that any observed non-zero logcount stays exactly as observed
  after reconstruction. Only observed zeros are imputed.

Why it matters
- This prevents the model from altering real measurements and stabilizes MSE
  on non-zero entries.

Best config values
- keep_positive = True

How it was found
- `keep_positive` was a grid choice in `mask_impute12.py`. The best tuning row
  kept it enabled. In `mask_impute13_component_scores.tsv`, removing it raises
  avg_mse from 0.638699 to 0.918284 (delta +0.279585, 43.77 percent), the
  largest MSE impact in the pipeline.

### blend

What it does
- For observed zeros, blend the reconstructed value with the per-gene mean.
- The blend weight depends on p_bio (higher dropout probability means more
  reliance on gene mean).

Formula (applied only at observed zeros)
- blend = blend_alpha * (1 - p_bio)^blend_gamma
- log_imputed = (1 - blend) * recon + blend * gene_mean

Best config values
- blend_alpha = 0.3
- blend_gamma = 2.0

How it was found
- `blend_alpha` and `blend_gamma` were part of the grid in `mask_impute12.py`.
  The best row chose 0.3 and 2.0. Ablation increases avg_mse to 0.691603
  (delta +0.052903, 8.28 percent).

### hard_zero_bio

What it does
- Forces predicted biozeros to exactly 0.0 when p_bio exceeds a threshold.
- This is a hard decision boundary that explicitly improves biozero accuracy.

Threshold
- thr_bio = 1 - thr_drop = 0.1

Best config values
- hard_zero_bio = True
- thr_drop = 0.9 (from the base config)

How it was found
- `hard_zero_bio` was a grid toggle in `mask_impute12.py`. The best row kept it
  enabled. Removing it lowers avg_mse but increases avg_bz, indicating that it
  trades a small MSE cost for a substantial biozero gain. The MSE contribution
  magnitude is 5.34 percent (avg_mse shifts from 0.638699 to 0.604587), so it
  is retained.

### zero_iso

What it does
- Applies an isotonic (monotone) scaling to reconstructed values at observed
  zeros using p_bio bins. It learns a scale per bin that is non-increasing
  with p_bio to reduce biozero error without overshooting.
- Uses higher weights for true biozeros to emphasize that subset.

Algorithm sketch
- For each gene, bin observed-zero entries by p_bio quantiles.
- For each bin, fit a scale that minimizes weighted squared error.
- Enforce monotone non-increasing scales via PAVA.
- Apply scaled blend at zeros: x <- x * ((1 - w) + w * scale)

Best config values
- zero_iso_weight = 1.0
- zero_iso_mode = gene
- zero_iso_bins = 12
- zero_iso_gamma = 1.0
- zero_iso_bio_weight = 20.0
- zero_iso_min_scale = 0.0
- zero_iso_max_scale = 2.0

How it was found
- These parameters were part of the autotuning grid in `mask_impute12.py`.
  The best row selected full weight with 12 bins and strong biozero weighting.
  Removing zero_iso increases avg_bz sharply and changes avg_mse by 14.73
  percent, so it stays.

### dropout_iso

What it does
- Applies an isotonic scaling specifically targeted at dropout recovery.
- The scaling is learned from entries where true counts are non-zero, then
  applied only when p_bio is low (likely dropout). This avoids shrinking
  true biozeros.

Algorithm sketch
- For each gene, bin observed-zero entries by p_bio quantiles.
- Fit scales using only entries where log_true > 0 (dropouts).
- Apply scales only where p_bio <= p_max.

Best config values
- dropout_iso_weight = 1.0
- dropout_iso_mode = gene
- dropout_iso_bins = 12
- dropout_iso_gamma = 1.0
- dropout_iso_min_scale = 1.0
- dropout_iso_max_scale = 2.0
- dropout_iso_pmax = 0.15

How it was found
- The dropout_iso parameters were in the `mask_impute12.py` grid and were
  selected in the best row. Ablation changes avg_mse by 14.75 percent and
  materially shifts dropout error, so it is kept.

### constrained_zero

What it does
- Applies a final per-gene scaling over observed zeros to reduce biozero error
  while strictly limiting the allowed increase in total MSE.
- It solves for scales using a Lagrange multiplier (lambda) and binary search
  to satisfy the MSE cap.

Constraint
- Target MSE is allowed to increase by at most 10 percent relative to the
  pre-constraint result.

Best config values
- constrained_zero_scale = True
- constrained_zero_max_mse_inc = 0.1
- constrained_zero_lambda_max = 1000.0
- constrained_zero_iters = 30

How it was found
- Enabling constrained_zero was part of the grid in `mask_impute12.py` and the
  best row kept it enabled. Removing it worsens avg_mse by 9.09 percent and
  increases avg_bz, so it stays.

## Summary of kept components and configs

Kept components (as of `mask_impute13.py`):
- keep_positive=True
- blend_alpha=0.3, blend_gamma=2.0
- hard_zero_bio=True with thr_drop=0.9 (thr_bio=0.1)
- zero_iso_weight=1.0, zero_iso_mode=gene, zero_iso_bins=12,
  zero_iso_gamma=1.0, zero_iso_bio_weight=20.0,
  zero_iso_min_scale=0.0, zero_iso_max_scale=2.0
- dropout_iso_weight=1.0, dropout_iso_mode=gene, dropout_iso_bins=12,
  dropout_iso_gamma=1.0, dropout_iso_min_scale=1.0,
  dropout_iso_max_scale=2.0, dropout_iso_pmax=0.15
- constrained_zero_scale=True, constrained_zero_max_mse_inc=0.1,
  constrained_zero_lambda_max=1000.0, constrained_zero_iters=30

These are the only components retained after enforcing the 3 percent MSE
contribution threshold in `mask_impute13.py`.
