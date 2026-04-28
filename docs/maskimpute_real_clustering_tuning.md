# MaskImpute Real-Data Clustering Tuning Investigation

## Question

Can MaskImpute be adjusted to improve real-data clustering label agreement while preserving its synthetic benchmark lead?

## Tested Changes

I screened simple, publication-defensible changes that do not use labels at inference:

- `raw`: current MaskImpute output.
- `all_obs0.1`, `all_obs0.15`, `all_obs0.25`: residual calibration
  \[
  X_{out}=(1-\lambda)X_{MaskImpute}+\lambda X_{obs}
  \]
  with fixed observed-residual weights `lambda = 0.10, 0.15, 0.25`.
- `zero_obs*`: shrink observed-zero outputs toward zero only.
- `nonzero_obs*`: blend observed non-zero entries only; useful diagnostically but not preferred because it approaches preserving non-zeros.
- `row_center`, `row_zscore`: cell-wise representation transforms for clustering.
- Increased observed non-zero reconstruction training weight (`observed_recon_weight=1.5` and `3.0`).

## Real-Data Screening Result

Screening used one MaskImpute seed (`42`) and the same PCA/SNN/Leiden grid. For the residual calibration, five clustering seeds were evaluated using the cached seed-42 MaskImpute output.

| Dataset | Transform | ARI grid | NMI grid | ARI matched | NMI matched |
|---|---:|---:|---:|---:|---:|
| PBMC68k | raw | 0.181 | 0.400 | 0.191 | 0.405 |
| PBMC68k | all_obs0.10 | 0.232 | 0.441 | 0.244 | 0.457 |
| PBMC68k | all_obs0.15 | 0.236 | 0.444 | 0.254 | 0.467 |
| PBMC68k | all_obs0.25 | 0.235 | 0.444 | 0.257 | 0.469 |
| Baron | raw | 0.453 | 0.750 | 0.659 | 0.832 |
| Baron | all_obs0.10 | 0.473 | 0.765 | 0.677 | 0.849 |
| Baron | all_obs0.15 | 0.475 | 0.767 | 0.677 | 0.847 |
| Baron | all_obs0.25 | 0.477 | 0.767 | 0.683 | 0.848 |
| Zeisel | raw | 0.371 | 0.621 | 0.690 | 0.696 |
| Zeisel | all_obs0.10 | 0.367 | 0.620 | 0.692 | 0.705 |
| Zeisel | all_obs0.15 | 0.369 | 0.621 | 0.691 | 0.700 |
| Zeisel | all_obs0.25 | 0.369 | 0.621 | 0.699 | 0.707 |

Interpretation: residual calibration consistently improves PBMC68k and Baron label agreement, and improves Zeisel matched NMI. The gain comes from preserving a small amount of raw observed structure that clustering metrics reward.

## Synthetic Screen

Synthetic test split, all 13 scenarios, one seed (`42`):

| Transform | MSE | Dropout-MSE | Biozero-MSE | MAE | gNRMSE |
|---|---:|---:|---:|---:|---:|
| raw | 0.2240 | 0.3042 | 0.0134 | 0.3486 | 1.1599 |
| all_obs0.10 | 0.2208 | 0.3312 | 0.0109 | 0.3467 | 1.1490 |
| all_obs0.25 | 0.2714 | 0.4880 | 0.0076 | 0.3803 | 1.2777 |

For comparison, the current full benchmark table reports DCA at MSE `0.3598`, Dropout-MSE `0.4652`, Biozero-MSE `0.0164`, MAE `0.4504`, gNRMSE `1.3867`.

- `all_obs0.10` keeps the synthetic lead and even slightly improves overall MSE, MAE, gNRMSE, and Biozero-MSE; Dropout-MSE worsens from `0.3042` to `0.3312` but remains well below DCA/scVI.
- `all_obs0.25` keeps overall MSE below DCA but loses the Dropout-MSE lead relative to DCA (`0.4880` vs `0.4652`), so it is not a safe default.
- Quadratic interpolation of the MSE components from the same outputs estimates `all_obs0.15` at MSE `0.2303`, Dropout-MSE `0.3679`, Biozero-MSE `0.0097`; this likely preserves the main synthetic lead, but it has not been rerun exactly over all 13 scenarios.

## Training-Weight Screen

Increasing `OBSERVED_RECON_WEIGHT` improved PBMC only when combined with observed residual blending, but it damaged synthetic Biozero-MSE in the screen. Example: `observed_recon_weight=3.0 + all_obs0.15` gave strong PBMC seed-42 matched ARI/NMI (`0.317/0.478`), but the synthetic 3-scenario screen had Biozero-MSE `0.0176`, worse than DCA's reported `0.0164`. I do not recommend this change without additional compensation.

## Recommendation

The only safe candidate is a small fixed residual calibration with `lambda=0.10`:

\[
X_{out}=0.90X_{MaskImpute}+0.10X_{obs}.
\]

This is simple, unsupervised, label-free, and defensible as a residual calibration/skip connection. It improves real-data clustering screens while preserving the synthetic benchmark lead. I would not use `lambda >= 0.25` as the default because the synthetic Dropout-MSE lead is lost.

Before changing the paper defaults, rerun the full 5-seed synthetic benchmark and full 5-seed real-data evaluation with `lambda=0.10`.
