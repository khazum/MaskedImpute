# MaskImpute Residual-Denoising Trial

## Implemented Experimental Paths

I added two disabled-by-default experimental options to `masked_imputation26.py`:

1. `residual_correction_scale`: strict residual-denoising decoder
   \[
   \hat{s}=a+\tau\tanh(d_\theta(\tilde{s}))
   \]
   where the anchor `a` is neutral for masked nonzeros and the scaled-zero value for masked zero-noise entries. This avoids masked-value leakage.

2. `residual_skip_weight`: trained residual skip
   \[
   \hat{s}=(1-\rho)d_\theta(\tilde{s})+\rho a
   \]
   where the skip is present during both training and inference and uses the same leakage-safe anchor during masking.

Both are currently off by default, so the published/current MaskImpute behavior is unchanged unless a config explicitly enables them.

## Synthetic Screen

Three-scenario synthetic screen (`groups_balanced_moderate_drop`, `groups_rare_high_drop`, `batch_effects_moderate_drop`; seed 42):

| Config | MSE | Dropout-MSE | Biozero-MSE | MAE | gNRMSE | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Current raw reference | 0.2567 | 0.3311 | 0.0118 | 0.3752 | 1.2190 | baseline for screen |
| strict residual `scale=0.5, reg=0.01` | 1.6781 | 2.5981 | 0.0001 | 0.9923 | 2.9634 | reject |
| strict residual `scale=1.0, reg=0.01` | 1.6246 | 2.5664 | 0.0001 | 0.9724 | 2.8993 | reject |
| trained skip `rho=0.10` | 0.2688 | 0.3568 | 0.0094 | 0.3903 | 1.2620 | keeps broad lead, weaker than current |
| trained skip `rho=0.15` | 0.2880 | 0.3892 | 0.0083 | 0.4089 | 1.3150 | likely still below DCA, weaker |
| trained skip `rho=0.25` | 0.3524 | 0.4944 | 0.0064 | 0.4611 | 1.4640 | reject; loses dropout lead |

The strict residual-correction architecture collapses toward the observed-zero anchor and fails dropout recovery. This is the core issue: in unsupervised real use, observed zeros do not reveal which zeros are technical dropouts, so a correction-only residual model gets insufficient signal to raise them.

## Real-Data Clustering Screen

Seed-42 real-data screen for trained skip configs:

| Dataset | Config | ARI grid | NMI grid | ARI matched | NMI matched | Verdict |
|---|---:|---:|---:|---:|---:|---|
| PBMC68k | current raw reference | 0.183 | 0.402 | 0.193 | 0.406 | reference |
| PBMC68k | trained skip `rho=0.10` | 0.185 | 0.407 | 0.209 | 0.437 | small improvement, not top |
| PBMC68k | trained skip `rho=0.15` | 0.187 | 0.411 | 0.211 | 0.420 | small improvement, not top |
| PBMC68k | trained skip `rho=0.25` | 0.195 | 0.419 | 0.213 | 0.441 | better, but synthetic dropout lead fails |
| Baron | trained skip `rho=0.10` | 0.452 | 0.747 | 0.671 | 0.837 | no meaningful improvement |
| Zeisel | trained skip `rho=0.10` | 0.360 | 0.621 | 0.702 | 0.704 | mixed; matched improves, grid drops |

For comparison, the earlier fixed post-hoc residual calibration `0.90*MaskImpute + 0.10*Observed` improved PBMC68k much more (`ARI/NMI grid ~0.232/0.441`) and preserved the full 13-scenario synthetic lead. Training the same idea as an architectural skip did not reproduce that gain.

## Conclusion

The more defensible strict residual-denoising formulation does **not** work for this task: it destroys synthetic dropout recovery. The trained skip formulation is safer, but it does not deliver top real-data clustering performance; stronger skips improve PBMC clustering but start sacrificing the synthetic Dropout-MSE lead.

Recommendation: do **not** adopt residual denoising as the default. Keep the current MaskImpute model for the paper. If real-data clustering is prioritized, the only currently effective option remains the small fixed residual calibration (`lambda=0.10`) as a clearly labeled optional representation-calibration step, not as the core imputation output.
