# Phase 6: Ablations, Sensitivity, and Downstream Synthetic Tasks

Efficacy score is `MSE + 2*Biozero-MSE`; lower is better. Values below are means over synthetic test scenarios unless noted.

## Ablation Summary

| Variant | Efficacy | MSE | Biozero-MSE | Dropout-MSE | ARI matched | NMI matched |
|---|---:|---:|---:|---:|---:|---:|
| MaskImpute | 0.251 | 0.224 | 0.013 | 0.304 | 0.718 | 0.764 |
| No zero shrinkage | 0.263 | 0.228 | 0.018 | 0.312 | 0.716 | 0.762 |
| No biozero regularization | 0.360 | 0.171 | 0.095 | 0.190 | 0.766 | 0.816 |
| Uniform masking | 0.319 | 0.305 | 0.007 | 0.441 | 0.699 | 0.734 |
| Plain masked AE | 0.703 | 0.682 | 0.010 | 0.792 | 0.639 | 0.676 |

## Calibration Search

Residual calibration uses `X_out=(1-lambda) X_MaskImpute + lambda X_observed`; it is label-free and has one scalar parameter.

| Output | Efficacy | MSE | Biozero-MSE | Dropout-MSE | ARI matched | NMI matched |
|---|---:|---:|---:|---:|---:|---:|
| MaskImpute | 0.251 | 0.224 | 0.013 | 0.304 | 0.718 | 0.764 |
| $0.95X_{MI}+0.05X_{obs}$ | 0.243 | 0.219 | 0.012 | 0.310 | 0.723 | 0.766 |
| $0.90X_{MI}+0.10X_{obs}$ | 0.243 | 0.221 | 0.011 | 0.331 | 0.740 | 0.772 |
| $0.85X_{MI}+0.15X_{obs}$ | 0.250 | 0.230 | 0.010 | 0.368 | 0.721 | 0.760 |
| $0.80X_{MI}+0.20X_{obs}$ | 0.264 | 0.247 | 0.009 | 0.420 | 0.730 | 0.769 |
| $0.75X_{MI}+0.25X_{obs}$ | 0.287 | 0.271 | 0.008 | 0.488 | 0.725 | 0.767 |
| Observed input | 1.524 | 1.524 | 0.000 | 3.366 | 0.642 | 0.663 |

## One-Scenario Sensitivity

Representative scenario: `groups_balanced_moderate_drop`.

| Parameter | Value | Efficacy | MSE | Biozero-MSE | Dropout-MSE | ARI matched |
|---|---:|---:|---:|---:|---:|---:|
| Shrinkage $\alpha$ | 0 | 0.255 | 0.221 | 0.017 | 0.292 | 0.996 |
| Shrinkage $\alpha$ | 0.1 | 0.250 | 0.218 | 0.016 | 0.287 | 0.995 |
| Shrinkage $\alpha$ | 0.2 | 0.246 | 0.217 | 0.014 | 0.285 | 0.995 |
| Shrinkage $\alpha$ | 0.3 | 0.245 | 0.218 | 0.013 | 0.287 | 0.995 |
| Shrinkage $\alpha$ | 0.5 | 0.248 | 0.227 | 0.011 | 0.306 | 0.993 |
| Shrinkage $\alpha$ | 0.7 | 0.264 | 0.247 | 0.008 | 0.350 | 0.993 |
| Shrinkage $\alpha$ | 1 | 0.319 | 0.308 | 0.005 | 0.479 | 0.992 |
| Biozero weight $\gamma$ | 0 | 0.361 | 0.166 | 0.097 | 0.184 | 0.989 |
| Biozero weight $\gamma$ | 0.5 | 0.241 | 0.192 | 0.025 | 0.244 | 0.998 |
| Biozero weight $\gamma$ | 1 | 0.242 | 0.211 | 0.015 | 0.276 | 0.996 |
| Biozero weight $\gamma$ | 1.2 | 0.245 | 0.218 | 0.013 | 0.287 | 0.995 |
| Biozero weight $\gamma$ | 2 | 0.260 | 0.243 | 0.008 | 0.328 | 0.995 |
| Biozero weight $\gamma$ | 5 | 0.307 | 0.300 | 0.004 | 0.414 | 0.983 |
| Zero mask rate $p_0$ | 0.005 | 0.244 | 0.216 | 0.014 | 0.284 | 0.996 |
| Zero mask rate $p_0$ | 0.01 | 0.245 | 0.218 | 0.013 | 0.287 | 0.995 |
| Zero mask rate $p_0$ | 0.02 | 0.246 | 0.222 | 0.012 | 0.293 | 0.995 |
| Zero mask rate $p_0$ | 0.05 | 0.251 | 0.229 | 0.011 | 0.307 | 0.991 |

## Interpretation

- The biozero regularizer is the main component protecting biological zeros; removing it lowers aggregate MSE but substantially worsens Biozero-MSE.
- Zero shrinkage provides a smaller but consistent Biozero-MSE gain with little effect on downstream synthetic clustering.
- Uniform masking is not a good default: it can reduce Biozero-MSE but increases full-matrix and dropout errors.
- Residual calibration is simpler than earlier clustering-specific post-processing because it is a fixed convex combination with the observed matrix and no labels or graph construction.

## Calibration Recommendation

- Keep uncalibrated MaskImpute as the primary denoising output for the headline benchmark because it has the best Dropout-MSE.
- If a single calibrated output is needed, `lambda=0.05` is the safest synthetic-error trade-off: efficacy improves from `0.251` to `0.243`, MSE improves from `0.224` to `0.219`, and Dropout-MSE only changes from `0.304` to `0.310`.
- If downstream label agreement is prioritized, `lambda=0.10` has the best matched synthetic ARI/NMI among the screened values while preserving clear leads over DCA on MSE, Dropout-MSE, and Biozero-MSE.
