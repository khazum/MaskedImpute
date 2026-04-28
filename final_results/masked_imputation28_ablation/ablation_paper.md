## Ablation of Reconstruction-Guided Zero Typing and Compact Latent Design

Ablation study of MaskedImputation28 on the synthetic test split. We start from the final publication-ready configuration and remove or modify one component at a time. Lower is better for avg MSE, avg biozero MSE, and the combined score. Runtime is reported to show whether gains come from materially higher computational cost.

| Variant | avg MSE | avg biozero MSE | Score | Runtime (min) | Delta vs. full |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full model | 0.1845 | 0.0373 | 0.1295 | 20.1 | +0.0000 |
| No recon refinement | 0.2040 | 0.2608 | 0.3628 | 19.0 | +0.2332 |
| No posthoc shrink | 0.1783 | 0.0413 | 0.1305 | 20.7 | +0.0009 |
| No bio regularization | 0.1757 | 0.1850 | 0.2729 | 20.5 | +0.1433 |
| Wider bottleneck | 0.1859 | 0.0373 | 0.1303 | 19.9 | +0.0007 |
| No weight decay | 0.1859 | 0.0371 | 0.1300 | 20.1 | +0.0005 |
| No input noise | 0.1844 | 0.0372 | 0.1294 | 20.3 | -0.0002 |

Key takeaways:
- Reconstruction-guided zero refinement is essential: `No recon refinement` gives score `0.3628`, avg MSE `0.2040`, and avg biozero MSE `0.2608`, versus `0.1295`, `0.1845`, and `0.0373` for the full model.
- Biological regularization is also critical for biologically plausible zeros: `No bio regularization` gives score `0.2729`, avg MSE `0.1757`, and avg biozero MSE `0.1850`, versus `0.1295`, `0.1845`, and `0.0373` for the full model.
- Post-hoc shrink provides a modest but consistent gain in the combined objective: `No posthoc shrink` gives score `0.1305`, avg MSE `0.1783`, and avg biozero MSE `0.0413`, versus `0.1295`, `0.1845`, and `0.0373` for the full model.
- The compact latent bottleneck is helpful but not dominant: `Wider bottleneck` gives score `0.1303`, avg MSE `0.1859`, and avg biozero MSE `0.0373`, versus `0.1295`, `0.1845`, and `0.0373` for the full model.
- Among mild regularizers, weight decay helps slightly, while input noise has negligible effect: `No weight decay` gives score `0.1300`, avg MSE `0.1859`, and avg biozero MSE `0.0371`, while `No input noise` gives score `0.1294`, avg MSE `0.1844`, and avg biozero MSE `0.0372`.
