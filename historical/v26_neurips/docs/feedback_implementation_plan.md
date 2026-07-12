# Feedback Implementation Plan

This plan tracks the work needed to implement the reviewer/action-item feedback in `feedback/Action_Items.md.pdf` against the current NeurIPS-style manuscript in `paper/`.

## Current Status

- Phase 0--1 completed; see `docs/feedback_phase0_1_status.md`.
- Phase 2 completed; see `docs/feedback_phase2_results.md`.
- Retuning completed after Phase 2; see `docs/feedback_phase2_retune.md`.
- Phase 3 completed for the retained methods; see `docs/feedback_phase3_results.md`.
- Phase 4 completed; see `docs/feedback_phase4_results.md`.
- Phase 5 completed with label-free HVG real-data clustering grid, PBMC68k, batch diagnostics, and DCA time-budget reporting; see `docs/feedback_phase5_results.md`.
- Phase 6 completed with ablations, one-scenario sensitivity, residual-calibration sweep, and synthetic downstream clustering diagnostics; see `docs/feedback_phase6_results.md`.
- Phase 7 completed with artifact regeneration, PDF compilation, numerical-claim checks, citation/source checks, and obsolete-term checks; see `docs/feedback_phase7_verification.md`.
- Runtime scaling was not regenerated in Phase 3, per instruction.

## Guiding Principles

- Keep the paper centered on MaskImpute as a publishable imputation method, not on obsolete MaskClass/clustering variants.
- Treat the biozero prior correction as a blocking methodological fix: update the math, update the code, then rerun the benchmark before finalizing claims.
- Separate paper-compliance edits from experiment-generation work so the manuscript can improve while longer computations run.
- Only make claims supported by regenerated tables, figures, and logs in the repository.

## Phase 0: Audit Current State

Goal: establish exactly what is already implemented, computed, and still missing.

- Create a status matrix covering every feedback item: `paper edit`, `code change`, `experiment`, `figure/table`, or `verification`.
- Inventory existing results under `results_imputation_py/`, `results_imputation_r/`, `paper/generated/`, and `paper/figures/`.
- Confirm the active implementation path: `balanced_mse` / `maskimpute` routes to `masked_imputation26.py`.
- Record the exact benchmark data split used by each current table and figure.
- Identify paper text that still says MaskClass, Balanced-MSE, clustering, UAI, or obsolete variant names.

Deliverable: short checklist file or section in this document with done/missing status for each action item.

## Phase 1: Compliance And Manuscript Cleanup

Goal: handle fast, low-risk paper corrections before rerunning experiments.

- Confirm the paper compiles with the NeurIPS 2026 template and remove any remaining UAI footer/template artifacts.
- Remove or rewrite de-anonymizing language such as references to "our previous work" and ccImpute.
- Update the title, abstract, and introduction to avoid unsupported claims such as cross-modality generality or "scalable" unless backed by results.
- Standardize terminology to `MaskImpute` throughout the paper, generated captions, method labels, and figure labels.
- Remove clustering-related methods/results unless explicitly reintroduced as a validated downstream task experiment.
- Fix notation conflicts:
  - avoid using `d` for both bottleneck dimension and dropout indicator;
  - avoid using `B` for both mini-batch and biozero sets;
  - separate shrinkage weights from loss weights.
- Fix minor typographic issues noted in feedback:
  - dangling `m` in masking equation;
  - duplicate "Algorithm 1";
  - inconsistent `p_bio` notation;
  - negative Biozero-MSE axis display;
  - missing baseline definitions in captions;
  - missing hardware statement.
- Ensure the NeurIPS checklist is included and filled consistently with the experiments actually reported.

Deliverable: manuscript compiles cleanly, with naming and formatting issues resolved.

## Phase 2: Correct Biozero Prior Math And Code

Goal: fix the methodological issue before final benchmark claims are regenerated.

- Update the paper's biozero posterior equation. For an observed zero, use the correct biological-zero numerator:

  ```text
  p_bio(i,j) = f_ij(0) / [delta_ij + (1 - delta_ij) f_ij(0)]
  ```

  where `f_ij(0)` is the latent count probability at zero and `delta_ij` is dropout probability.

- Update `masked_imputation26.py` so the implemented prior matches the corrected equation.
- Add a small verification script or test that compares the implementation to manually computed posterior values for representative `delta` and `f0`.
- Rerun MaskImpute on the synthetic test split using the corrected implementation.
- Compare corrected results against the current reference values:
  - Avg MSE: `0.2264506`
  - Avg Biozero-MSE: `0.01173623`
  - Avg Dropout-MSE: `0.3080080`

Deliverable: corrected code, corrected method text, and regenerated MaskImpute-only synthetic test results.

## Phase 3: Regenerate Synthetic Benchmark Evidence

Goal: rebuild the main benchmark around the corrected implementation and updated metric definitions.

- Rerun all methods on the synthetic test split using the updated benchmark protocol in `benchmark_datasets_and_metrics.md`.
- Include all required metrics:
  - MSE;
  - MAE;
  - gNRMSE;
  - Biozero-MSE;
  - Biozero-MAE;
  - Biozero-gNRMSE;
  - Dropout-MSE;
  - Dropout-MAE;
  - Dropout-gNRMSE;
  - marker-gene versions of MSE, MAE, and gNRMSE where required.
- Compute per-scenario means and standard deviations over repeats.
- Verify or revise claims such as "wins on 12/13 scenarios".
- Add statistical tests across scenarios or repeats where appropriate, preferably paired tests such as Wilcoxon signed-rank for headline comparisons.
- Regenerate all paper tables and figures from result files rather than manually editing numbers.
- Explicitly discuss gNRMSE values above 1 when they occur.

Deliverables:

- Updated `results_imputation.md`.
- Updated files in `paper/generated/`.
- Updated figures in `paper/figures/`.
- Updated result tables in `paper/main.tex`.

## Phase 4: Add Modern Baselines

Goal: address the baseline gap before making competitiveness claims.

- Add scVI as a Python baseline if the dependency stack is available in the current environment or a reproducible conda environment can be created.
- Add ALRA as an R baseline through the existing R execution path.
- Define each baseline's input/output scale clearly:
  - whether it outputs counts, logcounts, or normalized expression;
  - whether target-sum normalization is applied;
  - how predictions are converted to benchmark scale.
- Run scVI and ALRA on the synthetic test split.
- Include them in the same summary tables and per-scenario tables as existing methods.
- Reassess wording such as "state-of-the-art" after these baselines are included.

Deliverable: synthetic benchmark tables include DCA, MAGIC, SAVER, AutoClass if retained, scVI, ALRA, and MaskImpute.

## Phase 5: Add Real-Data Experiments

Goal: add externally recognizable datasets and downstream evidence requested by feedback.

- Prepare real datasets:
  - PBMC 68k;
  - Baron human pancreas;
  - Zeisel mouse cortex.
- Define a reproducible preprocessing pipeline:
  - gene filtering;
  - count/logcount construction;
  - target-sum normalization;
  - train/test or full-data transductive usage;
  - labels used only for evaluation.
- Run MaskImpute and selected baselines on each dataset.
- Compute label-based clustering metrics after imputation:
  - Leiden ARI;
  - Leiden NMI.
- Compute gene-gene correlation recovery if a defensible reference is available.
- Compute DE concordance only where a defensible bulk or pseudo-bulk reference exists; otherwise state why it is omitted.
- Add a concise real-data results section that does not overclaim beyond the available labels/reference.

Deliverable: real-data table and short analysis section in `paper/main.tex`. Completed with `results_real_data/real_data_clustering_summary.tsv` and `paper/generated/real_data_table.tex`.

## Phase 6: Ablations, Sensitivity, And Downstream Synthetic Tasks

Goal: demonstrate which MaskImpute components matter and quantify robustness.

- Run the required ablations:
  - no shrinkage: `alpha = 0`;
  - no biozero regularization: `lambda_bio = 0`;
  - uniform masking: `p_zero = p_nz`, e.g. `0.03`;
  - plain masked autoencoder: remove shrinkage, biozero regularization, and differential masking.
- Report each ablation with:
  - efficacy score if used in the paper;
  - average MSE;
  - average Biozero-MSE;
  - standard deviation or confidence interval.
- Run hyperparameter sensitivity on a representative scenario:
  - shrinkage alpha in `{0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}`;
  - `lambda_bio` in `{0, 0.5, 1.0, 2.0, 5.0}`;
  - `p_zero` in `{0.005, 0.01, 0.02, 0.05}`.
- Add downstream synthetic tasks if time permits:
  - Leiden ARI/NMI on synthetic labels;
  - trajectory/pseudotime metric such as DPT Kendall tau only if the synthetic data includes a defensible ground truth trajectory.

Deliverable: ablation table, sensitivity figure/table, and concise discussion of component importance. Completed with `paper/generated/phase6_ablation_table.tex`, `paper/generated/phase6_sensitivity_table.tex`, `paper/generated/phase6_calibration_table.tex`, `paper/generated/phase6_downstream_table.tex`, and `docs/feedback_phase6_results.md`.

## Phase 7: Final Integration And Verification

Goal: make the paper internally consistent and submission-ready.

- Regenerate all tables and figures from scripts.
- Compile `paper/main.tex` and check page budget.
- Verify every numerical claim against generated artifacts.
- Verify every citation used in method and benchmark descriptions.
- Confirm all source files referenced by the paper exist in the repo.
- Confirm no obsolete terms remain:
  - MaskClass as method name;
  - Balanced-MSE as public method name;
  - MaskedImpute typo;
  - clustering-only sections unless supported by new downstream experiments;
  - UAI-specific text.
- Record final commands used for regeneration in a reproducibility note.

Deliverable: final compiled manuscript PDF plus reproducibility commands. Completed with `paper/main.pdf`, `scripts/verify_phase7_paper_claims.py`, and `docs/feedback_phase7_verification.md`.

## Critical Path

The order that avoids wasted work is:

1. Fix biozero posterior math in paper and code.
2. Rerun MaskImpute synthetic test results.
3. Rerun full synthetic benchmark with all retained baselines.
4. Add scVI and ALRA.
5. Generate per-scenario, standard-deviation, and significance tables.
6. Add ablations and sensitivity.
7. Add real-data experiments.
8. Finalize manuscript text, figures, and checklist.

## Initial Execution Order

Start with these concrete tasks:

1. Update `masked_imputation26.py` and the biozero equations in `paper/main.tex`.
2. Add a posterior-prior sanity check for the corrected equation.
3. Rerun MaskImpute only on the synthetic test split to quantify the effect of the math correction.
4. Update the paper cleanup items that do not depend on computation.
5. Launch full benchmark regeneration once the corrected MaskImpute result is accepted.
