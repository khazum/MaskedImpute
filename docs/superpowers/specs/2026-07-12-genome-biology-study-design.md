# MaskImpute Genome Biology Study Redesign

**Status:** Binding execution specification for the publication rebuild

**Target article type:** Genome Biology Methodology

## Objective

Rebuild MaskImpute as a calibrated, selective scRNA-seq imputation method and evaluate it with an auditable protocol that supports publication-quality claims. Development may iterate on architecture and hyperparameters, but final evidence must come from a fresh suite that cannot be run with unfrozen code.

Competitive results are a target, not a guaranteed conclusion. If the frozen method misses the prespecified final gates, the result must be reported honestly or a new development round must be opened; an evaluated final suite can never be relabeled as untouched.

## Non-negotiable study rules

1. The legacy Splatter tune/test data and every existing result are development data because the historical test split influenced method selection.
2. Each independent simulated dataset draw, not a training seed or matrix entry, is an experimental unit.
3. Model seeds are nested within dataset draws. They quantify optimization variability and never inflate the independent sample size.
4. All methods receive the same observed matrix, genes, cells, normalization target, development-only tuning access, and wall-clock search budget.
5. The final runner requires a recursively clean Git commit (including initialized submodules), raw tracked-file bytes matching the index without clean filters or replacement objects, and a raw filesystem walk showing no ignored, ordinary untracked, empty-directory, or special-file state except exact state records and hash-declared result files. Config, protocol, environment, and the unseen final manifest are hash-bound. Rounds are direct children of one canonical root, while their authoritative registry, repository-instance identity, Git-common/state-root/registry filesystem identities, and inode-checked locks live in non-symlinked private directories under Git's common directory so copied worktree artifacts and ordinary same-path replacement clones cannot replay a holdout. Registry entries form a lifecycle-validated digest chain rooted at `freeze.json`. Transition records, authority directories, declared result bytes, and their containing directories are synchronously persisted. Receipt crash recovery revalidates the receipt's own declared files, and a final check after registry publication terminally supersedes a round if inputs or results changed. Verification atomically claims the sole execution before a one-use evaluation receipt can be written.
6. After final evaluation, method or hyperparameter changes require a new numbered round and a newly generated holdout. Earlier final results remain archived as prior evidence.
7. No result is omitted because it is unfavorable. Failed runs, timeouts, memory peaks, and unavailable method-dataset combinations are reported.

## Review items 1–10 and binding remedies

| Item | Remedy | Acceptance artifact |
|---|---|---|
| 1 | Add a state-controlled study registry with development, frozen, evaluated, and superseded rounds; bind final execution to code/config/protocol hashes. | `study/rounds/<round>/freeze.json`, final manifest, and evaluation receipt |
| 2 | Generate independent biological/measurement draws for every condition; use 3 model seeds within each final draw for stochastic neural methods. | Dataset manifest with generator seed, measurement seed, model seed, and checksums |
| 3 | Use four mechanisms beyond Splatter: SymSim, SERGIO, SPARSim, and empirical high-depth downsampling/count splitting. Splatter remains development-only. | Four adapter manifests and generator smoke tests |
| 4 | Replace `biozero`/`p_bio` with `pre_dropout_zero`/`p_pre_zero`. Describe this as a model-dependent probability that the pre-capture count was zero, never structural non-expression. | Obsolete-term test over code, tables, and manuscript |
| 5 | Validate `p_pre_zero` among observed zeros with AUROC, AUPRC, Brier score, log loss, calibration intercept/slope, ECE, and reliability curves, overall and by expression/library-size strata. | Score-metric CSV and calibration figure |
| 6 | Report dropout-enabled conditions separately; use medians/IQRs, paired effects and confidence intervals, win counts, and zero-preservation/dropout-recovery Pareto fronts. | Generated aggregate and per-condition tables |
| 7 | Add high-depth thinning, technical/replicate concordance, ERCC/spike-in recovery where present, CITE-seq RNA–protein concordance, pseudobulk/bulk concordance, null and positive-control DE, and trajectory recovery. | Molecular-validation result bundle with dataset provenance |
| 8 | Add scSDAE, scCR, D3Impute on the matched-bulk track (or PbImpute if D3 cannot run), and a capacity-matched masked-AE ablation; retain observed, DCA, scVI, ALRA, MAGIC, SAVER, and relevant current baselines. | Adapter tests, version ledger, and complete method-status table |
| 9 | Aggregate with a generator/condition/dataset hierarchical bootstrap; average model seeds within draws for the primary analysis and report seed variance separately. Correct families of pairwise tests with Holm's method. | Statistical-analysis code, deterministic fixtures, and inference tables |
| 10 | Report overall/dropout/pre-dropout-zero/nonzero errors; distribution, mean/variance, gene–gene and cell–cell correlation distortion; downstream metrics; runtime, peak RSS/GPU memory; failures and timeouts. | Long-form metric schema with completeness validation |

## Validation panel

All adapters emit a common `h5ad` schema:

- `X`: observed nonnegative integer counts, cells by genes;
- `layers["pre_capture_counts"]`: exact discrete pre-capture counts for SymSim;
- `layers["latent_expression"]` or `layers["pre_dropout_expression"]`: exact continuous simulator truth for SPARSim or SERGIO;
- `layers["reference_counts"]`: high-depth proxy counts for semisynthetic data;
- `layers["heldout_counts"]`: an independent count-split replicate when available;
- `layers["expected_counts"]`: simulator expectation when available;
- `obs`: constant per-file `dataset_id`, `mechanism`, `condition`, `biological_id`, `technical_view`, and positive-integer `draw`; exact observed `library_size`; evaluator-only `group`; optional `batch` and `pseudotime`;
- `var`: stable feature identifiers and optional marker/ERCC flags;
- `uns["truth_kind"]`: one of `exact_pre_capture`, `exact_continuous`, `proxy_high_depth`, or `orthogonal_only`;
- `uns["primary_truth_layer"]`: the one evaluator layer used for reconstruction endpoints, absent for `orthogonal_only`;
- `uns["provenance"]`: source accession/URL, source checksum, package version/commit, full parameters, and seeds.

The four mechanisms are deliberately distinct:

1. **SymSim:** mechanistic two-state promoter kinetics with explicit molecule, capture, amplification, sequencing, and UMI stages. Retain exact molecule counts and vary capture efficiency.
2. **SERGIO:** GRN-driven stochastic differential equations followed by independently controlled outlier, library-size, dropout, and UMI effects. Retain clean expression and pre-dropout technical expression separately.
3. **SPARSim:** Gamma biological expression followed by multivariate-hypergeometric sequencing. Retain continuous biological expression and measured counts; pre-dropout-zero metrics are undefined for this positive truth.
4. **Semisynthetic:** Gamma-Poisson thinning and independent binomial count splitting of high-depth UMI data. Treat the original as a proxy, not error-free biological truth.

Development uses two independent biological draws per simulated mechanism. Final evaluation uses five new biological draws per simulated mechanism with paired moderate/severe measurement views and three model seeds per stochastic method. Measurement views and semisynthetic thinning seeds are repeated technical observations, not additional biological replicates. Final simulated matrices contain approximately 2,700 cells and 1,000–1,200 genes. A separate scaling panel uses 10k, 25k, 50k, and 100k cells with one prespecified mechanism and reports accuracy only at sizes whose truth fits the metric implementation.

## Method-development ladder

### v27: selective output and corrected zero-score use

- Preserve every observed positive entry in the primary imputed matrix.
- Reconstruct artificially masked positive entries during training; use a learned mask token or explicit mask indicator rather than the scaled numeric value zero.
- Impute only observed zeros.
- Replace the circular reconstruction-derived zero score with the original count-model `p_pre_zero` unless a development-only calibration model improves Brier score on at least three of four mechanisms.
- Apply a soft zero gate on count scale: predicted zero value is the decoder estimate multiplied by a monotone function of `1 - p_pre_zero`.
- Return the denoised full reconstruction separately from the selective imputed matrix so users can choose the appropriate estimand.

### v28: count-likelihood revision, triggered if v27 misses a gate

- Replace the scaled-Gaussian output loss with a negative-binomial mean/dispersion decoder using observed library size as an offset.
- Retain the masked-positive prediction loss and score-weighted natural-zero preservation loss.
- Select dispersion mode, gate family, and loss weights on development data only.

### v29: structure-preserving revision, triggered if CorrErr or downstream safety misses a gate

- Add a minibatch covariance penalty on variable genes and a local-neighborhood consistency loss.
- Keep both penalties off in the capacity-matched masked-AE baseline.
- Retain a revision only if it improves the development Pareto rank without violating zero-preservation or DE-null safeguards.

Candidate selection is Pareto-based; the historical `MSE + 2*Biozero-MSE` score is retired. A candidate is eligible to freeze only if, on development draws:

- median rank is at most 2 for overall MSE, dropout MSE, and gNRMSE;
- it is non-dominated across dropout MSE, pre-dropout-zero MSE, CorrErr, and nonzero MSE;
- it beats the strongest learned competitor by at least 5% on dropout MSE in at least three of four mechanisms;
- relative degradation is at most 10% for pre-dropout-zero MSE and CorrErr versus the best non-identity learned competitor;
- null-DE false-positive rate is at most 0.06 at nominal 0.05 and no more than 0.01 above observed counts;
- no orthogonal real-data endpoint has a hierarchical 95% interval showing material degradation versus observed counts.

## Baseline policy

- The core same-input panel is observed counts, matched masked AE, MaskImpute, DCA, scVI, ALRA, MAGIC, SAVER, scSDAE, and scCR.
- D3Impute is evaluated only on datasets with a prespecified matched bulk reference and is labeled an external-reference method. PbImpute substitutes only if D3Impute is technically unusable after a documented integration attempt.
- scMAE and any newer directly applicable method discovered before freeze are included or given an explicit incompatibility rationale.
- Published defaults are the starting point. Development tuning uses a common maximum of 20 configurations or 8 GPU-hours per method, whichever comes first; CPU-only methods receive 24 wall-clock hours. Per-run final timeout is 6 hours and memory limit is 48 GB RAM/14 GB GPU.
- A failed run is retried once only for a documented transient failure. Algorithmic failures remain in the status table.

## Endpoints and inference

Primary exact-truth endpoints are overall MSE, induced-dropout MSE, pre-dropout-zero MSE, gNRMSE, and CorrErr in log2(CP10k+1) space. Pre-dropout-zero reconstruction and score-calibration endpoints require discrete pre-capture truth; they are emitted with the explicit reason `undefined_for_continuous_truth` for SERGIO/SPARSim continuous truth and `proxy_truth_not_exact` for semisynthetic references. Safety endpoints are observed-positive MSE, mean/variance distortion, false positive expression, null-DE FPR, and marker-rank loss. Secondary endpoints include MAE, calibration metrics, cell–cell distance distortion, clustering, pseudotime, RNA–protein concordance, and bulk/pseudobulk concordance.

For each method and biological draw, stochastic-seed metrics are averaged first and paired moderate/severe technical views are then averaged for the across-view primary comparison. A technical view never creates an additional independent unit. The hierarchical bootstrap resamples mechanisms and `biological_id` values within mechanism; model seeds are resampled only inside their biological draw to propagate optimization uncertainty. View-stratified analyses retain the same biological unit. Report median paired percent change, 95% percentile interval, probability of improvement, Holm-adjusted two-sided paired p-values, and the count of independent biological draws won. Report between-draw, between-view, and within-draw seed variance separately.

Final success is not defined as winning every metric. The publication claim may say “competitive” only if the frozen method has median rank at most 2 on the three efficacy endpoints, is Pareto non-dominated, and passes every safety gate. Stronger superiority language requires a multiplicity-corrected 95% interval excluding zero against the strongest applicable competitor on the named endpoint.

## Genome Biology manuscript contract

Use the Springer Nature LaTeX template and this Methodology order: title page; unstructured abstract of at most 100 words; 3–10 keywords; Background; Results; Discussion; Conclusions; Methods; Abbreviations; Declarations; References; Additional-file descriptions.

The main Results must show the multi-mechanism benchmark, direct zero-score validation, orthogonal molecular validation, ablations, and resource scaling. The Discussion must lead with when selective imputation helps, when observed counts should be preferred, simulator limitations, and external-reference tradeoffs. All claims are generated from final machine-readable assets. Restore non-anonymous author metadata only from user-provided values; never invent names, affiliations, funding, ethics determinations, contributions, repository DOI, or competing-interest declarations.

Required reproducibility assets are an OSI license, public source URL placeholder clearly marked for author completion, archived-release/DOI placeholder, pinned Python and R environments, container recipe, dataset accession ledger, exact commands, raw long-form results, machine-readable tables, and an end-to-end smoke workflow. Remove the tracked virtual environment and replace it with lockfiles.

## Completion gates

Implementation is complete only when:

1. every new unit and integration test passes and vendored/submodule tests are excluded from default collection;
2. all four validation mechanisms pass schema and deterministic-seed checks;
3. the selected method is frozen before the final manifest is materialized;
4. the final receipt proves code/config/protocol hashes and one-use execution;
5. every prespecified metric has a value or explicit reason code for every method/dataset run;
6. paper numerical claims exactly match generated assets;
7. the Genome Biology PDF and additional file compile without missing references, assets, or provisional scientific claims;
8. human-only submission fields are listed in a blocking author checklist rather than fabricated.
