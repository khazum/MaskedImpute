# Leakage-Safe Method Competition and Revision Plan

> **Execution rule:** all architecture selection occurs in `dev/`. The sealed final suite is run once only after the method, calibration, competitors, and environments are frozen. Missing a final gate starts a numbered new round with new data; it never licenses tuning on the evaluated round.

**Goal:** Replace the legacy reconstruction script with a calibrated selective imputer and establish whether v27, an NB v28, or a structure-preserving v29 is competitive under matched inputs and budgets.

**Scientific estimand:** `p_pre_zero` is the probability that a discrete pre-capture count was zero conditional on an observed zero under the fitted count model. It is not structural non-expression. The primary imputed matrix changes observed zeros only; a separate denoised reconstruction may change all entries.

## Development gates

A candidate may advance only if development results show:

- median rank ≤2 for overall MSE, induced-dropout MSE, and gNRMSE;
- Pareto non-domination across induced-dropout MSE, pre-dropout-zero MSE, CorrErr, and non-dropout-nonzero MSE;
- ≥5% induced-dropout-MSE improvement over the strongest learned comparator in at least three of four mechanisms;
- ≤10% degradation in pre-dropout-zero MSE and CorrErr versus the best non-identity learned comparator;
- null-DE FPR ≤0.06 and ≤0.01 above observed counts;
- no orthogonal endpoint with a hierarchical interval showing material degradation versus observed counts.

The candidate-selection table includes every attempted configuration. A failure triggers the next prespecified revision; no configuration is selected by a single combined score.

---

### Task 1: Canonical method package and terminology migration

**Files:**

- Create: `maskimpute/__init__.py`
- Create: `maskimpute/config.py`
- Create: `maskimpute/result.py`
- Create: `maskimpute/prezero.py`
- Test: `tests/test_maskimpute_api.py`
- Test: `tests/test_obsolete_terms.py`

**Requirements:**

- Define immutable `MaskImputeConfig` and `ImputationResult(selective_counts, denoised_counts, p_pre_zero, latent, diagnostics)`.
- Extract the corrected NB/dropout posterior into `p_pre_zero_from_counts`; inputs/outputs are count-scale and finite.
- Active package, runner, generated tables, and manuscript may not contain `biozero`, `p_bio`, or structural-zero claims. Historical archives are explicitly excluded.
- Keep a migration wrapper for the legacy CLI but route publication execution only through the package API.

**Commit:** `refactor: define calibrated MaskImpute API`

---

### Task 2: v27 explicit-mask selective autoencoder

**Files:**

- Create: `maskimpute/model.py`
- Create: `maskimpute/train.py`
- Create: `maskimpute/impute.py`
- Test: `tests/test_maskimpute_v27.py`

**Requirements:**

- Normalize counts using observed library size only; record the invertible output-scale contract.
- Encoder input concatenates expression with an availability/corruption indicator. A learned per-gene mask token represents unavailable entries, so an artificially masked positive is never encoded as an ordinary numeric zero.
- Self-supervision masks observed positives in count/expression strata. Targets are known masked positives; evaluator truth, labels, markers, and pseudotime are structurally unavailable.
- Natural observed zeros contribute only a soft `p_pre_zero`-weighted preservation regularizer. They are never treated as known positive or known biological-zero targets.
- Inference marks observed zeros unavailable, decodes candidates, and applies a monotone gate such as `(1-p_pre_zero)^gamma` or a prespecified sigmoid family.
- `selective_counts[observed>0]` must equal input counts bit-for-bit. `denoised_counts` is returned separately and clearly labeled.
- Early stopping uses a held-out artificial-positive mask fixed before training, not simulator truth.

**Commit:** `feat: implement selective MaskImpute v27`

---

### Task 3: Calibration without circular reconstruction

**Files:**

- Create: `maskimpute/calibration.py`
- Create: `scripts/fit_prezero_calibration.py`
- Test: `tests/test_prezero_calibration.py`

**Requirements:**

- The uncalibrated count posterior is always available and is the default score.
- Candidate calibrators are identity, logistic, beta calibration, and isotonic. Fit only on pooled development draws with leave-one-biological-draw-out assessment and mechanism-balanced weights.
- A calibrator is retained only if Brier score improves on at least three mechanisms without worse log loss or calibration slope outside a prespecified tolerance.
- The calibrator consumes count-derived features only. Reconstruction output cannot feed the score.
- Serialize coefficients/knots and training-manifest hashes in a tracked calibration artifact before freeze.

**Commit:** `feat: calibrate pre-capture-zero scores`

---

### Task 4: Capacity-matched ablation panel

**Files:**

- Create: `maskimpute/ablations.py`
- Create: `study/ablations.json`
- Test: `tests/test_maskimpute_ablations.py`

**Variants:**

- capacity-matched masked AE: uniform positive masking, no score regularizer, ungated output;
- no gate;
- no pre-zero regularizer;
- no explicit mask channel/token;
- full denoising instead of selective output;
- direct versus calibrated `p_pre_zero`.

All variants share parameter count, optimizer budget, seeds, and preprocessing unless the named component logically changes them.

**Commit:** `feat: define capacity-matched method ablations`

---

### Task 5: Unified method registry and run contract

**Files:**

- Create: `maskimpute_benchmark/methods/base.py`
- Create: `maskimpute_benchmark/methods/registry.py`
- Create: `study/methods.json`
- Test: `tests/test_method_registry.py`

**Requirements:**

- Each adapter declares same-input versus external-reference track, input/output scale, stochasticity, version/source pin, environment, timeout, CPU/GPU needs, and whether observed positives are preserved.
- All adapters receive only `make_inference_view`; output shape/IDs/source hash are validated.
- Every run emits status, runtime, peak RSS, peak GPU memory, stdout/stderr hashes, output hash, method seed, and explicit failure reason.
- Method statuses are retained for all dataset views; unavailable methods are never dropped from summary denominators.

**Commit:** `feat: define matched method execution contract`

---

### Task 6: Same-input baseline adapters

**Files:**

- Create: `maskimpute_benchmark/methods/observed.py`
- Create: `maskimpute_benchmark/methods/alra.py`
- Create: `maskimpute_benchmark/methods/magic.py`
- Create: `maskimpute_benchmark/methods/dca.py`
- Create: `maskimpute_benchmark/methods/scvi.py`
- Create: `maskimpute_benchmark/methods/saver.py`
- Test: `tests/test_core_method_adapters.py`

**Requirements:**

- Observed counts and the capacity-matched AE are mandatory controls.
- DCA, scVI, ALRA, MAGIC, and SAVER use pinned upstream releases/defaults as the starting point; conversions back to the evaluator count/log scale are tested on fixtures.
- Published output conventions are retained and disclosed; no post-hoc selective copying is added to a comparator unless reported as a separate transformation/control.
- ccImpute may remain as a historical comparator but cannot substitute for the newer panel.

**Commit:** `feat: add matched core imputation adapters`

---

### Task 7: Recent competitor and matched-bulk adapters

**Files:**

- Create: `maskimpute_benchmark/methods/scsdae.py`
- Create: `maskimpute_benchmark/methods/sccr.py`
- Create: `maskimpute_benchmark/methods/d3impute.py`
- Create: `scripts/fetch_method_sources.py`
- Test: `tests/test_recent_method_adapters.py`

**Pins:**

- scSDAE `fa7ded1080695e38e6193ef137dc8d635ae64ec9`;
- scCR `f8ccf889bbdd7d22047716eb1d6ef793ce00260b`;
- D3Impute `f8f247a54a7ff1fcfca3232b9c0016b6929b5825`.

**Requirements:**

- Integrate upstream behavior faithfully in isolated environments. Compatibility shims live in adapters and are logged; upstream checkouts remain pristine.
- scCR's missing public graph utility is reconstructed only from its published algorithm/source call contract and covered by equivalence fixtures; otherwise status is `upstream_incomplete`.
- D3Impute runs only when a prespecified matched bulk reference exists and is labeled external-reference. PbImpute is attempted only if D3Impute is unusable after a documented integration attempt.
- scSDAE legacy TensorFlow compatibility is containerized; failure to build/run is reported, not silently replaced by a reimplementation.

**Commit:** `feat: add recent imputation competitors`

---

### Task 8: Budget-matched development runner

**Files:**

- Create: `maskimpute_benchmark/runner.py`
- Create: `scripts/run_development_competition.py`
- Create: `study/development_search.json`
- Test: `tests/test_benchmark_runner.py`

**Requirements:**

- Same observed matrix, genes, cells, normalization target, and development access for all same-input methods.
- Maximum 20 configurations or 8 GPU-hours per method; CPU-only methods receive 24 wall-clock hours. Failed configurations still consume budget unless failure is infrastructure-only.
- Run deterministic methods once and stochastic methods at seeds 42/43/44; seed rows remain nested.
- Compute the complete long-form metric/status schema and completeness checks after every run.
- No method code may inspect evaluator layers; a process-level integration test attempts and fails to access them.

**Commit:** `feat: run budget-matched development competition`

---

### Task 9: Pareto selection and revision trigger

**Files:**

- Create: `maskimpute_benchmark/selection.py`
- Create: `scripts/select_development_candidate.py`
- Test: `tests/test_candidate_selection.py`

**Requirements:**

- Aggregate model seeds within biological draws and paired technical views within draws.
- Compute ranks/gates per mechanism and overall; use no hidden weights or combined efficacy score.
- Emit eligible/ineligible reasons for every configuration and a deterministic Pareto set.
- If no v27 candidate passes, emit `trigger=v28`; if efficacy passes but CorrErr/downstream safety fails, emit `trigger=v29` after v28 assessment.

**Commit:** `feat: prespecify competitive candidate selection`

---

### Task 10: v28 NB decoder, conditional

**Files:**

- Create: `maskimpute/nb_model.py`
- Test: `tests/test_maskimpute_v28.py`

**Requirements:**

- Decoder mean is positive with observed-library-size offset; dispersion mode is global, gene-wise, or shrinkage gene-wise as declared.
- NB log likelihood is numerically checked against SciPy/manual values. Masked-positive prediction remains the primary self-supervised signal.
- Gate/calibration/output semantics remain unchanged from v27 so the decoder is the isolated revision.
- Retain v28 only if development Pareto rank improves without violating zero-preservation and DE-null safeguards.

**Commit:** `feat: add conditional NB MaskImpute v28`

---

### Task 11: v29 structure preservation, conditional

**Files:**

- Create: `maskimpute/structure.py`
- Test: `tests/test_maskimpute_v29.py`

**Requirements:**

- Add a minibatch covariance/correlation penalty on a frozen label-free variable-gene panel and an observed-input neighborhood consistency term.
- Neither term sees truth or evaluation labels. Neighbor construction is fixed per dataset before tuning.
- Capacity-matched ablation has both terms off.
- Retain only if CorrErr or downstream safety improves without a material induced-dropout-MSE loss.

**Commit:** `feat: add conditional structure-preserving v29`

---

### Task 12: Freeze selected method and competitor panel

**Files:**

- Create: `study/frozen_method.json`
- Create: `scripts/freeze_publication_round.py`
- Test: `tests/test_freeze_publication_round.py`

**Requirements:**

- Record selected version/config/calibrator/ablation, development result hashes, competitor pins, common correlation-gene panel rule, environments, and selection-gate table.
- Refuse freeze if any required competitor lacks either a successful adapter smoke test or a signed reason code, if the repository contains uninitialized gitlinks, or if generated development assets are incomplete.
- Freeze before final seeds are materialized. The final runner consumes exactly the frozen registry and cannot accept CLI hyperparameter overrides.

**Commit:** `feat: freeze publication method competition`

## Completion gate

The method phase is complete only when a reviewed development candidate passes every efficacy/safety gate or the manuscript claim is explicitly downgraded. Competitive superiority is never inferred from a tuned legacy test set or from a favorable subset of methods/metrics.
