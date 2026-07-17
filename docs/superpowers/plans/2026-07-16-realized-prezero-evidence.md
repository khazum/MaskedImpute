# Realized p_pre_zero Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve and evaluate MaskImpute's realized `p_pre_zero` matrix as audit-ready development and final publication evidence.

**Architecture:** A focused `prezero_evidence` module owns immutable evidence snapshots, policy binding, metric serialization, deterministic zlib encoding, and bounded validation. The runner extracts evidence at the adapter boundary; existing development and final stores persist and revalidate the same schema.

**Tech Stack:** Python 3.11+, dataclasses, NumPy, zlib, canonical JSON/SHA-256, pytest.

## Global Constraints

- Never reconstruct `p_pre_zero` from receipts.
- Never execute or inspect the untouched final panel while implementing this feature.
- Preserve explicit result rows for non-MaskImpute and noncompleted attempts.
- Use deterministic little-endian float64 and zlib level 6.
- Validate before publication use with bounded decompression and exact identity/policy bindings.

---

### Task 1: Adapter-to-evaluator evidence

**Files:**
- Create: `maskimpute_benchmark/prezero_evidence.py`
- Modify: `maskimpute_benchmark/methods/observed.py`
- Modify: `maskimpute_benchmark/methods/maskimpute.py`
- Modify: `maskimpute_benchmark/runner.py`
- Test: `tests/test_prezero_evidence.py`

**Interfaces:**
- Consumes: `AdapterExecution`, `RunPlanEntry`, `PreparedDataset`, and evaluator-private truth.
- Produces: immutable `PreZeroEvidence`, `evaluate_prezero_evidence(...)`, and `PreZeroEvidence.to_record()`.

- [ ] Write tests that require an exact realized matrix/policy for completed MaskImpute and explicit rows for every other status/method.
- [ ] Run the focused tests and verify they fail because the evidence interface is absent.
- [ ] Implement immutable matrix and policy snapshots plus semantic SHA-256 identity binding.
- [ ] Run the focused tests and verify adapter-to-evaluator propagation passes.
- [ ] Commit the adapter/evaluator slice.

### Task 2: Overall and stratified score reports

**Files:**
- Modify: `maskimpute_benchmark/prezero_evidence.py`
- Test: `tests/test_prezero_evidence.py`

**Interfaces:**
- Consumes: existing `zero_score_metrics` and `stratified_zero_score_metrics` outputs.
- Produces: canonical `auroc`, `auprc`, `brier`, `log_loss`, calibration intercept/slope, `ece`, and reliability-bin records.

- [ ] Write tests for exact pre-capture estimates and continuous/proxy reason codes across all four mechanisms.
- [ ] Run them and verify missing report serialization is the failure.
- [ ] Implement canonical metric conversion and explicit unavailable denominators.
- [ ] Run focused metric and evidence tests until green.
- [ ] Commit the metric schema slice.

### Task 3: Development persistence and resume validation

**Files:**
- Modify: `maskimpute_benchmark/prezero_evidence.py`
- Modify: `maskimpute_benchmark/runner.py`
- Modify: `maskimpute_benchmark/development_evaluation.py`
- Test: `tests/test_prezero_evidence.py`
- Test: `tests/test_benchmark_runner.py`

**Interfaces:**
- Consumes: `PreZeroEvidence.to_record()` and immutable artifact publication.
- Produces: zlib artifact receipts and bounded `validate_stored_prezero_evidence(...)`.

- [ ] Write fresh, resume, compressed/uncompressed tamper, partial-receipt, and zip-bomb tests.
- [ ] Run them and verify raw/unvalidated storage fails the new expectations.
- [ ] Implement deterministic compression and bounded validation in checkpoint store/load.
- [ ] Make downstream evaluation read only validated score evidence when needed.
- [ ] Run runner/development regressions and commit.

### Task 4: Final transaction and capacity accounting

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Test: `tests/test_final_runner.py`
- Test: `tests/test_prezero_evidence.py`

**Interfaces:**
- Consumes: validated stored evidence and final plan entries.
- Produces: score-aware transaction intents, resumable records, and score-inclusive disk preflight.

- [ ] Write final fresh/resume/tamper/zip-bomb and preflight-count tests.
- [ ] Run them and verify final storage omits the score artifact/accounting.
- [ ] Add the score artifact to transactions and exact preflight `compressBound` arithmetic.
- [ ] Validate final evidence against the existing execution-request calibration authority.
- [ ] Run final-runner regressions and commit.

### Task 5: Publication-integrity verification

**Files:**
- Modify only files required by failures introduced by this feature.

**Interfaces:**
- Consumes: the complete implementation branch.
- Produces: a reviewable commit with exact test evidence.

- [ ] Run focused evidence, runner, final-runner, metrics, and development-evaluation tests.
- [ ] Run Ruff, Python compilation, `git diff --check`, and inspect `git status`/diff statistics.
- [ ] Confirm no development/final result artifacts were generated and no claims were added.
- [ ] Commit the complete branch and report the exact commit and verification commands.
