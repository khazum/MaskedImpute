# MaskImpute v28 Negative-Binomial Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, truth-free v28 development candidate with a negative-binomial count decoder and the existing calibrated selective-output contract.

**Architecture:** Add isolated NB statistical primitives and inject their objective into the shared deterministic masked trainer. Route exact v28 benchmark configurations through the existing audited score, calibration, gate, and output pipeline without mutating the frozen v27 search or public API.

**Tech Stack:** Python 3.11, NumPy, SciPy, PyTorch, pytest, Ruff.

## Global Constraints

- The NB mean is `observed_library_size * decoded_gene_fraction`.
- The likelihood uses `variance = mean + mean^2 / inverse_dispersion`.
- Likelihood arithmetic is float64 and rejects counts, means, or cell libraries above 10,000,000 and inverse dispersion outside `[0.01, 10000]`.
- Dispersion uses exposure-adjusted winsorized moments and deterministic log-scale shrinkage to a robust global median.
- v28 is a benchmark-development candidate only; the public v27 API and tracked v27 search remain unchanged.
- Score artifacts, development calibration, gate, and output semantics remain shared with v27.
- No evaluator truth, labels, annotations, or reconstruction-derived score enters dispatch.
- Conditional execution requires a byte- and semantics-revalidated fixed selection report with `trigger=v28`.

---

### Task 1: NB statistical primitives

**Files:**
- Create: `maskimpute/nb_model.py`
- Create: `tests/test_maskimpute_v28.py`

**Interfaces:**
- Produces: `NegativeBinomialDecoderConfig`, `GeneDispersionEstimate`, `estimate_shrunk_gene_dispersion`, `negative_binomial_nll`, and `NegativeBinomialMaskAutoencoder`.

- [ ] Write failing SciPy-equivalence, dispersion, decoder-simplex, and library-offset tests.
- [ ] Run `python -m pytest tests/test_maskimpute_v28.py -q` and confirm imports fail because `maskimpute.nb_model` does not exist.
- [ ] Implement strict immutable configuration, robust shrunk dispersion, exact differentiable NB NLL, and the explicit-mask simplex decoder.
- [ ] Re-run the focused tests and confirm all Task 1 cases pass.

### Task 2: Deterministic NB training objective

**Files:**
- Modify: `maskimpute/train.py`
- Modify: `maskimpute/nb_model.py`
- Modify: `tests/test_maskimpute_v28.py`

**Interfaces:**
- Consumes: `NegativeBinomialDecoderConfig` and the Task 1 primitives.
- Produces: optional `objective_factory` support in `_train_with_policies` and `train_v28` returning the existing `TrainingOutcome` plus dispersion audit data.

- [ ] Write a failing deterministic small-fit test that checks finite histories, identical state/output across fits, and caller RNG restoration.
- [ ] Run the single test and confirm the missing `train_v28` failure.
- [ ] Add an objective-factory seam whose `None` default preserves v27, and implement v28 training with validation-excluded dispersion estimation.
- [ ] Run v28 and v27 training tests and confirm both pass.

### Task 3: Shared calibrated selective development execution

**Files:**
- Modify: `maskimpute/ablations.py`
- Modify: `maskimpute_benchmark/methods/maskimpute.py`
- Modify: `tests/test_maskimpute_v28.py`

**Interfaces:**
- Consumes: `train_v28` and the existing exact score/calibration artifacts.
- Produces: `run_v28_development_candidate`, using the existing ablation execution primitive for score resolution, LODO calibration, gating, copying, and diagnostics.

- [ ] Write a failing end-to-end v28 candidate test with exact score/calibration artifacts.
- [ ] Confirm failure because the v28 development route is missing.
- [ ] Add the fixed decoder branch and v28 method-adapter entry point without adding a public package API.
- [ ] Confirm calibrated score equivalence, observed-positive copying, nonnegative finite output, and decoder diagnostics.

### Task 4: Exact candidate configuration dispatch

**Files:**
- Modify: `maskimpute_benchmark/runner.py`
- Modify: `tests/test_maskimpute_v28.py`

**Interfaces:**
- Consumes: generic `AuthorizedConfiguration` payloads.
- Produces: strict `maskimpute_decoder_for_configuration` validation and dispatcher construction of `NegativeBinomialDecoderConfig`.

- [ ] Write failing tests for accepted v27/v28 pairs and rejected version, decoder, kind, score-policy, and decoder-hyperparameter drift.
- [ ] Confirm the missing dispatch validator failure.
- [ ] Implement strict routing and pass the exact decoder configuration into in-tree execution.
- [ ] Run focused runner, method-adapter, ablation, v27, and v28 tests.

### Task 5: Verification and handoff

**Files:**
- Modify only files required by verified review findings.

**Interfaces:**
- Produces: one committed isolated branch suitable for cherry-pick or merge after the active competition run.

- [ ] Run `/tmp/maskimpute-supported/bin/python -m pytest tests/test_maskimpute_v28.py tests/test_maskimpute_v27.py tests/test_maskimpute_ablations.py tests/test_maskimpute_method_adapter.py tests/test_benchmark_runner.py -q`.
- [ ] Run `/tmp/maskimpute-supported/bin/python -m ruff check maskimpute/nb_model.py maskimpute/train.py maskimpute/ablations.py maskimpute_benchmark/methods/maskimpute.py maskimpute_benchmark/runner.py tests/test_maskimpute_v28.py`.
- [ ] Inspect `git diff --check`, `git status --short`, and the complete diff.
- [ ] Request independent code review, address all critical and important findings, rerun verification, and commit explicit paths.
