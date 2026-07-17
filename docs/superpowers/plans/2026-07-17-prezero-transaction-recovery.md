# Pre-zero Transaction Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make realized conversion-terminal score evidence mandatory from trusted execution provenance and make interrupted development artifact publication safely retryable.

**Architecture:** `CheckpointStore` derives expected score presence independently of stored score evidence and passes the result to the shared semantic validator. Development append mirrors the final runner's transaction-intent ordering, with recovery bound to a canonical, stable checkpoint prefix and restricted to unreferenced owned artifacts.

**Tech Stack:** Python 3.11, NumPy, pathlib, canonical JSON/SHA-256 receipts, pytest, Ruff.

## Global Constraints

- Do not weaken exact probability, policy, storage, semantic checksum, or report validation.
- Preserve legitimate MaskImpute failures that never produced an `AdapterExecution` and therefore have no score matrix.
- Preserve existing final transaction semantics.
- Recovery may unlink only unreferenced, regular, nonsymlink, current-user-owned, single-link artifacts from a canonical intent.
- Every production change follows a witnessed RED/GREEN test cycle.

---

### Task 1: Bind required score presence to execution provenance

**Files:**
- Modify: `maskimpute_benchmark/runner.py`
- Modify: `maskimpute_benchmark/prezero_evidence.py`
- Test: `tests/test_prezero_evidence.py`
- Test: `tests/test_final_runner.py`

**Interfaces:**
- Consumes: `RunPlanEntry`, persisted run receipt, and the canonical `evaluator_conversion_*_detail_<sha256>` reason domain.
- Produces: `expected_matrix_present: bool` passed to `validate_stored_prezero_evidence` and used to select exact execution-authority derivation.

- [ ] **Step 1: Write development coordinated-removal regression**

Create a real conversion-terminal attempt, persist it, replace matrix/policy/storage with their absent schema, regenerate unavailable reports with `_score_report(None, ...)`, recompute `evidence_sha256` and `checkpoint_sha256`, remove the score file, and assert `CheckpointStore.load` rejects the missing required score.

- [ ] **Step 2: Run development regression and verify RED**

Run:

```bash
PYTHONPATH=. /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest -q tests/test_prezero_evidence.py::test_development_conversion_terminal_score_cannot_be_coordinately_removed -W error
```

Expected: FAIL because the coherently absent evidence currently loads.

- [ ] **Step 3: Write final coordinated-removal regression**

Persist a real final conversion-terminal record, apply the same absent evidence replacement and checksum rebinding, remove the score file, and assert a fresh `FinalResultStore.load_records` rejects it.

- [ ] **Step 4: Run final regression and verify RED**

Run:

```bash
PYTHONPATH=. /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest -q tests/test_final_runner.py::test_final_conversion_terminal_score_cannot_be_coordinately_removed -W error
```

Expected: FAIL because the coherently absent evidence currently loads.

- [ ] **Step 5: Implement the provenance-derived requirement**

Add a strict conversion-reason predicate and derive mandatory presence for a count-score MaskImpute row when it completed, retains native-output provenance, or records a canonical unavailable conversion disposition. Pass that Boolean to `_expected_prezero_authority` and `validate_stored_prezero_evidence`. Reject any observed/expected presence mismatch before matrix validation.

- [ ] **Step 6: Run both regressions and existing conversion tests**

Run both new tests plus the development/final matrix-policy-report parameterizations with `-W error`. Expected: all pass.

### Task 2: Recover interrupted development publication

**Files:**
- Modify: `maskimpute_benchmark/runner.py`
- Test: `tests/test_prezero_evidence.py`

**Interfaces:**
- Produces: private `CheckpointStore` intent publication, prefix inspection, interrupted-transaction recovery, and intent completion helpers.
- Consumes: staged-valid `EvaluatedAttempt`, `CompetitionPlan`, current canonical checkpoint, and flat immutable artifact naming from `_stored_attempt`.

- [ ] **Step 1: Write artifact-boundary restart regression**

Parameterize interruption after `stdout`, `stderr`, `native-f64`, `log2-cp10k-f64`, and `p-pre-zero-f64.zlib`. Patch the real store's immutable publisher to raise only after closing the selected final artifact, assert an intent remains, construct a new store, retry identical matrices/evidence with changed stdout/stderr, and assert success with no transaction directory or unreferenced old log bytes.

- [ ] **Step 2: Run regression and verify RED**

Run:

```bash
PYTHONPATH=. /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest -q tests/test_prezero_evidence.py::test_development_restart_recovers_every_interrupted_artifact_boundary -W error
```

Expected: FAIL because no development intent exists and changed log retry collides with immutable stdout.

- [ ] **Step 3: Implement canonical transaction intents**

Before final publication, write an intent bound to schema version, plan SHA-256, ordinal, checkpoint position, run ID, and sorted exact artifact paths. Keep invalid-attempt staging ahead of intent creation.

- [ ] **Step 4: Implement prefix-safe recovery**

Validate intent name/schema/checksum/canonical bytes and exact plan-derived path set. Parse and checksum the checkpoint without following links. Retain artifacts for a matching committed record. Otherwise collect committed artifact references, require candidates to be disjoint, re-read identical checkpoint bytes, and unlink only owned unique regular files. Durably remove each intent and empty transaction/run directories.

- [ ] **Step 5: Wire append ordering and verify GREEN**

Recover before staging/publication, publish intent, publish artifacts, atomically write the checkpoint, then durably remove the intent. Re-run the boundary regression and existing invalid-attempt/final recovery tests with `-W error`.

### Task 3: Focused verification and commit

**Files:**
- Verify all files changed by Tasks 1-2.

**Interfaces:**
- Consumes: completed implementation and regression suite.
- Produces: one clean implementation commit and exact Git range.

- [ ] **Step 1: Run focused warning-strict suites**

Run `tests/test_prezero_evidence.py`, `tests/test_final_runner.py`, and the directly affected benchmark checkpoint tests under the compatible Python environment with `-W error`.

- [ ] **Step 2: Run static checks**

Run Ruff check on changed Python files, `py_compile`, and `git diff --check`. Expected: zero errors.

- [ ] **Step 3: Review the final diff**

Confirm no final transaction changes, no weakened authority comparisons, no unsafe unlink path, and no unrelated user changes.

- [ ] **Step 4: Commit**

Stage only the design, plan, implementation, and tests, then commit with message `Recover interrupted score evidence publication`.
