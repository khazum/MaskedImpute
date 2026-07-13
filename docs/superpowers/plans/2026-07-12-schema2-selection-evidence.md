# Schema-2 Selection Evidence Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the publication selection consumer accept schema 2 only when its fixed evaluation sidecar and every byte-level reconstruction, orthogonal, source, and audit binding revalidate.

**Architecture:** `selection.py` will read the fixed evaluation manifest from the repository rather than trusting caller-supplied provenance. It will verify the acyclic selection-evidence digest, rebuild the reconstruction authority/plan, load its checkpoint, compare orthogonal and source manifests to their exact files, and cross-check audit identities against the records consumed by selection. The CLI will accept only the fixed input and report paths and pass the repository derived from those paths to the same consumer.

**Tech Stack:** Python 3.12, canonical JSON/SHA-256, existing runner/evaluator validators, pytest, Ruff.

## Global Constraints

- Start from `b60c3bc` in a new worktree and cherry-pick evaluator commit `f67a6c3e91cc718c21a383243977ea0c968cebf6`.
- The fixed sidecar is `artifacts/study/development/evaluation/evaluation_manifest.json`.
- The fixed report is `artifacts/study/development/evaluation/development_selection_report.json`.
- Schema 2 has exactly eight top-level fields, including `evaluation_manifest_sha256`; no caller may provide design authority.
- The evaluation manifest contains no `result_sha256` or self file hash, preventing a digest cycle.
- Do not change the `genome-biology` worktree or benchmark artifacts.

---

### Task 1: Fail-closed evaluation sidecar validator

**Files:**
- Modify: `maskimpute_benchmark/selection.py`
- Test: `tests/test_selection_authority.py`

**Interfaces:**
- Consumes: `_validate_selection_evaluation_manifest(repository, data, authority, status)` with the exact schema-2 payload and validated repository authority.
- Produces: a mapping of verified evaluation/checkpoint/orthogonal/source hashes for `SelectionReport.authority_bindings`.

- [ ] **Step 1: Write failing sidecar tests**

Add a compact canonical fixture and tests that call the real validator, accept an acyclic valid sidecar, then reject independently tampered evaluation file bytes, internal `manifest_sha256`, `selection_evidence_sha256`, checkpoint file/internal/plan/input bindings, aliased orthogonal bytes, source bytes, null-DE audit output hashes, and orthogonal-audit interval mismatches.

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
/tmp/maskimpute-supported/bin/python -m pytest tests/test_selection_authority.py -q -k 'evaluation_manifest or schema2'
```

Expected: fail because the exact field and fixed sidecar validator do not exist.

- [ ] **Step 3: Implement stable fixed-file validation**

Implement stable no-symlink byte reads and canonical JSON checks, then validate this exact acyclic relationship:

```python
evidence_core = {
    "schema_version": data["schema_version"],
    "dataset_manifest_sha256": data["dataset_manifest_sha256"],
    "count_score_manifest_sha256": data["count_score_manifest_sha256"],
    "retained_calibration_artifact_sha256": data[
        "retained_calibration_artifact_sha256"
    ],
    "records": data["records"],
    "orthogonal_intervals": data["orthogonal_intervals"],
}
```

Require the top-level sidecar file SHA, its internal canonical hash, and `selection_evidence_sha256 == _canonical_sha256(evidence_core)`. Require exact count/calibration paths and hashes. Rebuild the current runner authority and competition plan from the validated dataset status, load the fixed completed checkpoint against that plan, and compare its file/internal/input/plan/raw-artifact evidence. Re-read the fixed orthogonal manifest, require exact authority/configuration/seed/record denominator and every completed output hash/size. Re-run `validate_real_source_artifacts(repository)` and compare the source block exactly. Cross-check null-DE audit run/output identities and orthogonal audit interval identities.

- [ ] **Step 4: Run tests and record GREEN**

Run the Task 1 selection tests and require every new tamper test to pass.

---

### Task 2: Schema-2 consumer and fixed CLI integration

**Files:**
- Modify: `maskimpute_benchmark/selection.py`
- Modify: `scripts/select_development_candidate.py`
- Test: `tests/test_selection_authority.py`
- Test: `tests/test_candidate_selection.py`

**Interfaces:**
- Consumes: Task 1 validator and the exact top-level `evaluation_manifest_sha256`.
- Produces: one canonical selection report whose authority bindings include the sidecar file/internal evidence hashes.

- [ ] **Step 1: Write failing consumer/CLI tests**

Update payload fixtures to include `evaluation_manifest_sha256`. Add an integration test that passes the CLI-loaded schema-2 object into the real repository consumer and verifies the resulting report binds the fixed evaluation manifest. Add rejection tests for a missing/extra field.

- [ ] **Step 2: Run tests and record RED**

Run:

```bash
/tmp/maskimpute-supported/bin/python -m pytest tests/test_candidate_selection.py tests/test_selection_authority.py -q -k 'cli or result_payload or ready_public_selection'
```

Expected: fail because the consumer and CLI still use the seven-field schema.

- [ ] **Step 3: Implement the minimal integration**

Add `evaluation_manifest_sha256` to the exact consumer and CLI field sets. Validate `result_sha256` over the seven evidence fields plus the sidecar file hash, call Task 1 before candidate evaluation, and merge returned hashes into `authority_bindings`.

- [ ] **Step 4: Run focused tests and record GREEN**

Run both test files, distinguishing the two known ready/pending baseline assertions from new failures.

---

### Task 3: Verification and independent review

**Files:**
- Verify: `maskimpute_benchmark/selection.py`
- Verify: `scripts/select_development_candidate.py`
- Verify: `tests/test_selection_authority.py`
- Verify: `tests/test_candidate_selection.py`
- Verify: `tests/test_development_evaluation.py`

**Interfaces:**
- Consumes: completed Tasks 1-2.
- Produces: one reviewed commit ready for the `genome-biology` owner.

- [ ] **Step 1: Run formatting, Ruff, focused and relevant suites**

```bash
/home/marcinmaleclocal/.local/bin/ruff format maskimpute_benchmark/selection.py scripts/select_development_candidate.py tests/test_selection_authority.py tests/test_candidate_selection.py
/home/marcinmaleclocal/.local/bin/ruff check maskimpute_benchmark/selection.py scripts/select_development_candidate.py tests/test_selection_authority.py tests/test_candidate_selection.py tests/test_development_evaluation.py
/tmp/maskimpute-supported/bin/python -m pytest tests/test_candidate_selection.py tests/test_selection_authority.py tests/test_development_evaluation.py -q
git diff --check
```

- [ ] **Step 2: Commit exact owned changes**

Stage only the plan, selection consumer, CLI, and their tests; commit with a schema-2 evidence validation message.

- [ ] **Step 3: Obtain independent review**

Ask a fresh reviewer to inspect the full branch diff for digest cycles, authority laundering, path aliasing, TOCTOU gaps, incomplete denominators, and test blind spots. Fix every Critical or Important issue and rerun verification.
