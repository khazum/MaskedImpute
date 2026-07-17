# Pre-Zero Semantic Resume Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make development and final resume validation recompute every stored `p_pre_zero` score report from the persisted realized matrix and authoritative evaluator-private data, while binding the score policy to independent execution authority.

**Architecture:** A shared runner helper extracts raw observed counts, truth, and truth kind from `PreparedDataset`; both fresh evaluation and persisted validation use that helper. `validate_stored_prezero_evidence` requires those arrays plus independently derived score-policy digests, decompresses the realized matrix once, rebuilds the complete overall and stratified report with `_score_report`, and requires byte-canonical equality. Development and final stores receive `PreparedDataset` mappings explicitly; there is no count/shape-only or missing-authority fallback.

**Tech Stack:** Python 3.11, NumPy, zlib, canonical JSON/SHA-256, pytest.

## Global Constraints

- Never reconstruct the realized `p_pre_zero` matrix from receipts.
- Use the persisted, bounded-decompressed little-endian float64 matrix as the report input.
- Require evaluator-private observed/truth/truth-kind authority on every persisted-record validation path.
- Recompute all overall, library-size-quartile, truth-expression-bin, metric, and reliability-bin fields exactly.
- Keep count-model input SHA-256 and runner method-input SHA-256 in their distinct documented digest domains.
- Preserve explicit non-MaskImpute and noncompleted evidence rows.

---

### Task 1: Fail on coordinated report and policy drift

**Files:**
- Modify: `tests/test_prezero_evidence.py`
- Modify: `maskimpute_benchmark/prezero_evidence.py`

**Interfaces:**
- Consumes: `validate_stored_prezero_evidence(...)`, authoritative raw `observed`, `truth`, and `truth_kind`.
- Produces: required validator keywords `observed`, `truth`, `truth_kind`, `expected_score_input_sha256`, and `expected_score_config_sha256`.

- [ ] **Step 1: Write the failing tests**

Add one test that keeps the compressed matrix unchanged, changes a valid Brier value, refreshes `evidence_sha256`, and expects `PreZeroEvidenceError`. Add a second test that changes `policy.score_input_sha256`, refreshes policy/semantic/evidence hashes, and expects `PreZeroEvidenceError` against the independently recomputed count-matrix digest.

```python
record["overall"]["metrics"]["brier"]["value"] = 0.123456
record["evidence_sha256"] = canonical_sha256(_evidence_body(record))
with pytest.raises(PreZeroEvidenceError, match="report differs"):
    validate_stored_prezero_evidence(
        record,
        observed=observed,
        truth=truth,
        truth_kind="exact_pre_capture",
        expected_score_input_sha256=count_input_sha256,
        expected_score_config_sha256=score_config_sha256,
        **required_bindings,
    )
```

- [ ] **Step 2: Run tests to verify RED**

Run: `/tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider tests/test_prezero_evidence.py -k 'metric_drift or policy_drift'`

Expected: FAIL because coordinated valid-range metric/policy drift is currently accepted or because the wished-for required authority keywords do not exist.

- [ ] **Step 3: Implement exact semantic validation**

After bounded decompression, require the stored `truth_kind` to equal the authoritative truth kind, require score input/config/source/calibration fields to match independent authority, call `_score_report` with the decompressed matrix (or the explicit unavailable state), and compare canonical bytes for both `overall` and `strata`.

```python
expected_overall, expected_strata = _score_report(
    probability,
    observed,
    truth,
    truth_kind=truth_kind,
    unavailable_status=None if evidence_status == "completed" else evidence_status,
    unavailable_reason=evidence_reason,
)
if _canonical_bytes(value["overall"]) != _canonical_bytes(expected_overall):
    raise PreZeroEvidenceError("p_pre_zero overall report differs from authority")
if _canonical_bytes(value["strata"]) != _canonical_bytes(expected_strata):
    raise PreZeroEvidenceError("p_pre_zero stratified report differs from authority")
```

- [ ] **Step 4: Run tests to verify GREEN**

Run the RED command again, then run the complete `tests/test_prezero_evidence.py` module.

- [ ] **Step 5: Commit**

Commit the validator and direct tests as one reviewable TDD slice.

### Task 2: Require PreparedDataset authority for development resume

**Files:**
- Modify: `maskimpute_benchmark/runner.py`
- Modify: `maskimpute_benchmark/development_evaluation.py`
- Modify: `maskimpute_benchmark/evaluation_manifest.py`
- Modify: `maskimpute_benchmark/revision_evaluation.py`
- Modify: `tests/test_prezero_evidence.py`
- Modify: `tests/test_benchmark_runner.py`
- Modify: `tests/test_development_evaluation.py`

**Interfaces:**
- Consumes: `Mapping[str, PreparedDataset]` keyed by plan dataset ID.
- Produces: mandatory prepared-data arguments on `CheckpointStore.load`, `write`, and `append`, and on `load_completed_reconstruction_checkpoint`.

- [ ] **Step 1: Write the failing integration test**

Persist a valid development checkpoint, rehash a plausible metric mutation, then call the wished-for `store.load(plan, prepared_datasets)` and require rejection.

```python
prepared_by_id = {prepared.binding.dataset_id: prepared}
_rewrite_checkpoint(store, mutate_valid_brier_and_refresh_hashes)
with pytest.raises(RunnerContractError, match="report differs"):
    store.load(plan, prepared_by_id)
```

- [ ] **Step 2: Run test to verify RED**

Run: `/tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider tests/test_prezero_evidence.py -k development_checkpoint_rejects_metric_drift`

Expected: FAIL because `CheckpointStore.load` does not yet accept or use prepared-data authority.

- [ ] **Step 3: Thread the authority without fallback**

Extract score targets through one `_prezero_evaluator_targets(prepared)` helper shared with fresh evaluation. Validate mapping coverage and dataset identity before every stored record. Pass count-model input/config and calibration digests from prepared data and the plan/execution authority into `validate_stored_prezero_evidence`. Update every production loader and test fixture to supply the exact prepared mapping.

```python
def load(
    self,
    plan: CompetitionPlan,
    prepared_datasets: Mapping[str, PreparedDataset],
) -> CheckpointReport:
    ...

observed, truth, truth_kind = _prezero_evaluator_targets(prepared)
validate_stored_prezero_evidence(
    evidence_value,
    observed=observed,
    truth=truth,
    truth_kind=truth_kind,
    expected_score_input_sha256=_count_score_input_sha256(
        prepared.method_input.counts
    ),
    expected_score_config_sha256=expected_score_config_sha256,
    **record_bindings,
)
```

- [ ] **Step 4: Run development GREEN suites**

Run: `/tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider tests/test_prezero_evidence.py tests/test_benchmark_runner.py tests/test_development_evaluation.py`

Expected: all pass.

- [ ] **Step 5: Commit**

Commit the development threading slice.

### Task 3: Require PreparedDataset authority for final resume

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Modify: `tests/test_final_runner.py`

**Interfaces:**
- Consumes: final `PreparedDataset` mapping and `ExecutionAuthorityContext` score/config/calibration bindings.
- Produces: a `FinalResultStore` that cannot append, resume, or finalize a record without evaluator-private authority for its dataset.

- [ ] **Step 1: Write the failing final-record tamper test**

Persist a completed MaskImpute final record, change a plausible score metric, refresh record/evidence hashes as applicable, and require `load_records()` to reject it while the score artifact remains unchanged.

```python
_rewrite_final_record(record_path, mutate_valid_brier_and_refresh_evidence_hash)
with pytest.raises(FinalRunnerContractError, match="report differs"):
    FinalResultStore(output, plan, prepared_by_id, authority).load_records()
```

- [ ] **Step 2: Run test to verify RED**

Run: `/tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider tests/test_final_runner.py -k prezero_metric_drift`

Expected: FAIL because `FinalResultStore` does not yet retain prepared-data authority.

- [ ] **Step 3: Thread final authority**

Require the prepared mapping and score execution authority in `FinalResultStore`, select the exact prepared dataset by plan entry, and pass it through both append-time and resume-time `_validate_stored_record` calls. Update the production final-round constructor and test fixtures.

```python
class FinalResultStore:
    def __init__(
        self,
        output_dir: Path,
        plan: FinalExecutionPlan,
        prepared_datasets: Mapping[str, PreparedDataset],
        execution_authority: ExecutionAuthorityContext,
    ) -> None:
        ...
```

- [ ] **Step 4: Run final GREEN suites**

Run: `/tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider tests/test_final_runner.py tests/test_prezero_evidence.py`

Expected: all pass.

- [ ] **Step 5: Commit**

Commit the final threading slice.

### Task 4: Verify the complete fix

**Files:**
- Modify only files required by failures introduced by Tasks 1-3.

**Interfaces:**
- Consumes: all three committed TDD slices.
- Produces: one clean branch with exact RED/GREEN and static-verification evidence.

- [ ] **Step 1: Run focused and adjacent tests**

Run the complete prezero, runner, development-evaluation, final-runner, metrics, and scaling-panel test modules with `/tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider`.

- [ ] **Step 2: Run static checks**

Run `/tmp/maskimpute-supported/bin/python -m ruff check` on modified Python files, `/tmp/maskimpute-supported/bin/python -m compileall -q` on modified package files, and `git diff --check`.

- [ ] **Step 3: Inspect branch state**

Require a clean worktree, review the exact base-to-head diff, and report the commit range plus RED/GREEN command outputs.
