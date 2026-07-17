# Publication Synthesis Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fail-closed, read-only publication synthesis that permits competitive or endpoint-specific superiority language only from exact, complete frozen-final evidence.

**Architecture:** Extend `final_analysis.py` to compute the reconstruction rank/Pareto claim gate while authoritative normalized records are present. Add `publication_synthesis.py` as a thin production orchestrator that derives fixed evidence locations, replays existing loaders, cross-checks their evaluated-round bindings, computes the final null-DE gate, and emits only canonical in-memory claim permissions.

**Tech Stack:** Python 3.11, immutable dataclasses/mappings, existing MaskImpute canonical SHA-256 contracts, pytest, Ruff.

## Global Constraints

- The public production API accepts exactly `repository: Path` and `round_dir: Path`; it accepts no artifact path, source kind, threshold, candidate, comparator, endpoint, or fixture override.
- Final rank/Pareto and final null-DE are the only numerical gates introduced here.
- Rank and null-DE aggregation always averages model seeds within a dataset view before averaging paired technical views within a biological draw.
- Rank thresholds are median biological-draw rank `<= 2` for `mse`, `mse_dropout`, and `gnrmse`.
- Pareto dimensions are exactly `mse_dropout`, `mse_pre_dropout_zero`, `corr_err`, and `mse_non_dropout_nonzero`.
- Final null-DE uses inclusive limits `maximum FPR <= 0.06` and `maximum paired candidate-minus-observed FPR <= 0.01` over a complete denominator.
- Downstream and scaling numerical statuses are `not_prespecified`; no threshold may be inferred.
- Trajectory is always `descriptive_only` and has no gate influence.
- Binding/tamper/source-kind errors abort synthesis; scientific gates use only `passed`, `failed`, and `unavailable`.
- No scientific artifact or numerical manuscript claim is written.
- Run Python with `PYTHONDONTWRITEBYTECODE=1`, `PYTHONNOUSERSITE=1`, and `LD_LIBRARY_PATH` unset.

---

### Task 1: Authoritative frozen-final reconstruction claim gate

**Files:**
- Modify: `tests/test_final_analysis.py`
- Modify: `maskimpute_benchmark/final_analysis.py`

**Interfaces:**
- Consumes: normalized final records, the validated frozen metric-direction contract, `selection_contract["required_comparator_ids"]`, and protocol primary metrics.
- Produces: `report["reconstruction_claim_gate"]`, including exact rank/Pareto statuses and complete draw-collapsed comparator medians used by synthesis.

- [x] **Step 1: Write failing final-analysis gate tests**

Add a full six-metric panel fixture with two mechanisms, two biological draws,
paired views, three candidate seeds, and deterministic required comparators.
Assert the new section has this behavior:

```python
gate = report["reconstruction_claim_gate"]
assert gate["status"] == "passed"
assert [row["metric"] for row in gate["rank_gates"]] == [
    "mse", "mse_dropout", "gnrmse"
]
assert all(row["median_biological_draw_rank"] <= 2 for row in gate["rank_gates"])
assert gate["pareto_gate"]["dimensions"] == [
    "mse_dropout",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
]
assert gate["pareto_gate"]["non_dominated"] is True
```

Add separate tests that (a) alter duplicated seed/view values without changing
the seed-then-view mean, (b) make a complete median rank exceed 2, (c) make a
required comparator row terminal, (d) make the candidate Pareto dominated, and
(e) verify pre-dropout-zero exact structural unavailability is accepted only
with its truth-kind reason. Expected statuses are respectively unchanged,
`failed`, `unavailable`, `failed`, and passed/unavailable according to the
exact reason.

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_final_analysis.py -k reconstruction_claim_gate -W error
```

Expected: FAIL because `reconstruction_claim_gate` does not exist.

- [x] **Step 3: Implement complete draw collapse, ranks, Pareto, and comparator summaries**

Add private helpers that:

```python
_CLAIM_RANK_METRICS = ("mse", "mse_dropout", "gnrmse")
_CLAIM_PARETO_METRICS = (
    "mse_dropout",
    "mse_pre_dropout_zero",
    "corr_err",
    "mse_non_dropout_nonzero",
)

def _average_rank(target: float, comparators: Sequence[float]) -> float:
    below = sum(value < target for value in comparators)
    tied = sum(value == target for value in comparators)
    return 1.0 + below + tied / 2.0
```

Build one method/metric/draw table only after validating identical applicable
dataset-view identities and successful seed rows for the candidate and every
required comparator. Check the direction contract's immutable `rank_metrics`
and `pareto_metrics` against the constants. Emit `unavailable` rather than
guessing when authority or denominator completeness is absent. Compute the
strongest comparator for each lower-is-better primary metric from complete
draw-collapsed medians, choose the lowest `(median, method_id)`, and disclose all
method IDs tied at the lowest median.

Add the section before calculating `analysis_sha256`:

```python
"reconstruction_claim_gate": _reconstruction_claim_gate(
    evidence,
    candidate=candidate,
    primary_metrics=primary,
    selection_contract=selection_contract,
    metric_direction_contract=metric_direction_contract,
),
```

- [x] **Step 4: Run focused final-analysis tests and verify GREEN**

Run the Step 2 command, then:

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_final_analysis.py -W error
```

Expected: all final-analysis tests pass with no warnings.

---

### Task 2: Pure null-DE and claim-permission synthesis

**Files:**
- Create: `tests/test_publication_synthesis.py`
- Create: `maskimpute_benchmark/publication_synthesis.py`

**Interfaces:**
- Consumes: already replayed primary report, frozen method payload, final downstream plan/manifest, final null-DE plan/manifest, and scaling checkpoint.
- Produces: `_LoadedPublicationEvidence`, `_build_final_null_de_gate(loaded: _LoadedPublicationEvidence) -> dict[str, object]`, and `_build_publication_synthesis(loaded: _LoadedPublicationEvidence) -> dict[str, object]`; no public free-form evidence builder.

- [x] **Step 1: Write failing null-DE and claim tests**

Create complete immutable fixture objects matching the real loader outputs. Add
tests for:

```python
assert synthesis["gates"]["final_null_de"]["status"] == "passed"
assert synthesis["gates"]["final_null_de"]["maximum_fpr"] == pytest.approx(0.06)
assert synthesis["gates"]["final_null_de"]["maximum_above_observed"] == pytest.approx(0.01)
assert synthesis["claim_permissions"]["competitive"] is True
assert synthesis["trajectory"]["role"] == "descriptive_only"
assert synthesis["downstream"]["numerical_gate_status"] == "not_prespecified"
assert synthesis["scaling"]["numerical_gate_status"] == "not_prespecified"
```

Add independent tests for FPR `0.0600001`, paired delta `0.0100001`, one missing
or noncompleted candidate/observed row, mismatched view coverage, a failed
reconstruction gate with favorable trajectory evidence, and an unavailable
reconstruction gate. Assert `failed` for complete threshold violations,
`unavailable` for denominator failures, and never allow trajectory to alter the
competitive result.

Add superiority tests proving it is allowed only for the deterministic
strongest complete comparator when `ci_95_upper < 0`, `holm_status == "ok"`,
`holm_adjusted_p_value <= 0.05`, and the family is
`protocol_primary_metrics`. Test ties, a stronger different comparator, a
zero-crossing interval, missing or partial-family Holm evidence, an excluded
draw with otherwise favorable evidence, and no complete comparator. Require
the strongest comparator's complete draw count, both paired views per draw,
the exact frozen direction source, and no duplicate, zero-denominator,
nonrepresentable, or bootstrap exclusions.

- [x] **Step 2: Run publication-synthesis tests and verify RED**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONDWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_publication_synthesis.py -W error
```

Expected: collection FAIL because `maskimpute_benchmark.publication_synthesis`
does not exist.

- [x] **Step 3: Implement the minimal private synthesis logic**

Create `PublicationSynthesisError` and private helpers for exact finite-number,
digest, binding, null-DE collapse, gate precedence, strongest-comparator, and
superiority validation. Null-DE identities are
`(method_id, mechanism, biological_id, technical_view, dataset_id, model_seed)`;
duplicates are errors. Candidate and observed expected identities come from the
final source plan, not from successful rows.

Use one closed internal carrier so the pure logic and production loader share
the same complete interface:

```python
@dataclass(frozen=True, slots=True)
class _LoadedPublicationEvidence:
    primary_report: Mapping[str, object]
    frozen_method: Mapping[str, object]
    downstream_plan: DownstreamEvidencePlan
    downstream_manifest: DownstreamEvidenceManifest
    null_de_plan: FinalNullDEPlan
    null_de_manifest: FinalNullDEManifest
    scaling_checkpoint: ScalingCheckpoint
```

The canonical result has this closed top-level schema:

```python
body = {
    "schema_version": 1,
    "status": "completed",
    "candidate_method_id": candidate,
    "evidence_bindings": bindings,
    "freeze_prerequisite": freeze_prerequisite,
    "downstream": downstream_summary,
    "scaling": scaling_summary,
    "trajectory": {
        "role": "descriptive_only",
        "gate_influence": "none",
        "evidence_sha256": trajectory_evidence_sha256,
    },
    "gates": {
        "reconstruction": reconstruction_gate,
        "final_null_de": null_de_gate,
        "competitive": competitive_gate,
    },
    "claim_permissions": {
        "competitive": competitive_gate["status"] == "passed",
        "superiority": superiority_rows,
    },
}
return {**body, "synthesis_sha256": canonical_sha256(body)}
```

Keep the evidence-object builder private and out of `__all__`.

- [x] **Step 4: Run the focused synthesis tests and verify GREEN**

Run the Step 2 command. Expected: all publication-synthesis tests pass with no
warnings.

---

### Task 3: Production-only authoritative loader orchestration

**Files:**
- Modify: `tests/test_publication_synthesis.py`
- Modify: `maskimpute_benchmark/publication_synthesis.py`

**Interfaces:**
- Consumes: `generate_publication_synthesis(repository: Path, round_dir: Path)`.
- Produces: `_load_publication_evidence(repository: Path, round_dir: Path) -> _LoadedPublicationEvidence` and one canonical, self-hashed in-memory mapping or raises `PublicationSynthesisError`.

- [x] **Step 1: Write failing production-boundary tests**

Test the exact two-parameter signature and use complete loader return fixtures
to exercise behavior rather than call counts. Assert rejection when:

- downstream `source_kind != "final"` or `evidence_scope != "all"`;
- independently rebuilt and persisted downstream plans differ;
- null-DE's source plan differs from downstream;
- round ID, evaluation receipt, result manifest, final plan, execution
  manifest, scaling, or denominator bindings disagree;
- final analysis lacks `trajectory_evidence_sha256`;
- frozen selected assessment is absent, differs from its gate table, or does
  not have `eligible`, `efficacy_pass`, and `safety_pass` all true; or
- a locally self-hashed mapping is offered without the authoritative loader
  result (there is no public parameter or alternate source kind that can select
  it).

- [x] **Step 2: Run the production-boundary tests and verify RED**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_publication_synthesis.py -k 'production or binding or source_kind' \
  -W error
```

Expected: FAIL because the public production entry point is absent.

- [x] **Step 3: Implement exact location derivation and loader replay**

Implement only:

```python
def generate_publication_synthesis(
    repository: Path,
    round_dir: Path,
) -> dict[str, object]:
    if not isinstance(repository, Path) or not isinstance(round_dir, Path):
        raise TypeError("repository and round_dir must be pathlib.Path values")
    loaded = _load_publication_evidence(repository, round_dir)
    return _build_publication_synthesis(loaded)
```

Inside it, call `generate_final_analysis`, `validate_frozen_method`,
`build_final_downstream_evidence_plan`, the downstream plan/manifest loaders,
`build_final_null_de_plan`, `expected_final_null_de_output_directory`, the
null-DE manifest loader, and `load_publication_scaling_evidence`. Derive the
downstream directory from the evaluated receipt binding using the fixed
`<repo>-final-analysis/downstream/<round>/<receipt>` namespace. Wrap existing
contract errors as `PublicationSynthesisError` while preserving a concise stage
name; never downgrade tamper to a scientific `unavailable` gate.

Export only:

```python
__all__ = ["PublicationSynthesisError", "generate_publication_synthesis"]
```

- [x] **Step 4: Run all publication-synthesis tests and verify GREEN**

Run the Step 2 command without `-k`. Expected: all tests pass with no warnings.

---

### Task 4: Cross-module verification and atomic handoff

**Files:**
- Review: `maskimpute_benchmark/final_analysis.py`
- Review: `maskimpute_benchmark/publication_synthesis.py`
- Review: `tests/test_final_analysis.py`
- Review: `tests/test_publication_synthesis.py`
- Review: `docs/superpowers/specs/2026-07-17-publication-synthesis-safety-design.md`

**Interfaces:**
- Consumes: completed implementation and tests.
- Produces: one verified atomic commit based on `584300b55038f60abccd2fc068a7023ededa16d6`.

- [x] **Step 1: Run warning-strict relevant suites**

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_final_analysis.py tests/test_publication_synthesis.py \
  tests/test_downstream_evidence.py tests/test_final_null_de.py \
  tests/test_scaling_panel.py tests/test_trajectory_dataset.py -W error
```

Require zero failures and zero warnings. Do not run runtime-lock test nodes.

- [x] **Step 2: Run static verification**

```bash
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m ruff check \
  maskimpute_benchmark/final_analysis.py \
  maskimpute_benchmark/publication_synthesis.py \
  tests/test_final_analysis.py tests/test_publication_synthesis.py
env -u LD_LIBRARY_PATH PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m compileall -q \
  maskimpute_benchmark/final_analysis.py \
  maskimpute_benchmark/publication_synthesis.py \
  tests/test_final_analysis.py tests/test_publication_synthesis.py
git diff --check 584300b55038f60abccd2fc068a7023ededa16d6..HEAD
```

- [x] **Step 3: Self-review requirements and repository state**

Re-read the design line by line, inspect `git diff --stat` and `git status
--short`, and confirm no scientific artifact, numerical claim file, test-only
production source kind, caller-selected evidence path, or runtime-lock edit was
introduced.

- [x] **Step 4: Amend into one atomic commit and report the exact range**

Stage the plan, implementation, and tests, then amend the design commit with:

```bash
git commit --amend --no-edit
git rev-parse 584300b55038f60abccd2fc068a7023ededa16d6
git rev-parse HEAD
git status --short
```

Report exact range `584300b55038f60abccd2fc068a7023ededa16d6..HEAD` and do not merge.
