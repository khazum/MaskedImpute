# Schema-4 Selection Promotion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable production promotion from each exact schema-2/3 development selection input and cumulative downstream archive to the only schema-4 input accepted by stage selectors.

**Architecture:** Extend the closed revision path model with distinct source and selection-complete inputs, strengthen schema-4 attachment with an exact source-input projection, and add one fixed-path promotion service plus a no-option CLI. Producers keep writing schema 2/3, downstream planning keeps reading those source inputs, while selectors, revision activation, and publication freeze consume only downstream-suffixed schema 4.

**Tech Stack:** Python 3.12, canonical JSON and SHA-256 receipts, POSIX atomic hard links and `fsync`, pytest, Ruff.

## Global Constraints

- Work only in `.worktrees/schema4-selection-promotion` from integration commit `3485d03`.
- Preserve schema-2/3 inputs as immutable source evidence; never replace them.
- Production stage and path choices are repository-owned and expose no CLI override.
- Promotion before the latest discovered stage's downstream manifest exists must fail and must not fall back to an earlier stage.
- Write and run each production-shaped regression before its implementation change.
- Do not run development or final evidence.

---

### Task 1: Closed source and selection-complete path contract

**Files:**
- Modify: `maskimpute_benchmark/revisions.py`
- Modify: `maskimpute_benchmark/runner.py`
- Test: `tests/test_revision_authority.py`
- Test: `tests/test_maskimpute_v28.py`

**Interfaces:**
- Consumes: `revision_stage_paths(version: str) -> RevisionStagePaths`.
- Produces: `DevelopmentSelectionStagePaths`, `development_selection_stage_paths(through_version: str | None)`, and `RevisionStagePaths.selection_complete_input`.

- [ ] **Step 1: Write failing fixed-path tests**

Add assertions that base/v28/v29 source inputs remain unsuffixed or version-suffixed, their complete inputs end in `-downstream.json`, v28 activation consumes base complete input, v29 activation consumes v28 complete input, and `_V28_SELECTION_INPUT_PATH` equals the base complete input.

- [ ] **Step 2: Verify the new path assertions fail**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_revision_authority.py::test_revision_stage_paths_are_fixed_and_version_separated \
  tests/test_maskimpute_v28.py::test_selection_cli_and_v28_activation_share_canonical_fixed_paths \
  -q -W error
```

Expected: failures because complete-input fields and downstream-suffixed activation paths do not exist.

- [ ] **Step 3: Implement the closed path model**

Add the exact model:

```python
@dataclass(frozen=True, slots=True)
class DevelopmentSelectionStagePaths:
    through_version: str | None
    source_selection_input: str
    selection_complete_input: str
    selection_report: str
    downstream_directory: str

def development_selection_stage_paths(
    through_version: str | None,
) -> DevelopmentSelectionStagePaths:
    if through_version not in {None, "v28", "v29"}:
        raise ValueError("through_version must be null, v28, or v29")
    suffix = "" if through_version is None else f"-{through_version}"
    return DevelopmentSelectionStagePaths(
        through_version=through_version,
        source_selection_input=f"{_EVALUATION_ROOT}/development_selection_input{suffix}.json",
        selection_complete_input=f"{_EVALUATION_ROOT}/development_selection_input{suffix}-downstream.json",
        selection_report=f"{_EVALUATION_ROOT}/development_selection_report{suffix}.json",
        downstream_directory=f"{_EVALUATION_ROOT}/downstream{suffix}",
    )
```

Make `revision_stage_paths` retain `selection_input` as the source, add `selection_complete_input`, and derive `activation_selection_input` from the prior stage's complete path. Point the runner's fixed base selection input to the complete path.

- [ ] **Step 4: Run the fixed-path tests to green**

Run the Step 2 command. Expected: both pass.

- [ ] **Step 5: Commit the path contract**

```bash
git add maskimpute_benchmark/revisions.py maskimpute_benchmark/runner.py \
  tests/test_revision_authority.py tests/test_maskimpute_v28.py
git commit -m "refactor: separate selection source and complete paths"
```

### Task 2: Exact source projection in schema 4

**Files:**
- Modify: `maskimpute_benchmark/selection.py`
- Test: `tests/test_downstream_evidence.py`
- Test: `tests/test_selection_authority.py`

**Interfaces:**
- Consumes: a canonical source input at `development_selection_stage_paths(stage).source_selection_input` and fixed downstream evidence.
- Produces: schema-4 binding fields `source_selection_input_path`, `source_selection_input_file_sha256`, and `source_selection_result_sha256`, plus read-only source projection validation.

- [ ] **Step 1: Write failing base and revision source-binding tests**

Persist the schema-2/3 source fixture at its fixed source path before calling `attach_downstream_evidence_to_selection_result`. Assert all three source fields are present. Add mutations for the source bytes, source path, source file hash, source result hash, an extra source schema field, and a mismatched revision chain; each must raise `SelectionAuthorityError` before selection.

- [ ] **Step 2: Verify source-binding tests fail**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_downstream_evidence.py -k 'selection_schema_four or revision_development_downstream' \
  tests/test_selection_authority.py -k 'schema_four' -q -W error
```

Expected: missing source-binding assertions or unexpected acceptance.

- [ ] **Step 3: Implement source schema and projection validation**

Require exact source fields for schema 2 and schema 3, canonical checksum, and exact revision chains. During attachment, securely re-read the inferred fixed source and require `source == payload`; bind its path, file digest, and source result digest. Add a helper equivalent to:

```python
def _validate_schema_four_source_projection(repository, data, binding):
    stage = None if data["revision_versions"] == [] else data["revision_versions"][-1]
    paths = development_selection_stage_paths(stage)
    source, file_sha = _read_canonical_json(
        repository / paths.source_selection_input,
        "source development selection input",
        indented=False,
    )
    projected = {key: value for key, value in data.items()
                 if key not in {"downstream_evidence", "result_sha256"}}
    projected["schema_version"] = 2 if stage is None else 3
    if stage is None:
        projected.pop("revision_versions")
    projected["result_sha256"] = binding["source_selection_result_sha256"]
    if source != projected or file_sha != binding["source_selection_input_file_sha256"]:
        raise SelectionAuthorityError("promoted selection source differs")
```

Call the helper in `_select_for_repository` before downstream completeness validation.

- [ ] **Step 4: Run source-binding tests to green**

Run the Step 2 command. Expected: selected tests pass warning-clean.

- [ ] **Step 5: Commit the source binding**

```bash
git add maskimpute_benchmark/selection.py tests/test_downstream_evidence.py \
  tests/test_selection_authority.py
git commit -m "feat: bind schema-4 inputs to immutable source evidence"
```

### Task 3: Immutable promotion API and fixed CLI

**Files:**
- Create: `maskimpute_benchmark/selection_promotion.py`
- Create: `scripts/promote_development_selection_input.py`
- Create: `tests/test_selection_promotion.py`

**Interfaces:**
- Consumes: `promote_development_selection_input(repository: Path, through_version: str | None)` and `development_downstream_revision_version(repository)`.
- Produces: `SelectionPromotionReceipt.to_dict()` and `promote_latest_development_selection_input(repository)`.

- [ ] **Step 1: Write failing promotion lifecycle tests**

Use production-shaped source and downstream fixtures to test base, v28, and v29 promotion; wrong-round evidence; invalid source schema; source and manifest tampering; conflicting destination retention; identical idempotent retry; simulated `os.link` interruption with no destination or temporary residue; and latest-v29 source with absent v29 downstream evidence refusing to fall back to complete base/v28 evidence.

- [ ] **Step 2: Verify lifecycle tests fail on the missing module**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_selection_promotion.py -q -W error
```

Expected: collection or import failure for `maskimpute_benchmark.selection_promotion`.

- [ ] **Step 3: Implement atomic immutable publication and promotion**

Implement canonical encoding, same-directory `mkstemp`, write/file `fsync`, `os.link` create, directory `fsync`, cleanup, exact-existing idempotence, and conflict rejection. The promotion implementation must use only `development_selection_stage_paths(stage)`, call `attach_downstream_evidence_to_selection_result`, publish to `selection_complete_input`, securely re-read the published file, and return exact source, manifest, payload, and file digests.

- [ ] **Step 4: Implement the no-option CLI**

The CLI parses no options, calls `promote_latest_development_selection_input(REPOSITORY_ROOT)`, prints only the canonical receipt JSON, returns 0 on success, and returns 2 with canonical error JSON on validation or publication failure. It must never catch a missing latest-stage archive and retry an earlier stage.

- [ ] **Step 5: Run promotion tests to green**

Run the Step 2 command. Expected: all promotion tests pass warning-clean.

- [ ] **Step 6: Commit the promotion service**

```bash
git add maskimpute_benchmark/selection_promotion.py \
  scripts/promote_development_selection_input.py tests/test_selection_promotion.py
git commit -m "feat: promote immutable schema-4 selection evidence"
```

### Task 4: Fail-closed production consumers and runbook

**Files:**
- Modify: `scripts/select_development_candidate.py`
- Modify: `maskimpute_benchmark/revision_commands.py`
- Modify: `maskimpute_benchmark/publication_freeze.py`
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_selection_authority.py`
- Modify: `tests/test_revision_authority.py`
- Modify: `tests/test_freeze_publication_round.py`
- Create: `docs/development-selection-workflow.md`

**Interfaces:**
- Consumes: only `selection_complete_input` for base/v28/v29 selection and activation.
- Produces: a documented acyclic base/v28/v29 command sequence with no schema-2/3 selection fallback.

- [ ] **Step 1: Write failing consumer and CLI-contract tests**

Assert base selector default, revision selectors, revision activation, runner activation, and publication freeze point to downstream-suffixed inputs. Assert revision source detection requires schema 3 at source paths. Assert the promotion CLI help exposes none of `--input`, `--output`, `--version`, `--through-version`, or `--downstream`.

- [ ] **Step 2: Verify consumer tests fail**

Run:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_selection_authority.py tests/test_revision_authority.py \
  tests/test_freeze_publication_round.py -k 'path or entry_point or schema_four or fixed' \
  -q -W error
```

Expected: old base/revision selection paths are still observed.

- [ ] **Step 3: Switch consumers without fallback**

Change the base selector default and publication `_FIXED_PATHS["selection_input"]` to the base complete path. Change `select_revision_main` to read `paths.selection_complete_input`. Keep revision builders and `development_downstream_revision_version` on `paths.selection_input`, but require schema 3 there so a promoted artifact cannot masquerade as a source.

- [ ] **Step 4: Document the exact command order**

Document base build → downstream run → promotion → base selection; then, only on an exact trigger, v28 run → v28 build → downstream run → promotion → v28 selection; and the equivalent v29 sequence. State that every command fails closed and no selector reads a source input.

- [ ] **Step 5: Run consumer tests to green and commit**

Run the Step 2 command, then:

```bash
git add scripts/select_development_candidate.py maskimpute_benchmark/revision_commands.py \
  maskimpute_benchmark/publication_freeze.py maskimpute_benchmark/downstream_evidence.py \
  tests/test_selection_authority.py tests/test_revision_authority.py \
  tests/test_freeze_publication_round.py docs/development-selection-workflow.md
git commit -m "fix: require selection-complete development inputs"
```

### Task 5: Integrated verification

**Files:**
- Review every file changed in Tasks 1–4.

**Interfaces:**
- Consumes: the complete promotion and consumer flow.
- Produces: a clean exact commit range ready for integration review.

- [ ] **Step 1: Run focused warning-strict tests**

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_selection_promotion.py tests/test_downstream_evidence.py \
  tests/test_selection_authority.py tests/test_revision_authority.py \
  tests/test_revision_evaluation.py tests/test_maskimpute_v28.py \
  tests/test_freeze_publication_round.py -q -W error
```

- [ ] **Step 2: Run static verification**

```bash
/home/marcinmaleclocal/.local/bin/ruff check \
  maskimpute_benchmark/revisions.py maskimpute_benchmark/selection.py \
  maskimpute_benchmark/selection_promotion.py maskimpute_benchmark/revision_commands.py \
  maskimpute_benchmark/downstream_evidence.py maskimpute_benchmark/publication_freeze.py \
  maskimpute_benchmark/runner.py scripts/promote_development_selection_input.py \
  scripts/select_development_candidate.py tests/test_selection_promotion.py
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m compileall -q \
  maskimpute_benchmark scripts
git diff --check 3485d03..HEAD
```

- [ ] **Step 3: Review and report**

Inspect `git diff --stat 3485d03..HEAD`, `git status --short`, and the exact commit log. Report the commit range, test counts, and any integration conflict expected from concurrent downstream work.
