# Revision-Aware Publication Freeze Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans to
> implement this plan task by task.

**Goal:** Freeze the exact terminal base/v28/v29 schema-4 development selection,
its complete activation chain, and its closed evidence inventory without
fallback, downgrade, or mutable publication.

**Architecture:** Resolve the latest known operational stage footprint before
reading selection evidence; validate every source/promoted input, report,
activation, downstream archive, evaluation manifest, reconstruction checkpoint,
and orthogonal archive in the consecutive prefix; then build a stage-qualified
artifact inventory and versioned stage receipt. Preserve the base comparator
denominator while substituting the selected revision candidate's execution
evidence. Revalidate the same receipt at round freeze and publish
`frozen_method.json` with create-only, concurrent-identical idempotence.

**Tech stack:** Python 3.12, canonical JSON/SHA-256 receipts, POSIX `openat` and
hard-link publication, Git-bound tracked authorities, pytest, Ruff.

## Prerequisites and constraints

- Integrate the schema-4 selection-promotion branch before production changes.
  Its reviewed contract must provide `development_selection_stage_paths`,
  immutable schema-4 complete inputs, selectors/activations that consume only
  complete inputs, and immutable/idempotent reports.
- Rebase this plan's implementation branch onto that integration result.
- Do not change fixed scientific configuration, selection policy, stage paths,
  or the frozen method's outward `schema_version == 1`.
- Do not add a stage/path CLI option and do not run development or final
  evidence.
- Write each regression first, run it red for the intended reason, implement
  the smallest production change, then run it green.

---

### Task 1: Add the closed stage-footprint resolver

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py`
- Test: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Add private immutable stage/layout records in `publication_freeze.py`.
- Add `_resolve_publication_stage(repository: Path) -> PublicationStageLayout`.
- Consume only `development_selection_stage_paths` and
  `revision_stage_paths` for stage-owned paths.

- [ ] Add table-driven tests for complete base, v28, and v29 footprints. Assert
  exact stage order, revision versions, active stage, activation paths, and
  stage-qualified core paths.
- [ ] Add a parameterized test over source input, promoted input, report,
  downstream directory/manifest, evaluation manifest, reconstruction
  directory/checkpoint, and orthogonal directory/manifest. For each v28/v29
  component, create only that newest component over a complete earlier stage
  and assert the resolver selects the newest stage and fails as partial; it
  must never return the earlier layout.
- [ ] Add tests for broken symlinks, an empty newest directory, reordered v29
  without v28, and an unknown stage-family suffix. Each must fail closed.
- [ ] Run the new tests and confirm they fail because the resolver is absent:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'publication_stage or newest_footprint or unknown_stage' \
  -q -W error -p no:cacheprovider
```

- [ ] Implement immutable path/layout records, `lexists` footprint probing,
  newest-first stage selection, exact prefix construction, and safe-path/type
  checks. Exclude pretracked revision specs from presence detection, but retain
  their fixed paths in activated layouts.
- [ ] Run the focused tests to green and commit:

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "refactor: resolve publication freeze stage"
```

---

### Task 2: Revalidate the exact schema-4 selection and activation prefix

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py`
- Test: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Add `_validate_publication_stage_evidence(repository, layout)` returning an
  immutable stage evidence receipt plus the active selection report.
- Reuse `_select_for_repository`, source projection/downstream validators, and
  `validate_revision_activation`; do not duplicate their scientific logic.

- [ ] Extend the repository fixture to produce production-shaped schema-2 base
  and schema-3 revision sources, schema-4 promoted inputs, canonical reports,
  downstream plan/manifest/records, evaluation manifests, reconstruction
  checkpoints, and orthogonal manifests at fixed paths.
- [ ] Add successful base, v28, and v29 tests. Assert base has no activation;
  v28 retains the exact base `trigger=v28` activation; v29 retains exact base
  and v28 activations with triggers `v28` then `v29`; the active report is
  byte-equal to recomputation and has `freeze_candidate`.
- [ ] Add failing tests for schema-2/3 used as complete input, wrong source
  projection/path/file/result hash, wrong downstream path/manifest/plan/source
  binding, report byte drift, missing/extra report fields, wrong activation
  hash, a preceding selected configuration, terminal revision trigger,
  `downgrade_claim`, and selected version older than the resolved stage.
- [ ] Add a specific no-fallback regression: retain a valid freezeable v28
  report, create a malformed or downgrade v29 footprint, and assert freeze
  fails without reading v28 as the active report.
- [ ] Run these tests red:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'schema_four_stage or activation_chain or no_selection_fallback or downgrade' \
  -q -W error -p no:cacheprovider
```

- [ ] Implement secure canonical reads and retained file hashes, call the
  existing schema-4 validators, recompute every prefix report, independently
  call `validate_revision_activation` for activated revisions, and enforce the
  report state machine and active selected version.
- [ ] Reread every prefix input/report/downstream manifest after semantic
  validation and reject any byte or stage change.
- [ ] Run the focused tests to green and commit:

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "feat: validate revision selection freeze chain"
```

---

### Task 3: Build the dynamic closed artifact and tree inventory

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py`
- Test: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Replace monolithic `_FIXED_PATHS` classification with common tracked/common
  development maps plus `_publication_artifact_paths(layout)`.
- Add exact `development_stage_receipt` construction and validation.
- Add fixed-directory tree receipts for each stage's reconstruction,
  orthogonal, and downstream roots.

- [ ] Add base/v28/v29 expected-key tests for the eight stage-qualified keys
  per stage. Assert one key per core file, exact fixed paths, both v28/v29
  tracked authorities, both scaling and trajectory panel authorities, and the
  conditional external-reference checkpoint only. Assert the trajectory panel
  never participates in stage-presence resolution.
- [ ] Add tests rejecting missing, extra, wrong-stage, unsafe, duplicate-path,
  or legacy alias artifact bindings. Add tests for a mismatched
  `artifact_names`, reordered stage list, changed activation reference, and
  invalid `inventory_sha256`.
- [ ] Add tests that add an unreferenced file or coherently replace a manifest
  and all of its referenced hashes inside reconstruction, orthogonal, or
  downstream directories. The per-stage tree receipt must change and block
  idempotent preparation/round freeze.
- [ ] Add directory-closure tests for a symlink, FIFO, socket, and any other
  practical special-file fixture. Every stage tree must accept only directories
  and unique regular files.
- [ ] Run the new inventory tests red:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'dynamic_artifact or stage_receipt or tree_receipt' \
  -q -W error -p no:cacheprovider
```

- [ ] Implement the exact stage-qualified mapping; add
  `study/v29_revision.json` and `study/trajectory_panel.json` to common tracked
  authorities; and compute fixed-directory closed tree receipts using the
  existing secure receipt primitives where possible. Reject every symlink and
  special file. Hash the exact stage receipt body and selected artifact
  binding set.
- [ ] Make `_artifact_bindings` reject duplicate normalized paths as well as
  malformed keys/rows. Require exact dynamic key-set equality in clean
  validation rather than subset/superset checks.
- [ ] Recompute all core and tree receipts after stage replay and require exact
  equality before returning the expected frozen payload.
- [ ] Run the focused tests to green and commit:

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "feat: bind dynamic publication evidence inventory"
```

---

### Task 4: Add exact v29 configuration and mixed execution provenance

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py`
- Test: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Generalize `_candidate_configuration` over base search plus exact v28/v29
  tracked specifications.
- Add `_active_execution_evidence(...)` that retains base comparators and swaps
  only selected MaskImpute revision evidence.

- [ ] Add a v29 frozen-method test asserting exact tracked v29 configuration,
  configuration SHA-256, selected assessment version, and active-stage match.
  Add rejection tests for v29 authority bytes/hash/parent/configuration drift
  and duplicate selected IDs across authorities.
- [ ] Build distinguishable base, v28, and v29 checkpoints. For a v28/v29
  selection, assert the frozen MaskImpute row names and hashes the selected
  stage checkpoint while observed, learned comparators, and capacity-matched
  control retain base checkpoint evidence. Assert external-only methods retain
  the external-reference checkpoint.
- [ ] Add tests rejecting a selected-stage checkpoint with no selected
  MaskImpute configuration, multiple selected configurations, a wrong
  configuration hash, or attempts to replace comparator evidence from a
  revision checkpoint.
- [ ] Run these tests red:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'v29_configuration or selected_stage_execution or base_comparator_denominator' \
  -q -W error -p no:cacheprovider
```

- [ ] Load revision configurations through the tracked revision validator,
  require one selected authority and active-version equality, derive base
  method evidence first, independently validate selected-stage evidence, and
  replace only the `maskimpute` row. Use stage-qualified checkpoint artifact
  names in each evidence row.
- [ ] Run the focused tests to green and commit:

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "feat: freeze revision candidate execution evidence"
```

---

### Task 5: Rebuild and validate the dynamic frozen payload

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py`
- Test: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Extend `build_frozen_method_payload` with the exact
  `development_stage_receipt`.
- Make `_expected_frozen_method`, `_validate_clean_frozen_method`, and
  `_validate_development_evidence_package` consume the same derived layout and
  exact binding schema.

- [ ] Add payload unit tests for receipt retention and outer `payload_sha256`.
  Mutate every receipt field, stage order, activation, inventory hash, and
  dynamic artifact membership and assert clean validation rejects it.
- [ ] Add base/v28/v29 prepare-and-validate lifecycle tests. Clean validation
  must still work after ignored development evidence is removed, but must
  reopen and compare all common tracked authorities including v29, scaling,
  and trajectory.
- [ ] Add raw round-freeze tests that re-resolve the stage and rehash the exact
  dynamic package. After preparing a base/v28 receipt, introduce any higher
  stage footprint and assert round freeze rejects stage drift rather than
  freezing the older method.
- [ ] Add a preparation race test that introduces v29 between initial v28
  resolution and the final reread. Assert no frozen file is published.
- [ ] Run lifecycle tests red:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'stage_receipt_payload or revision_prepare or raw_stage_drift or stage_race' \
  -q -W error -p no:cacheprovider
```

- [ ] Thread the receipt through payload building. During clean validation,
  validate its closed schema and tracked bindings without requiring ignored
  files. During prepare and round freeze, independently derive the raw receipt
  and require exact equality before and after operational-root enumeration.
- [ ] Preserve the direct-parent/sole-commit rule for the tracked frozen
  receipt and existing final-round call signature.
- [ ] Run focused lifecycle tests to green and commit:

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "feat: seal revision-aware frozen method receipt"
```

---

### Task 6: Make create-only publication concurrently idempotent

**Files:**

- Modify: `maskimpute_benchmark/publication_freeze.py`
- Test: `tests/test_freeze_publication_round.py`

**Interfaces:**

- Strengthen `_atomic_write(path: Path, raw: bytes)` without introducing an
  overwrite path.

- [ ] Replace the old concurrent-publication expectation with two regressions:
  a concurrent byte-identical destination succeeds and returns the same
  payload; a concurrent different destination raises and preserves the
  competing bytes exactly.
- [ ] Add sequential identical retry, unsafe/symlink/multiply-linked existing
  target, simulated interruption before link, and post-link failure/retry
  tests. Assert no temporary residue and no target overwrite.
- [ ] Run these tests red:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  -k 'idempotent or concurrent or atomic_write or temporary_residue' \
  -q -W error -p no:cacheprovider
```

- [ ] On `FileExistsError`, securely read through the pinned parent descriptor
  and accept only exact raw-byte equality. Retain create-only hard-link
  publication, file/directory fsync, unique-regular-file checks, and temporary
  cleanup. Never unlink or replace the destination.
- [ ] Run the focused tests to green and commit:

```bash
git add maskimpute_benchmark/publication_freeze.py \
  tests/test_freeze_publication_round.py
git commit -m "fix: make frozen receipt publication idempotent"
```

---

### Task 7: Document and verify the complete workflow

**Files:**

- Modify: `docs/development-selection-workflow.md`
- Review: `maskimpute_benchmark/publication_freeze.py`
- Review: `scripts/freeze_publication_round.py`
- Review: `tests/test_freeze_publication_round.py`

- [ ] Document exact base promotion/selection/freeze order, conditional
  base-to-v28 and v28-to-v29 activation order, and the rule that any partial
  newest footprint or `downgrade_claim` blocks freeze. State that the freeze
  CLI has no stage override.
- [ ] Run the full warning-strict publication and schema-4/revision suites:

```bash
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_freeze_publication_round.py \
  tests/test_selection_promotion.py \
  tests/test_selection_authority.py \
  tests/test_revision_authority.py \
  tests/test_revision_evaluation.py \
  tests/test_downstream_evidence.py \
  tests/test_maskimpute_v28.py \
  tests/test_trajectory_dataset.py \
  -q -W error -p no:cacheprovider
```

- [ ] Run static verification:

```bash
/home/marcinmaleclocal/.local/bin/ruff check \
  maskimpute_benchmark/publication_freeze.py \
  scripts/freeze_publication_round.py \
  tests/test_freeze_publication_round.py
env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 \
  /tmp/maskimpute-supported/bin/python -m compileall -q \
  maskimpute_benchmark scripts
git diff --check
```

- [ ] Inspect the exact diff and confirm no evidence, generated artifact,
  stage/path override, or unrelated file is present. Commit the runbook update:

```bash
git add docs/development-selection-workflow.md
git commit -m "docs: explain revision-aware publication freeze"
```

- [ ] Use superpowers:requesting-code-review, address findings with focused
  regressions, rerun the complete verification command, and report the exact
  integration range.
