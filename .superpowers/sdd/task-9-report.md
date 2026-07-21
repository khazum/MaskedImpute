# Task 9 implementation report

## Scope and base

- Branch/worktree: `codex/publication-integration` at
  `/home/marcinmaleclocal/Coding/MaskedImpute/.worktrees/publication-integration`.
- Required and observed starting HEAD:
  `b5accfd7da4b2953a086a9761ff4218ead9c0de9`.
- Implemented only the fixed truth-free comparator smoke gate and its direct
  production evidence binding.
- No real comparator, smoke, tuning, evaluator, competition, final, scaling,
  or other scientific workload ran. Every execution test used deterministic
  fake outcomes/executors. The real smoke script was never executed.
- `.superpowers/sdd/progress.md` was not modified.

## Files changed

- `maskimpute_benchmark/comparator_tuning.py`
  - Added the exact little-endian float64 900-by-500 fixed input, alternating
    categorical batch column, immutable complete fixture descriptor, measured
    outcome type, 34-row receipt construction, projection/resource/budget
    gates, canonical loader/recomputation, create-only byte publication, and
    fake-injectable spawned-dispatch orchestration.
- `maskimpute_benchmark/runner.py`
  - Requires the fully revalidated receipt before the direct base boundary,
    before storage preflight/output creation, and forwards both its parsed
    value and complete canonical bytes through the private production chain.
- `maskimpute_benchmark/fair_comparator_plan.py`
  - Added immutable full parsed receipt/canonical-byte fields and a validated
    binding helper for the current direct plan implementation.
- `maskimpute_benchmark/fair_comparator_execution.py`
  - Extended the closed direct request with the optional exact smoke-fixture
    descriptor, validates it against the fixed typed input, and serializes the
    complete descriptor only on the smoke path.
- `maskimpute_benchmark/fair_comparator_checkpoint.py`
  - Extended the closed direct plan snapshot so checkpoints carry those exact
    parsed values and bytes through the existing complete plan snapshot.
- `scripts/run_comparator_tuning_smoke.py`
  - Added the executable no-override CLI with the fixed tracked repository and
    concise ready/count receipt output.
- `tests/test_comparator_tuning.py`
  - Added the sole authority-derived smoke fixture factory, exact input and
    projection tests, full loader recomputation, signed-zero/type mutation,
    fake-executor, create-only conflict, and CLI regressions.
- `tests/test_benchmark_runner.py`
  - Added fail-before-storage, evidence-forwarding, and direct
    plan/checkpoint canonical-byte regressions; updated the existing base-route
    fixture for the mandatory gate.
- `tests/test_fair_comparator_plan.py`
  - Extended the exact closed direct plan key assertion for the two receipt
    fields, proves unbound production plans/checkpoints fail before writing,
    and keeps unrelated older structural fixtures on an explicit private
    no-smoke validation seam.
- `.superpowers/sdd/task-9-report.md`
  - This report.

## Design notes

1. The fixed matrix is built from
   `(17*cell + 31*gene + 7*(cell^gene)) % 6`, converted once to C-order `<f8`,
   and paired with fixed cell/gene IDs plus one alternating categorical batch
   column. No truth, evaluator, metric, or score field exists.
2. The shared `MethodInput` retains only its pre-existing internal legacy slot;
   the smoke code never reads, copies, or uses that slot for identity. Fixture
   validation compares the complete permitted typed values directly.
3. Every receipt contains the readable authority revision, all 34 complete
   ordered `BoundComparatorConfiguration` values, the full fixed descriptor,
   seed 42, multiplier 48, every complete measured outcome, sorted projected
   method runtimes, resource provenance, and the fixed discard policy.
4. Receipt readiness requires exactly 34 completed/no-reason outcomes in
   authority order, finite nonnegative non-signed-zero float runtimes,
   nonnegative exact integer peaks, nonempty provenance, the 48 GiB/14 GiB
   caps, and projected 24-hour CPU or 8-hour GPU method budgets.
5. Loading re-reads and fully parses the tracked authority, rebinds every method
   and configuration, parses closed canonical JSON with duplicate/nonfinite
   rejection, reconstructs typed outcomes, recomputes the fixture and every
   derived field, and compares direct values plus the complete canonical bytes.
6. Publication uses a temporary owned file plus create-only hard link. An
   existing or concurrently appearing receipt is accepted only when its entire
   canonical bytes are identical; conflicts remain untouched and fail closed.
7. The sole `_executor` keyword seam receives a private complete request
   containing the bound configuration, complete descriptor, fixed input,
   registry method, seed 42, and ordinal. The default converts that boundary to
   a `DirectExecutionRequest` which retains, revalidates, serializes, and sends
   the same complete descriptor through the measured spawned dispatcher. Tests
   replace the actual process call with deterministic resource-only fake
   outcomes, so no adapter output or raw stream is retained or evaluated.
8. The direct plan stores the parsed receipt as frozen direct values and the
   exact receipt bytes as integer byte values in JSON. This preserves exact
   bytes without encoding them as a content summary or introducing a string
   value that could shadow forbidden lexical tokens. Checkpoints inherit the
   same complete fields through `plan_snapshot`.
9. The base and both revision production entries load smoke evidence before
   legacy panel preparation and before entering the direct segment. They pass
   already-prepared typed values through private arguments; the direct boundary
   constructs the canonical plan with the exact parsed receipt and bytes, then
   performs read-only `DirectCheckpointStore.inspect_prefix` validation. The
   current command still stops at the pre-existing production-adapter
   composition placeholder, so this task does not execute a workload or create
   a checkpoint. If a checkpoint already exists, its complete plan snapshot and
   smoke bytes must match. Pure structural tests use the explicit private
   no-smoke validator; public production plan/checkpoint validation rejects an
   empty receipt, and the production builder requires both values.

## TDD RED evidence

Initial required smoke API selector:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_comparator_tuning.py -k 'smoke_input or smoke_receipt' -q -W error -p no:cacheprovider
```

Output: collection failed with one expected `ImportError` for the absent
`ComparatorSmokeOutcome` API (`1 error in 0.74s`).

Expanded loader/orchestration/CLI selector:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_comparator_tuning.py -k 'smoke_input or smoke_receipt or smoke_run or smoke_loader or smoke_cli' -q -W error -p no:cacheprovider
```

Output: collection failed with one expected `ImportError` for the absent
`load_comparator_smoke_receipt` boundary (`1 error in 0.79s`).

Runner ordering/evidence selector:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_benchmark_runner.py -k 'scientific_execution_requires_smoke or forwards_complete_smoke_evidence' -q -W error -p no:cacheprovider
```

Output: `2 failed, 127 deselected in 2.69s`. The base entry reached the Task 8
placeholder instead of a smoke error and omitted both receipt arguments.

Direct plan/checkpoint evidence selector:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_benchmark_runner.py -k 'carry_complete_smoke_receipt_bytes' -q -W error -p no:cacheprovider
```

Output: collection failed with the expected missing
`bind_comparator_smoke_receipt_to_plan` import (`1 error in 2.24s`).

Executor-error normalization regression:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_comparator_tuning.py::test_smoke_run_uses_all_bound_rows_and_create_only_complete_bytes -q -W error -p no:cacheprovider
```

Output: `1 failed in 1.19s`; the injected private `RuntimeError` escaped instead
of becoming the closed comparator smoke boundary error.

Signed-negative-zero regression:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_comparator_tuning.py::test_smoke_loader_recomputes_complete_receipt -q -W error -p no:cacheprovider
```

Output: `1 failed in 0.74s`; the receipt failed only at later recomputed-value
equality instead of the required measurement validation, proving the signed
zero primitive needed explicit rejection.

Independent-review descriptor and production-carrier regressions:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_comparator_tuning.py::test_spawned_smoke_request_retains_complete_fixed_fixture_descriptor tests/test_fair_comparator_plan.py::test_production_plan_and_checkpoint_reject_unbound_smoke_evidence tests/test_benchmark_runner.py::test_direct_production_boundary_binds_smoke_into_plan_and_checkpoint_chain tests/test_benchmark_runner.py::test_revision_runners_inherit_complete_smoke_evidence
```

Output: `5 failed in 5.24s`. The inner request lacked `smoke_fixture`, public
production plan/checkpoint validation accepted an unbound receipt state, the
base direct boundary never constructed/inspected a plan, and both revision
runners still entered the legacy path.

## GREEN and adjacent verification

- Initial exact fixture/receipt selector: `2 passed, 71 deselected in 0.70s`.
- Expanded smoke/loader/run/CLI selector after implementation:
  `5 passed, 71 deselected in 1.25s`.
- Runner ordering/evidence selector: `2 passed, 127 deselected in 2.18s`.
- Plan/checkpoint receipt-byte selector: `1 passed, 129 deselected in 2.25s`.
- Executor normalization test: `1 passed in 1.10s`.
- Signed-zero loader test: `1 passed in 0.86s`.
- Required final Task 9 selector after review fixes:
  `21 passed, 189 deselected in 2.67s`.
- Complete comparator authority plus direct checkpoint suites after review
  fixes: `170 passed in 3.24s`.
- Complete direct plan suite after review fixes: `38 passed in 68.13s`.
- Focused direct-request/descriptor/constructor slice after review fixes:
  `33 passed, 19 deselected in 2.12s`.
- Base/revision binding plus public storage-plan slice after review fixes:
  `6 passed, 127 deselected in 3.39s`.
- Adjacent base route/smoke runner slice:
  `4 passed, 126 deselected in 2.25s`.
- Scoped direct source/schema/value audits after review fixes:
  `3 passed in 2.07s`.
- Ruff over every changed Python file: `All checks passed!`.
- `env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m compileall -q maskimpute_benchmark scripts/run_comparator_tuning_smoke.py`:
  exit 0, no output.
- `git diff --check`: exit 0, no output.
- Focused fair-comparator lexical scan found no forbidden occurrence except the
  approved descriptor field `shape`; there were no evaluator/checkpoint calls
  on the smoke path.

An intentionally over-broad initial baseline command was cancelled by the
parent after more than twelve minutes of CPU-active pre-existing 2,896-row
runner coverage. It had reached `144 passed in 766.59s` with no test failure.
This cancelled run is recorded only as context and is not acceptance evidence;
the focused fresh commands above are the acceptance evidence.

## Self-review and concerns

- The implementation follows the current accepted Task 8 direct
  storage/evidence owners rather than recreating superseded legacy runner
  summary fields. The two direct-module edits are therefore intentional
  deviations from the older file-location list.
- Static and dynamic review confirmed no real workload entrypoint was invoked,
  no content-summary helper/field was added to the fair-comparator path, and
  existing legacy provenance outside the direct segment was not changed.
- The first independent review rejected two Important defects: the complete
  descriptor was dropped at the inner spawned request, and receipt evidence
  was forward-only/bypassable rather than bound to the real base/revision
  direct plan/checkpoint chain. Both received focused RED regressions and the
  fixes described above. Independent re-review accepted the result with no
  Critical or Important findings and independently reran the required focused
  selector (`21 passed, 189 deselected`).
- Concerns: none at implementation-report time beyond the factually recorded,
  intentionally cancelled broad baseline, which is not used as verification.

## Independent review fixes

The exact-range independent review in `.superpowers/sdd/task-9-review.md`
reported two Important and two Minor findings. No real comparator, smoke,
evaluator, competition, tuning, final, or other scientific workload was run
while resolving them.

### Focused RED and baseline evidence

Signed-zero descriptor and inner-request chain:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_comparator_tuning.py::test_smoke_input_rejects_byte_distinct_negative_formula_zero tests/test_comparator_tuning.py::test_spawned_smoke_request_retains_complete_fixed_fixture_descriptor
```

Output: `2 failed in 0.77s`. Both expected exceptions were absent because
`np.array_equal` treated the formula's canonical `+0.0` and the mutated
little-endian `-0.0` bit pattern as equal.

Receipt-required exported builder:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_fair_comparator_plan.py::test_public_direct_builder_requires_and_validates_complete_smoke_evidence
```

Initial output: `1 failed in 4.04s`; the public builder returned an unbound
plan instead of requiring receipt arguments. After making the public arguments
mandatory, the strengthened no-default assertion was separately observed RED
with the same command: `1 failed in 3.11s`, because the dataclass still
silently defaulted `comparator_smoke_receipt` to `()`.

Complete resource and projected-budget threshold regressions:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_comparator_tuning.py::test_smoke_receipt_rejects_each_resource_cap_with_complete_denominator tests/test_comparator_tuning.py::test_smoke_receipt_rejects_projected_method_budget_with_complete_denominator
```

Baseline output: `4 passed in 0.69s`. This Minor finding was a coverage gap;
the existing 48 GiB RSS, 14 GiB GPU, 24-hour CPU, and 8-hour GPU method gates
already rejected complete 34-row outcome sets at the required thresholds.

Deterministic missing-smoke ordering regression:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_benchmark_runner.py::test_scientific_execution_requires_smoke_before_storage_or_output
```

Baseline output: `1 passed in 2.01s`. The test now injects the loader failure
and spies that panel preparation, storage preflight, and the output/checkpoint
boundary remain untouched, independent of checkout artifact state.

### Changes

- `comparator_smoke_input_descriptor` now compares the complete immutable
  canonical `<f8` count bytes directly. The regression independently asserts
  the exact little-endian `+0.0` (`00 00 00 00 00 00 00 00`) and `-0.0`
  (`00 00 00 00 00 00 00 80`) encodings before proving rejection at both the
  descriptor and `DirectExecutionRequest` boundaries.
- Receipt-free full-grid construction is named and confined to
  `_build_structural_direct_competition_plan`. The exported
  `build_direct_competition_plan` requires the parsed receipt and exact bytes,
  binds them through the canonical loader, and strictly validates the complete
  production plan before returning. The plan dataclass no longer supplies
  silent empty receipt defaults, and structural fixtures state empty evidence
  explicitly.
- Resource-cap coverage now exercises both RSS and GPU caps, while projected
  budget coverage exercises CPU and GPU methods with all 34 ordered outcomes
  present and completed.
- The missing-smoke runner test no longer depends on whether a receipt happens
  to exist in the checkout.

### Fresh GREEN verification

The two focused commands from the RED section were rerun first. Signed-zero
descriptor/inner-request output: `2 passed in 0.94s`. Final
public-builder/no-default output: `1 passed in 4.96s`.

Required Task 9 selector:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_comparator_tuning.py tests/test_benchmark_runner.py -k 'smoke or scientific_execution_requires_smoke'
```

Output: `26 passed, 189 deselected in 2.81s`.

Complete comparator authority plus direct checkpoint suites:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_comparator_tuning.py tests/test_fair_comparator_checkpoint.py
```

Output: `175 passed in 3.40s`.

Complete direct plan suite:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_fair_comparator_plan.py
```

Output: `39 passed in 70.55s`.

Focused direct-request/descriptor/constructor slice:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_fair_comparator_execution.py -k 'request or descriptor or constructor'
```

Output: `33 passed, 19 deselected in 2.02s`.

Base/revision binding plus public storage-plan runner slice:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_benchmark_runner.py -k 'development_base_entry_routes or development_base_forwards or direct_production_boundary_binds or revision_runners_inherit or development_storage_fixture_passes_public'
```

Output: `6 passed, 127 deselected in 4.42s`.

Scoped direct source/schema/value audits:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest -q -W error -p no:cacheprovider tests/test_selection_authority.py::test_scoped_direct_source_and_schema_migration_audit tests/test_selection_authority.py::test_scoped_direct_schema_audit_covers_every_synthetic_artifact tests/test_selection_authority.py::test_direct_closed_metadata_vocabularies_are_forbidden_token_free
```

Output: `3 passed in 2.29s`.

Static checks:

```text
ruff check maskimpute_benchmark/comparator_tuning.py maskimpute_benchmark/fair_comparator_plan.py maskimpute_benchmark/runner.py tests/test_benchmark_runner.py tests/test_comparator_tuning.py tests/test_development_evaluation.py tests/test_fair_comparator_checkpoint.py tests/test_fair_comparator_plan.py tests/test_selection_authority.py
```

Output: `All checks passed!`.

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m compileall -q maskimpute_benchmark tests scripts/run_comparator_tuning_smoke.py
```

Output: exit 0, no output.

```text
git diff --check
```

Output: exit 0, no output.
