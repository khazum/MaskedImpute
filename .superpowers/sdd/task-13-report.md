# Task 13 implementation report

## Result and scope

Implemented candidate-only v28/v29 revision execution and exact reuse of base
comparator evidence from accepted base
`2d4b76dbecd39cd3a8b42e014db3e4a59df965e6` in the isolated
`codex/publication-integration` worktree.

The implementation:

- validates activated revision authority before production planning and carries
  the immutable complete five-field base comparator-selection object through
  `RevisionActivation`, `RunnerAuthority`, evaluation, and freeze-stage replay;
- builds and validates one direct 48-row MaskImpute-only plan for each revision,
  excluding observed, capacity-control, registry-default, and comparator rows;
- executes revision plans through the public authority-validated direct runner,
  a direct MaskImpute numerical adapter, and `DirectCheckpointStore`, without
  constructing the superseded request/runtime-registry/evaluator bridge;
- requires exact calibration evidence, projects exactly six direct evaluation
  metrics, and retains bounded output/log receipts without comparator content
  summaries;
- loads revision evidence only from a complete direct 48-row candidate
  checkpoint and retains its complete ordered plan snapshot and prepared-input
  descriptors;
- reuses every inherited base record and interval unchanged and in its original
  order, while revision checkpoints contain only the new candidate rows;
- keeps the pre-existing outer legacy revision provenance fields while avoiding
  summary fields inside the direct plan, checkpoint, and comparator segment;
  and
- extends schema-4 freeze-stage replay to preserve the complete comparator
  selection and reject any cross-stage or activation mismatch by direct value
  equality.

No comparator, smoke, tuning, evaluator, competition, final, scaling, or other
real scientific workload ran. All execution evidence used by tests was
synthetic. The progress ledger was not edited.

## Independent-review fix

The independent review found that the first implementation produced a direct
record only after traversing the legacy request, runtime registry, spawned
dispatcher, and evaluator bridge. The fix replaces that reached production
path completely:

- `DirectRevisionMaskImputeAdapter` consumes the exact
  `ComparatorRunIdentity`, `PreparedInputDescriptor`, full
  `DirectAuthorizedConfiguration`, `MethodSpec`, `MethodInput`, and timeout;
- every supplied value is revalidated directly against the activated candidate
  and prepared numerical input before the scientific fit begins;
- `run_revision_maskimpute_direct` reuses the existing MaskImpute fit core but
  returns only the direct native matrix, aligned `p_pre_zero`, and raw bounded
  streams;
- direct evaluator conversion produces the exact six reconstruction metrics,
  matrices, log receipts, and create-only compressed `p_pre_zero` evidence;
  and
- production revision composition no longer constructs an
  `ExecutionEnvironmentRegistry`, `RepositoryAdapterDispatcher`,
  `SpawnedRepositoryExecutor`, legacy `ExecutionRequest`, or legacy evaluated
  outcome.

The completed regression is synthetic but unmocked at the public execution and
checkpoint layers. It executes all 48 planned revision attempts through
`execute_fair_comparator_plan` and a real `DirectCheckpointStore`, asserts the
exact six-metric order and numerical values, checks native/evaluator matrices
and bounded log receipts, reopens the compressed `p_pre_zero` matrix, and
replays the same run IDs to prove create-only byte equality. No real scientific
workload ran.

## Production resource-envelope review fix

The second independent review found that the real production revision adapter
accepted an authorized deadline but executed its numerical fit in-process,
did not enforce either method resource ceiling, and published zero runtime and
resource peaks for successful fits. The correction now:

- constructs a complete `DirectRevisionExecutionRequest` containing the exact
  direct identity, prepared-input descriptor, authorized configuration,
  `MethodSpec`, `MethodInput`, decision timeout, RSS ceiling, and GPU-memory
  ceiling;
- revalidates those complete values and exact byte ceilings both before spawn
  and inside the child process;
- reuses the accepted parent-owned spawned-process monitor to enforce the
  deadline and process-tree RSS/GPU limits and to classify timeout, resource,
  infrastructure, and executor failures into the existing direct statuses;
- reconstructs immutable calibration evidence inside the child from its full
  payload, avoiding an unpicklable object without introducing a content
  summary or legacy runtime registry;
- returns a successful child numerical result without placeholder resource
  values, then replaces it at the parent boundary with measured nonzero elapsed
  time, measured RSS, and measured GPU memory (or zero from telemetry when the
  process uses no GPU); and
- persists those measured values through all 48 public direct attempts so
  `DirectDevelopmentBudget` consumes the actual elapsed time and restores the
  same consumption from checkpoint replay.

The production-adapter regressions keep only the numerical fit synthetic. They
exercise the real `DirectRevisionMaskImputeAdapter`, real spawn boundary,
parent sampling/enforcement, public plan executor, and real checkpoint store.
No legacy request/evaluator/runtime-registry bridge or content-summary field
was restored.

## TDD evidence

Production changes were introduced from focused failing regressions. Observed
RED cases included:

- incomplete revision activation accepted without the five-field comparator
  object;
- activated runner construction failing before it could carry that object;
- the revision runner lacking its public direct-executor seam and production
  adapter composition;
- terminal budget and authorized MaskImpute direct execution paths being
  absent;
- revision evaluation rejecting direct reconstruction evidence and retaining
  legacy checkpoint calls;
- inherited-row and cross-authority direct-equality helpers being absent; and
- freeze-stage schema fixtures and replay omitting the now-required comparator
  selection.

Each focused regression passed after its production boundary was added. The
final freeze-stage RED was four failures on the missing schema-4 field/equality
boundary; its focused GREEN was:

```text
12 passed, 106 deselected in 1.32s
```

For the independent-review fix, the completed direct regression was first RED
because `DirectMaskImputeExecution` did not exist. The scoped source audit also
reproduced the review finding before implementation (`1 failed, 1 passed`).
After the direct boundary was implemented, the completed, unavailable, budget,
and production-composition cases all passed, and both scoped audits passed.

For the production resource-envelope review fix, the initial focused run was
RED with eight failures: the revision executor omitted both resource ceilings,
the production adapter had no spawned measured entry point, and the 48-row
regression still substituted the adapter above the production boundary. After
the correction, focused regressions prove exact timeout/RSS/GPU forwarding,
CPU and GPU parent-measured completion, timeout classification, both resource
ceiling classifications, and measured budget replay through the real 48-row
checkpoint.

## Verification evidence

Required warning-strict revision, activation, promotion, and freeze-stage
acceptance command:

```text
53 passed, 113 deselected in 190.18s (0:03:10)
```

Adjacent complete suites and focused direct boundaries run during development:

```text
tests/test_revision_authority.py: 22 passed in 105.16s
tests/test_revision_evaluation.py: 10 passed in 91.00s
tests/test_fair_comparator_plan.py: 37 passed in 51.82s
tests/test_fair_comparator_checkpoint.py: 93 passed in 3.77s
development/downstream direct audits: 9 passed, 104 deselected in 371.50s
required candidate-only slice: 2 passed, 168 deselected in 21.34s
direct revision-evaluation paths: 5 passed, 145 deselected in 28.63s
runner Task 13 paths: 5 passed, 135 deselected in 2.35s
activation plus production gate: 2 passed in 37.73s
```

The broad benchmark-runner run reached `135 passed` and exposed three failures
in a test-local synthetic direct-result helper whose terminal reason map was
incomplete. After completing that helper for all intrinsic terminal statuses,
the affected parametrization passed:

```text
4 passed in 2.21s
```

Static gates after the final source edit:

```text
ruff format: 1 file reformatted, 10 files unchanged
ruff check: All checks passed!
python -m compileall -q [touched production modules]: exit 0
git diff --check: exit 0
```

Final production resource-envelope evidence, rerun after removing successful
child placeholder measurements, is:

```text
focused production adapter and 48-row checkpoint replay:
  11 passed, 134 deselected in 160.82s
required revision/activation/base-comparator cross-stage suite:
  53 passed, 113 deselected in 186.72s
adjacent MaskImpute adapter/v28/v29 suites: 34 passed in 3.54s
accepted direct spawned-executor adjacency: 3 passed, 49 deselected in 4.09s
checkpoint budget adjacency: 8 passed, 85 deselected in 1.86s
scoped direct source/schema audits: 2 passed in 2.86s
ruff format --check: 2 files already formatted
ruff check: All checks passed!
python -m compileall -q [touched source and test]: exit 0
```

Final independent-review-fix evidence, all under the required supported
interpreter with warnings treated as errors, is:

```text
post-format direct execution plus scoped audits: 7 passed, 233 deselected in 36.61s
required revision/activation/base-comparator cross-stage suite: 53 passed, 113 deselected in 188.25s
adjacent MaskImpute adapter/v28/v29 suites: 34 passed in 4.30s
ruff format: 4 files reformatted, 1 unchanged
ruff check: All checks passed!
python -m compileall -q [touched production modules]: exit 0
git diff --check: exit 0
```

The adjacent v28 suite exposed one stale test that still invoked the superseded
legacy full-denominator planner for revision authority. That test now checks
the accepted v28/v29 candidate-only authority and activation contract directly;
no production behavior was relaxed.

## Implementation decisions

1. Revision authorities remain candidate-only. Base comparator evidence is an
   immutable authority input and is never rescheduled or copied into a revision
   checkpoint.
2. The direct revision executor passes complete typed values to the in-tree
   MaskImpute numerical adapter, independently evaluates the returned matrices,
   and closes them directly into the public schema. Structural test execution
   remains separate from the production activation gate.
3. Complete comparator-selection values are normalized through the shared
   direct-value codec before freezing. This keeps nested object/array identity
   stable across construction, replacement, serialization, and equality.
4. Direct revision checkpoints contain exactly 48 records for one candidate.
   Combined selection evidence appends those rows to a directly equal base
   prefix and explicitly rejects changed or reordered inherited evidence.
5. Existing legacy dataset, calibration, revision-manifest, and publication
   provenance remains in its established outer envelope. Direct evidence uses
   full typed identities, plan snapshots, prepared descriptors, receipt bytes,
   and byte-equal artifacts rather than comparator content summaries.

## Concerns

No blocking Task 13 concern remains. The broad benchmark-runner suite was not
rerun in full; the complete 11-test affected slice includes the real
production adapter, all resource terminal paths, public 48-row execution, and
checkpoint/budget replay. The required cross-stage acceptance command, both
scoped audits, the accepted spawn adjacency, checkpoint budget adjacency, and
the adjacent MaskImpute suites pass warning-strict. No scientific production
run was authorized or performed.
