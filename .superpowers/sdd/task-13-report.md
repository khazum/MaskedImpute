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
- executes revision plans through the public authority-validated direct runner
  and `DirectCheckpointStore`, with the existing MaskImpute adapter translated
  into the closed direct result schema;
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

## Implementation decisions

1. Revision authorities remain candidate-only. Base comparator evidence is an
   immutable authority input and is never rescheduled or copied into a revision
   checkpoint.
2. The direct revision executor delegates only the in-tree MaskImpute numerical
   adapter, then closes the result into the public direct schema. Structural
   test execution remains separate from the production activation gate.
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
rerun in full after the test-helper-only correction; its three previously
failing cases were rerun and passed, and the required cross-stage acceptance
command plus all Task 13 focused paths are green. No scientific production run
was authorized or performed.
