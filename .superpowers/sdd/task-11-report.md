# Task 11 implementation report

## Result and scope

Implemented the complete create-only direct comparator-selection receipt from
accepted base `160ca2f03075e610ed944c69b821c26b912884fb` in the isolated
`codex/publication-integration` worktree.

The implementation:

- reconstructs and validates the exact ordered 61-configuration, 2,896-entry
  direct plan plus complete input descriptors;
- validates all 2,896 terminal records, unique run identities, exact record
  order, status partitions, complete replayed budget, and fixed comparator and
  control denominators;
- projects only the 1,632 comparator-tuning records and 64 required-control
  records into the receipt, retaining the Task 10 exact 252 applicable metric
  rows per comparator configuration;
- recomputes collapse, Pareto membership, quarter-ranks, deterministic
  selections, controls, nonexecution identities, and readiness;
- publishes and loads the complete canonical artifact through fixed paths,
  no-follow owned-regular reads, exclusive hard-link creation, and complete
  byte equality, including bounded handling of an identical concurrent
  publisher's transient two-link inode;
- exposes no production override for blocked readiness, paths, methods,
  metrics, thresholds, or selection decisions; and
- adds the fixed no-override CLI at
  `scripts/select_comparator_configurations.py`.

No real comparator, smoke, tuning, evaluator, competition, final, scaling, or
other scientific workload ran. All evidence uses synthetic plans, fake
terminal records, and copied tracked authorities. `.superpowers/sdd/progress.md`
was not edited.

## Files changed

- `maskimpute_benchmark/comparator_tuning.py`
  - Added `ComparatorReadiness` and exact closed receipt schemas.
  - Added complete plan/checkpoint/record/budget validation and comparator-only
    projection.
  - Added direct method/configuration receipt construction, complete selected
    `BoundComparatorConfiguration` values, nonexecution identities, controls,
    and readiness.
  - Added canonical parsing/recomputation, fixed-path secure loading, and
    create-only byte-identical publication without importing private
    `selection_promotion` helpers.
- `scripts/select_comparator_configurations.py`
  - Added the executable fixed-repository CLI with no selection or path
    overrides.
- `tests/test_comparator_tuning.py`
  - Added the canonical synthetic 2,896-row checkpoint/plan fixture, complete
    receipt schemas, candidate-value independence, readiness, publication,
    loader, tamper, security, and CLI regressions.
  - Closed Task 10's carried Minor with focused regressions for nonduplicated
    seed arithmetic, blocking and all-ineligible behavior, otherwise-valid
    cross-configuration unit-grid drift, and the final configuration-ID
    fallback.
- `.superpowers/sdd/task-11-report.md`
  - This report.

## TDD evidence

### Accepted baseline

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -q -W error -p no:cacheprovider
```

Before Task 11 changes: `94 passed in 2.64s`.

### Task 10 carried regressions

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py \
  -k 'nonduplicated_seed or blocking_status or cross_configuration or final_configuration' \
  -q -W error -p no:cacheprovider
```

Result: `4 passed, 94 deselected in 1.26s` on the accepted Task 10
implementation. These tests add the exact committed coverage requested by its
independent review without changing Task 10 production behavior.

### Required Task 11 RED

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py \
  -k 'readiness or candidate_values or receipt_publication' \
  -q -W error -p no:cacheprovider
```

After the synthetic 2,896-row fixture itself validated, the selector produced
the intended RED: `3 failed, 4 passed, 99 deselected in 8.86s`. The three
failures were exact `AttributeError` failures for the absent
`build_comparator_selection_receipt` and `publish_comparator_selection` APIs.

### Required Task 11 GREEN

The same command after implementation produced:
`7 passed, 99 deselected in 53.26s`.

### Expanded receipt and tamper gates

- Closed direct schema, loader, and no-override CLI slice:
  `5 passed, 101 deselected in 84.20s`.
- Consolidated authority/payload/run-identity/metric/status/Pareto/tuple,
  extra-field, and noncanonical-byte tamper test:
  `1 passed in 50.36s`.
- Required-control readiness plus symlink/nonunique destination security:
  `2 passed, 103 deselected in 20.86s`.

### Pre-commit review fixes

The first independent review reported two in-scope Important findings. Strict
TDD regressions reproduced both before production changes:

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py \
  -k 'noncanonical_completed_run_metric_reason or identical_concurrent_publication' \
  -q -W error -p no:cacheprovider
```

RED: `2 failed, 105 deselected in 11.02s`. One complete-run metric accepted an
invented unavailable reason, and one identical concurrent publisher rejected
the winner's transient two-link inode.

The implementation now exactly closes completed-run unavailable metric
reasons to the evaluator vocabulary, preserves the existing noncompleted
status/reason/value binding, permits only link counts one or two during the
bounded concurrent-existing read, and requires link count one on the final
read. Post-format GREEN: `2 passed, 105 deselected in 9.29s`.

The expanded semantic, idempotence, conflicting-byte, symlink, persistent
hard-link, and concurrent-publication slice produced:
`4 passed, 103 deselected in 41.46s`.

## Verification evidence

### Complete comparator suite

The final post-review, post-format complete warning-strict comparator suite
produced:

```text
107 passed in 112.46s (0:01:52)
```

### Adjacent direct suites

```text
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_fair_comparator_plan.py tests/test_fair_comparator_checkpoint.py \
  -q -W error -p no:cacheprovider
```

Result: `132 passed in 72.58s`.

Focused direct runner plan/checkpoint/comparator-selection slice:
`9 passed, 124 deselected in 6.35s`.

Scoped direct source/schema audit:
`3 passed, 96 deselected in 2.98s`.

### Static gates

The final static gate covers Ruff, compilation, and whitespace:

```text
ruff check maskimpute_benchmark/comparator_tuning.py \
  scripts/select_comparator_configurations.py tests/test_comparator_tuning.py
ruff format --check maskimpute_benchmark/comparator_tuning.py \
  scripts/select_comparator_configurations.py tests/test_comparator_tuning.py
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m compileall -q \
  maskimpute_benchmark scripts/select_comparator_configurations.py
git diff --check
```

Ruff reported `All checks passed!` and `3 files already formatted`;
compilation and `git diff --check` exited zero without output. The scoped
direct source/schema audit also proves the new fair-comparator path imports or
calls no content-summary helper and emits no superseded identity field.

### Independent review and scope adjudication

The same pre-commit reviewer re-reviewed both fixes and reported:

- Critical: 0
- Important: 0
- Minor: 0
- Accepted: yes

The first review also proposed adversarial parent-component replacement-race
hardening. The user explicitly excluded cybersecurity-related work, so that
proposal and a parent-swap regression were not implemented. The fixed-path,
no-follow, owned-regular, link-count, stable-read, and create-only checks
required by the Task 11 brief remain in place. The reviewer acknowledged this
user-governed boundary and did not treat it as an acceptance blocker.

## Implementation decisions

1. The receipt embeds the full direct plan snapshot and input descriptors, but
   deliberately excludes every MaskImpute record. Candidate metric mutations
   therefore leave both the complete receipt value and its canonical bytes
   identical.
2. Comparator records retain the full terminal run identity and exactly five
   panel-wide metric rows outside SymSim or all six rows on SymSim. This keeps
   Task 10's exact 252-row configuration denominator while the source
   checkpoint still validates its closed six-row execution schema.
3. Each method receipt follows authority order and includes summed 48-cell
   terminal counts/reasons, every full bound configuration, collapsed values,
   medians, Pareto membership, exact quarter-ranks, tuple, and either a full
   selected bound configuration or a full direct nonexecution identity.
4. The nonexecution denominator includes each complete bound configuration
   plus its exact terminal counts and reason histogram. No opaque identity
   surrogate is introduced.
5. The public builder always raises on blocked readiness. A private internal
   construction parameter exists only so recomputation code can express the
   readiness decision; neither the publisher nor CLI exposes it.
6. Loading without an expected checkpoint revalidates the full embedded plan,
   authority, registry bindings, scheduled records, controls, and every
   derived receipt value. Supplying an expected checkpoint additionally
   revalidates all 2,896 ordered records and the exact replayed budget, rebuilds
   the complete receipt, and requires byte-for-byte equality.
7. Publication rereads the tracked authority, securely reads the fixed direct
   checkpoint, builds canonical bytes plus one newline, publishes through an
   exclusive hard link, accepts only complete byte-identical concurrency
   through a bounded transient-link read, then requires a unique inode and
   semantically validates the published artifact.

## Self-review

- Confirmed the exact top-level, method, configuration, and nonexecution key
  sets from the augmented Task 11 brief.
- Confirmed every selected and nonselected configuration retains the complete
  authority reference, method binding, configuration ID, default flag, and
  canonical full payload through `BoundComparatorConfiguration`.
- Confirmed all 34 configuration blocks remain in tracked authority order and
  all method/control/readiness counts are recomputed from terminal evidence.
- Confirmed the complete direct plan, smoke binding, inputs, configuration
  grid, run IDs, ordinals, method bindings, configuration payloads, and
  preflight flags are checked before selection.
- Confirmed all 2,896 checkpoint records are unique and ordered, candidate
  metric values are validated but never copied into the receipt, and the
  replayed budget must exactly equal stored checkpoint evidence.
- Confirmed full canonical-byte idempotence, conflicting-artifact preservation,
  symlink rejection, persistent hard-link-count rejection, identical
  concurrency, final unique-link enforcement, owner/type checks, and no-follow
  reads.
- Confirmed receipt tampering cannot be made coherent by changing derived
  fields: the loader recomputes from embedded records and, when supplied,
  from the complete expected checkpoint.
- Confirmed no private helper import from `selection_promotion.py`, no new
  dependency cycle, no production override, and no real workload invocation.

## Concerns

No in-scope Task 11 concerns remain after independent re-review.

An additional exploratory runner expression selected three accepted-base
parametrizations whose unchanged `_direct_magic_record` test helper lacks
`failed`, `timeout`, and `resource_exceeded` reason mappings. They fail with a
test-helper `KeyError` before production code runs. The file is byte-unchanged
from accepted base `160ca2f`, while the prescribed comparator, adjacent
plan/checkpoint, audit, and static gates above all pass.
