# Task 15 implementation report

## Result and scope

Implemented exact seed-preserving final and registered-trajectory comparator
plans from accepted Task 14 base
`a04679f57bda9fb5f6654598b8257b93e469b27c` in the isolated
`codex/publication-integration` worktree.

The implementation:

- adds the exclusive `FrozenPlanMethodAuthority` union for legacy
  configurations, complete selected `BoundComparatorConfiguration` values, or
  complete comparator nonexecution identities;
- recomputes the complete Task 14 comparator-selection projection, validates
  the selected/nonexecution maps as a disjoint ten-method partition, binds
  every selected method projection field by field to the current registry, and
  rejects selected-map, overlap, status, payload, or registry drift;
- removes registry-default fallback for all ten tuning-authority methods while
  preserving the existing legacy configuration path for every other method;
- preserves all three seeds for unavailable scheduled stochastic comparators,
  giving 120 final or three trajectory nonexecution rows without changing the
  structural denominator;
- enforces the production 1,760-row final and 44-row trajectory denominators,
  including the all-selectable final split of 1,480 executable and 280
  structural nonexecution rows;
- uses readable final and trajectory run IDs containing ordinal, method,
  dataset, seed, and configuration ID rather than deriving comparator identity
  from a content summary;
- adds the narrow `FinalComparatorExecutionRequest`, validates the complete
  frozen configuration, authority reference, registry method projection,
  decoded/re-encoded payload, seed policy, and exact method resource limits,
  and routes it through direct repository comparator adapters without the
  legacy `ExecutionRequest` comparator path;
- retains the complete selected or nonexecution identity across plan entries,
  run records, metric rows, execution-request receipts, immutable store replay,
  and evaluation replay; and
- rejects old or mixed comparator summary fields while leaving unrelated
  legacy request, record, calibration, dataset, final, and outer publication
  provenance behavior unchanged.

No comparator, smoke, tuning, evaluator, competition, final, scaling, or other
real scientific workload ran. Verification used only synthetic fixtures and
fake executors. The progress ledger was not edited.

## Exact denominator and identity behavior

For the production registry, final planning now yields:

- 1,760 total rows over 40 final datasets;
- 1,480 executable rows and 280 `not_applicable` rows when every selectable
  comparator has a chosen configuration; and
- exactly 120 rows with seeds 42, 43, and 44 when one scheduled stochastic
  comparator is unavailable, changing only action, reason, configuration kind,
  and complete typed nonexecution identity.

Registered trajectory planning always yields 44 rows. The same unavailable
stochastic comparator retains its three seed rows. Selected comparator entries
carry the full configuration payload, upstream-default marker, readable
authority reference, and complete method binding. Nonexecution rows contain no
selected payload or registry default.

Direct comparator run and metric artifacts omit the superseded
`configuration_sha256` field and instead store the complete bound
configuration or complete nonexecution identity. Attempted comparator terminal
outcomes retain their complete direct request receipt. Structural
`not_applicable` rows have no request receipt.

## TDD evidence

Production changes began from focused failing regressions. Observed RED cases
included:

- selected comparators reaching planning without `legacy_configuration`;
- unavailable stochastic methods collapsing the final denominator to 1,680
  rows and the trajectory denominator to 42 rows;
- exact nonexecution values changing nested container identity at the
  `RunPlanEntry` boundary;
- final/trajectory execution passing selected comparator authority to the
  legacy `ExecutionRequest`, which rejected it;
- the final result store accepting only legacy execution-request receipts;
- complete bound configurations losing their computed payload during generic
  dataclass serialization;
- direct request resource ceilings being replaceable without rejection; and
- scaling's reflected legacy schema widening when the shared run/metric types
  gained optional direct-only fields.

Focused GREEN evidence after the corresponding changes included:

```text
exact final/trajectory cardinality and unavailable-seed tests:
  3 passed, 102 deselected in 61.75s
selected direct final comparator request/dispatch:
  1 passed in 30.01s
selected payload, selected/nonexecution overlap, and status tamper matrix
plus direct dispatch:
  4 passed, 105 deselected in 48.32s
trajectory dispatch/store/replay with attempted-vs-structural request receipts:
  1 passed in 124.14s
legacy scaling serialization compatibility:
  2 passed, 42 deselected in 8.23s
stale runner summary-only schemas fail closed:
  2 passed, 143 deselected in 2.32s
post-format direct/tamper/legacy/scaling boundary subset:
  8 passed, 290 deselected in 54.09s
```

The migrated tests no longer request explicit comparator payload checksums or
digest-only comparator identities. They assert that those superseded shapes
fail closed as content summaries.

## Verification evidence

Required complete warning-strict final-runner suite:

```text
tests/test_final_runner.py:
  109 passed in 623.41s (0:10:23)
```

Complete adjacent runner and scaling suites:

```text
tests/test_benchmark_runner.py:
  145 passed in 1259.34s (0:20:59)
tests/test_scaling_panel.py:
  44 passed in 241.46s (0:04:01)
```

Final static gates:

```text
ruff format --check [five touched source/test files]: 5 files already formatted
ruff check [five touched source/test files]: All checks passed!
python -m compileall -q maskimpute_benchmark: exit 0
git diff --check: exit 0
```

The supported test interpreter did not package Ruff, so the tracked system
Ruff executable at `/home/marcinmaleclocal/.local/bin/ruff` ran the lint gate.
No packages or environments were changed.

## Minimal scaling compatibility

Task 15 adds two optional direct-only fields to the shared `RawRunResult` and
`LongFormMetric` types. Scaling previously serialized `RawRunResult` with
`dataclasses.asdict` and derived its closed legacy field set by reflecting all
dataclass fields. That would have added two null comparator fields to legacy
scaling artifacts and changed their accepted schema.

The narrowly scoped compatibility adjustment makes scaling use
`RawRunResult.to_dict()` and excludes the two direct-only names from its
reflected legacy expected field sets. It does not change scaling planning,
identity, defaults, scientific execution, records, or Task 16 behavior. The
complete 44-test scaling suite proves the pre-existing legacy behavior remains
intact.

## Implementation decisions

1. The complete validated Task 14 comparator selection remains the sole
   selected/nonexecution authority. Final planning never reconstructs a tuning
   method from registry defaults.
2. Legacy and direct configurations remain distinct types. A direct comparator
   never enters `ExecutionRequest`; a nonexecution identity is never
   dispatchable.
3. Direct request integrity is checked before and after dispatch. Effective
   typed adapter configuration must re-encode to the exact frozen payload.
4. Complete direct configuration values are repeated in run and metric rows so
   immutable store and evaluation replay can compare actual typed values rather
   than a summary token.
5. Existing outer final-runner provenance remains unchanged. The migration
   removes only superseded comparator-configuration summary mechanisms from the
   direct comparator segment.

## Concerns

No blocking Task 15 concern remains. Complete final-runner, runner, and scaling
suites pass warning-strict; lint, compilation, and diff checks pass. Independent
acceptance of the exact Task 15 commit remains the next workflow step. Task 16
retains responsibility for its downstream/scaling publication-schema work; no
Task 16 identity or default migration was implemented early.

## Task 15 review correction

The independent review of `870ac99f4b7921f3e8eab82ff57bb44992a3b90a`
identified one critical disposition-coupling defect and one important shallow
immutability defect. Both are corrected in this follow-up change.

### Disposition coupling and production split

Selected direct comparator authority is now coupled to the exact executable
disposition before `FrozenPlanMethodAuthority` construction and rechecked by
that frozen value's invariant. Complete comparator nonexecution authority is
likewise coupled to a reason-coded `not_applicable` disposition. A selected
configuration can therefore no longer be combined with a duplicate
`unavailable`/`never` method-denominator row.

The full all-selected production final plan now requires all three binding
counts: 1,760 total, 1,480 executable, and 280 not-applicable rows. This split
gate is deliberately limited to the full production registry when every
selectable comparator has a selected configuration. Subset/synthetic
registries remain ungated, and an explicitly unavailable stochastic comparator
continues to retain all three seeds in both final and trajectory plans.

### Recursive nonexecution immutability

`RunPlanEntry` now reconstructs its copied nonexecution identity with the
existing `freeze_direct_mapping` representation rather than thawing nested
objects and lists and wrapping only the outer mapping. Serialization remains
the only thaw point. Caller-owned nested dict/list mutation cannot alter the
entry, and serialized identities reconstruct to the same stable direct value.

The adjacent `direct_values.py` adjustment is intentionally narrow:
`direct_equal` recognizes `FrozenDirectObject` only as the immutable equivalent
of a JSON object and `FrozenDirectList` only as the immutable equivalent of a
JSON list. Ordinary tuple/list or list/object coercion remains forbidden. The
existing nested-list-of-pairs and list/object-collision regressions were run to
verify that exact container identity remains closed.

No comparator content digest was added or recomputed, and legacy plan checksum
or other outer provenance behavior was not expanded. Scaling code and Task 16
behavior were not changed. No scientific workload ran, and the progress ledger
was not edited.

### Correction TDD evidence

Supported-interpreter RED before production changes:

```text
selected-to-nonrun disposition drift, missing production split gate,
and nested dict/list immutability:
  3 failed, 254 deselected in 37.46s
```

Focused GREEN after the corrections and formatting:

```text
coupling, production split, selected happy path, stochastic final/trajectory
nonexecution seeds, and recursive immutable replay:
  6 passed, 251 deselected in 80.79s

adjacent direct-value nested-list-of-pairs and list/object-collision contracts:
  4 passed, 235 deselected in 2.11s
```

### Correction verification evidence

The required exact warning-strict commands used the bound supported
interpreter and disabled pytest's cache provider:

```text
tests/test_final_runner.py:
  111 passed in 616.08s (0:10:16)

tests/test_benchmark_runner.py:
  146 passed in 1245.79s (0:20:45)
```
