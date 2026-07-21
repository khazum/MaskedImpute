# Task 12 implementation report

## Result and scope

Implemented selected-only fair-comparator projection from accepted base
`fcfdb30c68b6ef5b15a6733ca461fb638457646c` in the isolated
`codex/publication-integration` worktree.

The implementation:

- defines immutable `SelectedComparatorConfiguration` and
  `ComparatorSelectionProjection` values carrying the full canonical Task 11
  receipt bytes/value, complete selected bound configurations, complete
  nonexecution identities, scheduled methods, and the exact ready population;
- semantically recomputes the complete 2,896-row receipt before projection and
  uses a process-local cache keyed by complete receipt bytes plus directly equal
  authority and method values, never by a content summary;
- maps only an exactly selected comparator run identity into development
  selection rows and leaves all nonselected configurations solely in the
  checkpoint and receipt;
- embeds the complete comparator-selection object in schema-2 development
  inputs, evaluation manifests, schema-3 revision evidence, and promoted
  schema-4 inputs;
- rereads the fixed receipt after development publication and promotion and
  requires byte-for-byte equality with the embedded full value;
- derives dynamic selection declarations from the exact ready population,
  removes stale comparator registry bindings, carries complete selected and
  nonexecution maps, and fails on missing required population rows; and
- keeps intrinsically unavailable methods scheduled in full receipt evidence
  while excluding them from declarations, numerical bindings, and the selected
  map.

No real comparator, smoke, tuning, evaluator, competition, final, scaling, or
other scientific workload ran. All new evidence is synthetic. The progress
ledger was not edited.

## TDD evidence

Accepted baseline:

```text
174 passed in 18.15s
```

Required focused RED after adding the selected-only reconstruction and exact
ready-population regressions:

```text
2 failed, 62 deselected
```

The failures were the absent `comparator_selection_projection` API and absent
`SelectionReport.comparison_population_ids` field.

Required focused GREEN:

```text
2 passed, 62 deselected in 21.28s
```

Dynamic-authority, intrinsic-unavailability, and scoped direct-schema slice:

```text
4 passed, 97 deselected in 35.39s
```

## Verification evidence

Required warning-strict development, manifest, candidate-selection, and
promotion suite:

```text
178 passed in 311.81s (0:05:11)
```

Complete adjacent comparator receipt suite:

```text
114 passed in 221.71s (0:03:41)
```

Revision-evaluation propagation suite:

```text
4 passed in 2.04s
```

Scoped direct source/schema audits:

```text
2 passed, 99 deselected in 3.02s
```

Static gates:

```text
ruff check ...
All checks passed!

ruff format --check ...
15 files already formatted

python -m compileall -q maskimpute_benchmark scripts tests
git diff --check
```

Compilation and whitespace checks exited zero without output.

## Implementation decisions

1. Selected comparator rows set the pre-existing legacy method-binding field
   to null and carry the complete run identity plus complete bound selected
   configuration in dedicated fields. Candidate and control rows retain their
   unchanged legacy bindings.
2. The dynamic authority preserves only candidate and ready comparison
   declarations. Required controls retain their pre-existing legacy bindings;
   selected comparators are identified exclusively by the complete direct map.
3. Receipt validation caching compares complete canonical bytes, the full typed
   comparator authority, and every full direct method binding. It does not
   compute, store, or compare an identity surrogate.
4. Promotion retains the embedded comparator object unchanged and rereads the
   fixed receipt before and after publication. Semantic recomputation is also
   performed by the schema-2 evaluation and selection consumers.
5. Existing legacy dataset, calibration, revision, downstream, and publication
   provenance remains unchanged outside the direct comparator segment.

## Concerns

No blocking Task 12 concern remains. The production scientific build path was
not executed, as expressly prohibited; its fixed comparator receipt is loaded
with the exact checkpoint before reconstruction evidence is accepted.

## Controller review fixes

### Findings resolved

The independent review's Critical production-route finding is fixed. The
schema-2 builder now loads the fixed Task 11 receipt against the exact base
checkpoint, reconstructs the complete `direct-v1` fair-comparator plan from its
embedded smoke receipt, validates all 2,896 terminal rows through
`DirectCheckpointStore`, and performs one explicit selected-map handoff into
legacy publication records. It no longer reads direct `input_hashes`, invokes
legacy `build_competition_plan`, or loads the base through legacy
`CheckpointStore`. Independent evaluation-manifest replay follows the same
typed direct route and compares the full `plan_snapshot` and
`input_descriptors`; revision assembly revalidates that direct base before its
separate legacy candidate-only stages.

The Important publication-boundary finding is also fixed. One shared
`validate_comparator_selection_object` boundary now fixed-loads the canonical
Task 11 receipt, semantically recomputes its projection, and directly compares
the exact five-field selection object. Schema-2, schema-3/revision, promotion,
evaluation-manifest, and final selection consumers use this validator.
Schema-2 and schema-3 writers revalidate it after each artifact write;
promotion revalidates the published object; every boundary requires unchanged
full canonical receipt bytes.

The adjacent downstream direct adapter was migrated from the obsolete
`selected_rows` call shape to the complete receipt-backed selection object.
Because direct checkpoints intentionally do not retain evaluator matrices, a
valid grouped null-DE design now emits an explicit fail-closed `unavailable`
row/audit with reason `direct_evaluator_output_not_retained` instead of
aborting schema-2 publication or fabricating a numeric value. Scientific
selection gates therefore remain blocked when that numerical evidence is not
retained.

### Controller-fix TDD evidence

- Promotion tampering of the selected map, nonexecution map, and ready
  population initially produced three `DID NOT RAISE` failures; all three pass
  after the shared validator was installed.
- The same three unchanged-receipt mutations at schema-3 initially reached the
  unrelated stage-denominator error; all three now fail at complete comparator
  validation and the focused slice passes.
- The schema-2 mutations initially reached the orthogonal boundary and failed
  with a `TypeError`; all three now fail before publication and the focused
  slice passes.
- The complete direct checkpoint/receipt handoff was RED on the absent direct
  reconstruction type and then GREEN. The real public schema-2 builder was RED
  on its obsolete active-repository restriction and then GREEN through the
  direct plan/store route.
- Forcing the direct manifest dispatcher off produced the expected legacy
  closed-schema failure; restoring direct dispatch passed independent replay.
- The downstream receipt-backed adapter test was RED on the obsolete function
  signature and then passed together with the required-boundary signature
  check (`2 passed in 2.08s`).
- The grouped direct null-DE regression was RED on the former hard exception
  (`1 failed in 98.13s`) and GREEN with explicit unavailable evidence
  (`1 passed in 98.00s`).
- The direct manifest fixed-path assertion was RED with `checkpoint.json` and
  GREEN with the fixed repository-relative binding (`1 passed in 134.34s`).

### Final controller-fix verification

```text
Required development/authority/candidate/promotion suite:
175 passed in 889.69s (0:14:49)

Direct comparator/plan/checkpoint suites:
243 passed in 273.27s (0:04:33)

Complete downstream-evidence suite:
89 passed in 19.53s

Revision-evaluation suite:
7 passed in 78.64s (0:01:18)

Scoped direct audits plus downstream direct boundary:
4 passed in 3.38s

Ruff check: All checks passed!
Ruff format --check: 12 files already formatted
python -m compileall -q maskimpute_benchmark scripts tests: exit 0
git diff --check: exit 0
```

No comparator, smoke, tuning, evaluator, competition, final, or other real
scientific workload ran. All new evidence and all verification fixtures were
synthetic. The progress ledger was not edited.
