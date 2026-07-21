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
