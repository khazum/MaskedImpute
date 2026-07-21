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

## Second independent re-review Critical fix

The remaining production downstream Critical is closed. The production
`build_development_downstream_evidence_plan` component now reads the checkpoint
identity mode before any legacy planning. A direct-v1 base rebuilds the exact
fair-comparator plan from the checkpoint's complete embedded smoke evidence,
loads and projects the fixed Task 11 selection receipt against that exact
checkpoint, and calls the accepted receipt-backed
`validate_direct_comparator_projection` adapter. That adapter validates the
complete direct plan and terminal checkpoint through `DirectCheckpointStore`.

Only the plain `{comparator_authority, selected_comparators}` map crosses into
the established downstream publication envelope. Selected comparator
configurations are adapted only after that boundary; nonselected comparator
configurations are excluded. Because direct checkpoint attempts intentionally
retain no evaluator matrices, completed selected rows become explicit
fail-closed `unavailable` downstream rows with reason
`direct_evaluator_output_not_retained` and no numerical output binding.

Direct base components also replay through the same production builder during
persisted-plan and revision-bundle revalidation. Separately activated v28/v29
candidate-only components remain on the unchanged legacy planner. Legacy
checkpoints without `identity_mode` retain the existing `input_hashes` and
`build_competition_plan` path; any other identity mode fails closed.

### Second-fix TDD evidence

Focused RED, from the production entry point after adding the guarded direct
checkpoint regression:

```text
FAILED tests/test_downstream_evidence.py::test_development_production_wrapper_routes_direct_base_without_legacy_planner
DownstreamEvidenceError: development source input authority is absent
1 failed in 2.36s
```

The final regression uses a guarded direct checkpoint mapping whose
`get("input_hashes")` fails the test, replaces legacy
`build_competition_plan` with an immediate failure, and observes the exact
adapter and plain selected-map handoff. Focused GREEN:

```text
1 passed in 2.15s
```

During boundary refactoring, the scoped source audit correctly rejected outer
publication-envelope provenance under a direct-named helper:

```text
1 failed, 1 passed in 2.95s
```

The helper was moved explicitly to the post-direct selected-map boundary; no
summary helper or generated summary field was added to the direct segment.
Focused regression plus both scoped direct audits then passed:

```text
3 passed in 3.38s
```

### Second-fix verification

```text
Complete downstream-evidence suite:
90 passed in 19.29s

Adjacent complete synthetic direct checkpoint/receipt validation:
1 passed in 80.99s (0:01:20)

ruff check maskimpute_benchmark/downstream_evidence.py tests/test_downstream_evidence.py:
All checks passed!

ruff format --check maskimpute_benchmark/downstream_evidence.py tests/test_downstream_evidence.py:
2 files already formatted

python -m compileall -q maskimpute_benchmark scripts tests: exit 0
git diff --check: exit 0
```

No real comparator, smoke, tuning, evaluator, competition, final, scaling, or
other scientific workload ran. All verification evidence was synthetic. The
progress ledger was not edited. The exact post-commit review package path is
`.superpowers/sdd/review-fcfdb30..task12-second-fix.diff`.

## Final re-review executable-handoff fix

The last Critical is closed without changing the runner-owned configuration
contract. The production regression no longer mocks
`_build_downstream_plan_from_selected_handoff`: it supplies one selected
comparator, one nonselected comparator, one observed control, matching direct
plan entries/checkpoint records, and a real downstream dataset binding through
`build_development_downstream_evidence_plan`.

The post-boundary adapter now uses a closed
`ProjectedDownstreamConfiguration` type for only `selected_comparator` and
`direct_control`. It uses the established seven-field downstream configuration
envelope and existing outer downstream provenance. Candidate, ablation, and
capacity-control configurations remain the existing runner-owned
`AuthorizedConfiguration` values; the closed runner kinds and its prohibition
on legacy comparator tuning were not weakened.

The real selected-map handoff retains the selected comparator and observed
control, excludes the nonselected comparator, and maps completed direct runs
to `status="unavailable"` with reason
`direct_evaluator_output_not_retained` and every evaluator output binding null.
The focused regression writes the resulting plan with zero evaluated
denominators and reloads it through the public persisted-plan validator,
requiring byte-equivalent plan values after full production-source replay.

### Final-fix TDD evidence

The unmocked production-entry RED reached the exact reviewed failure:

```text
FAILED tests/test_downstream_evidence.py::test_development_production_wrapper_routes_direct_base_through_real_handoff
maskimpute_benchmark.runner.RunnerContractError: configuration kind is invalid
1 failed in 2.58s
```

After introducing the closed downstream-owned type, the final focused GREEN
was:

```text
1 passed in 1.89s
```

### Final-fix verification

```text
Complete downstream-evidence module:
90 passed in 18.93s

Complete downstream-evidence module plus both scoped direct audits:
92 passed in 20.46s

Adjacent complete synthetic direct checkpoint/receipt validation:
1 passed in 80.31s (0:01:20)

ruff check maskimpute_benchmark/downstream_evidence.py tests/test_downstream_evidence.py:
All checks passed!

ruff format --check maskimpute_benchmark/downstream_evidence.py tests/test_downstream_evidence.py:
2 files already formatted

python -m compileall -q maskimpute_benchmark scripts tests: exit 0
git diff --check: exit 0
```

No real comparator, smoke, tuning, evaluator, competition, final, scaling, or
other scientific workload ran. All evidence was synthetic. No direct identity
summary field/helper was added, the scoped audits pass, and the progress ledger
was not edited. The full accepted-base review package remains
`.superpowers/sdd/review-fcfdb30..task12-second-fix.diff` and is regenerated
from `fcfdb30c68b6ef5b15a6733ca461fb638457646c` after the final commit.
