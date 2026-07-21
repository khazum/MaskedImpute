# Task 14 implementation report

## Result and scope

Implemented publication freeze of the complete direct comparator handoff from
accepted Task 13 base `aa802d22df6b9ef373f516badaf898664b1f37f3` in the
isolated `codex/publication-integration` worktree.

The implementation:

- adds `study/comparator_tuning.json` and the generated comparator-selection
  receipt to the existing closed outer publication inventory;
- replaces the frozen builder's legacy comparator-ID input with the validated
  typed `ComparatorTuningAuthority` and complete comparator-selection receipt;
- recomputes the receipt projection and freezes the exact four policy sets,
  ready population, complete selected bound configurations, complete
  unavailable nonexecution identities, exact five-field comparator-selection
  projection, and ordered 12-row scheduled same-input status table;
- makes scheduled controls and comparators receipt-authoritative, rejects
  execution-summary substitutes, requires every established comparator to be
  selected, and accepts only intrinsic modern/scSDAE unavailability with the
  complete matching nonexecution identity;
- binds selected and unavailable comparator method identities field by field
  to the current validated method registry without reconstructing registry
  defaults;
- fixed-loads the selection receipt against the base direct checkpoint,
  revalidates its complete authority and evidence, and rereads both direct
  artifacts before and after frozen-payload construction;
- requires clean frozen validation to reopen and recompute the direct
  comparator handoff while retaining the pre-existing outer publication,
  stage, tree, candidate, calibration, and environment provenance unchanged;
  and
- preserves Task 13 candidate-only revision behavior by recognizing the exact
  nested direct identity in a 48-row MaskImpute-only v28/v29 checkpoint and
  comparing the full selected candidate payload directly.

No comparator, smoke, tuning, evaluator, competition, final, scaling, or other
real scientific workload ran. All execution evidence used by tests was
synthetic. The progress ledger was not edited.

## Direct frozen schema

The frozen method now contains:

- `scheduled_same_input_ids`, `required_control_ids`,
  `established_comparator_ids`, and `modern_core_ids` in authority order;
- `ready_comparison_population_ids` derived only from the complete receipt;
- `selected_comparator_configurations`, whose values contain the full readable
  configuration payload, authority reference, and method binding;
- `unavailable_comparator_nonexecution_identities`, containing the complete
  Task 11 identity for every intrinsically unavailable allowed method;
- `scheduled_same_input_statuses`, with exactly 12 ordered rows containing
  aggregate status, terminal counts, reason histogram, and the full selected
  or nonexecution identity; and
- `comparator_selection`, the exact complete five-field Task 12 projection.

`required_comparator_ids` is absent from new frozen payloads. Scheduled
comparator rows have no legacy development-execution summary; they retain the
complete receipt-authoritative status and selected/nonexecution value instead.

## TDD evidence

Production changes began from focused failing regressions. Observed RED cases
included:

- the builder rejecting the new typed authority and receipt arguments;
- production preparation failing on the old required-comparator interface;
- unavailable comparator registry-binding drift being accepted;
- Task 13 direct revision rows being rejected because freeze still read flat
  legacy run fields; and
- typed receipt mutations and the production receipt/frozen-payload rewrite
  lacking Task 14 coverage.

Focused GREEN evidence after the corresponding production changes included:

```text
initial complete frozen payload shape: 1 passed, 118 deselected
production prepare plus clean recomputation: 1 passed, 118 deselected
clean intrinsic unavailability plus registry-drift rejection:
  2 passed, 117 deselected in 41.61s
direct revision plus adjacent legacy swap/rejection cases:
  6 passed, 114 deselected in 21.43s
complete selected receipt typed-field tamper matrix:
  7 passed, 121 deselected in 48.85s
production receipt plus coherently re-checksummed frozen payload tamper:
  1 passed, 128 deselected in 85.73s
v28 full prepare plus clean revalidation:
  1 passed, 128 deselected in 152.39s
v29 plus stage-race validation:
  2 passed, 127 deselected in 229.78s
```

The migrated suite also exposed legacy tests that deliberately removed all
ignored development evidence or rewrote legacy flat checkpoints. Those tests
now reflect the governing direct contract: comparator receipt/checkpoint bytes
remain required for clean validation, while revision checkpoints use nested
direct candidate identity. No production behavior was relaxed to preserve a
superseded expectation.

## Verification evidence

Required complete warning-strict publication-freeze suite:

```text
tests/test_freeze_publication_round.py:
  129 passed in 2057.04s (0:34:17)
```

Adjacent complete authority and consumer suites:

```text
tests/test_comparator_tuning.py:
  114 passed in 223.95s (0:03:43)
tests/test_revision_authority.py + tests/test_revision_evaluation.py:
  32 passed in 189.72s (0:03:09)
```

Scoped direct source/schema audits:

```text
3 passed, 98 deselected in 3.03s
```

Final static gates:

```text
ruff format --check: 2 files already formatted
ruff check: All checks passed!
python -m compileall -q [touched source and test]: exit 0
git diff --check: exit 0
```

## Implementation decisions

1. The Task 11 receipt remains the sole scheduled comparator availability and
   selection authority. Freeze does not infer availability from aggregate
   execution summaries or registry defaults.
2. The readable top-level selected map exposes the complete typed payload,
   while `comparator_selection` retains the exact accepted five-field Task 12
   representation. Both are recomputed from the same validated receipt.
3. Established comparator nonexecution always blocks. Only modern-core methods
   and scSDAE may carry intrinsic terminal nonexecution, and their complete
   method binding must still match current authority.
4. Existing outer legacy publication provenance remains unchanged. Direct
   comparator fields contain full typed values and complete canonical receipt
   evidence rather than comparator-specific content summaries.
5. Base direct checkpoints contribute candidate evidence only to the existing
   outer candidate provenance; scheduled controls and comparators are frozen
   exclusively from the complete selection receipt.

## Concerns

No blocking Task 14 concern remains. The full freeze suite, complete comparator
tuning suite, adjacent revision authority/evaluation suites, scoped direct
audits, formatting, lint, compilation, and diff checks all pass. Independent
acceptance remains the next workflow step after the required commit and review
package are generated.
