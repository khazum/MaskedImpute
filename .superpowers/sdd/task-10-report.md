# Task 10 implementation report

## Result

Implemented exact development-only comparator configuration selection in
`maskimpute_benchmark/comparator_tuning.py`. The implementation collapses the
fixed seed/view/draw grid, retains complete `BoundComparatorConfiguration`
values on every collapsed and ranked row, excludes intrinsic-terminal
configurations without aborting other configurations, applies strict Pareto
dominance, encodes median average ranks as exact integer quarter-ranks, and
selects by the prescribed deterministic tuple.

No comparator, smoke, tuning, evaluator, competition, final, or other
scientific workload was run.

## Files changed

- `maskimpute_benchmark/comparator_tuning.py`
  - Added the collapsed, ranked, and method-selection result types.
  - Added collapse, Pareto, quarter-rank, and one-method selection APIs.
  - Added closed direct record/identity/metric and unit-grid validation.
- `tests/test_comparator_tuning.py`
  - Added the tracked-authority golden helper and exact synthetic 48-record,
    252-metric-row-per-configuration factory.
  - Added golden nesting, Pareto, tie, tuple-order, complete-bound-identity,
    intrinsic-terminal eligibility, and fail-closed mutation regressions.
- `.superpowers/sdd/task-10-report.md`
  - Recorded the TDD evidence, verification, decisions, and self-review.

`.superpowers/sdd/progress.md` was not edited.

## TDD evidence

### Baseline

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest tests/test_comparator_tuning.py -q -W error -p no:cacheprovider
```

Result before Task 10 changes: `82 passed in 1.75s`.

### Required RED selector

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'collapse or quarter_rank or pareto' \
  -q -W error -p no:cacheprovider
```

Result: exit 2 with one collection error. Import failed exactly because
`collapse_comparator_configuration` was absent from
`maskimpute_benchmark.comparator_tuning`.

### Adjacent-interface RED

After changing the golden records to match the real direct execution record,
where comparator `p_pre_zero_evidence` is non-applicable even on SymSim:

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_seed_view_draw_collapse_and_quarter_rank_golden \
  -q -W error -p no:cacheprovider
```

Result: `1 failed`. The failure exposed incorrect coupling between comparator
pre-zero evidence and SymSim-only `mse_pre_dropout_zero` metric applicability.
Production validation was narrowed to keep those concepts separate.

### Complete-bound RED

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py::test_collapse_rejects_bound_authority_reference_drift \
  -q -W error -p no:cacheprovider
```

Result: `1 failed` because a drifted schema version did not raise. The bound
validator was then tightened to require the exact canonical authority reference
and one exact canonical authority row.

### Required GREEN selector

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py -k 'collapse or pareto or rank or selection_tuple' \
  -q -W error -p no:cacheprovider
```

Final result: `15 passed, 79 deselected in 1.41s`.

### Adjacent comparator, execution, and checkpoint suites

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_comparator_tuning.py tests/test_fair_comparator_execution.py \
  tests/test_fair_comparator_checkpoint.py \
  -q -W error -p no:cacheprovider
```

Result: `239 passed in 8.80s`.

### Direct source/schema audit suite

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_selection_authority.py \
  -k 'scoped_direct or direct_closed_metadata' \
  -q -W error -p no:cacheprovider
```

Result: `3 passed, 96 deselected in 2.68s`.

## Implementation decisions

- The entry point accepts canonical serialized direct checkpoint records and
  may receive the full checkpoint record sequence. It selects the requested
  method while validating every selected record against fixed authority rows.
- Every configuration requires exactly 48 unique `(dataset_id, model_seed)`
  cells and exactly 252 applicable metric rows: 240 panel-wide rows plus 12
  SymSim pre-zero rows.
- Seed values are averaged within each of 16 datasets. Moderate and severe
  means are paired by `(mechanism, biological_id)`, yielding eight ordered
  independent units for five metrics and two ordered SymSim units for pre-zero.
- Canonical unit IDs are readable `mechanism:biological_id` values. Pareto rows
  must have identical unit-ID tuples for each metric before ranking.
- A completed numeric metric row must contain a finite float. A canonical
  intrinsic-terminal row with `value=None` retains the configuration in the
  denominator but makes it ineligible. Blocking statuses fail closed.
- The method binding is reconstructed from the complete record identity,
  compared field by field within and across configurations, and retained with
  the exact authority configuration and reference.
- Pareto dominance requires weak improvement in all six medians and strict
  improvement in at least one. Ranking is Pareto-only and uses doubled average
  ranks followed by exact integer median quarter encoding.
- The selection tuple is exactly `(max ranks, sum ranks, six ranks, upstream
  default penalty, configuration_id)`. With no eligible Pareto row,
  `selected_configuration_id` is `None`.
- Comparator `p_pre_zero_evidence` is distinct from metric applicability. The
  former remains non-applicable for comparator methods; only the metric row is
  restricted to SymSim.

## Static checks

The final static gate was:

```bash
ruff check maskimpute_benchmark/comparator_tuning.py tests/test_comparator_tuning.py
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m compileall -q \
  maskimpute_benchmark/comparator_tuning.py
git diff --check
```

Observed final state: Ruff reported `All checks passed!`, compileall exited 0
without output, and `git diff --check` exited 0 without output.

## Self-review

- Confirmed each collapsed and ranked row retains the full bound configuration;
  no configuration/payload identity surrogate was added.
- Confirmed exact authority configuration ordering drives result ordering.
- Confirmed duplicate/missing metric rows, wrong pre-zero applicability,
  nonfinite values, identity drift, duplicate cells, malformed view/draw grids,
  and bound authority drift fail closed.
- Confirmed identical per-seed values still yield 8/2 independent units, not
  24/6 observations.
- Confirmed intrinsic-terminal evidence only removes its own configuration from
  eligibility and does not abort selection among the method's remaining rows.
- Confirmed quarter-ranks are integers and the default/ID tie breakers occupy
  the exact final tuple positions.
- Confirmed the direct source/schema audit remains green and no unrelated
  legacy behavior or tracked scientific authority changed.

## Concerns

None.
