# Task 6 v5 independent-review correction report

## Status

Complete.

Correction base:

```text
a3b1374679b939c73f18fedcc7b40952a36917be
```

## Scope

This correction closes only the four important gaps returned by the latest
independent Task 6 review. It does not change selected methods, fixed final or
scaling populations, seeds, revision allowances, estimands, execution
behavior, or claim permissions. No scientific workload ran.

The preceding v4 report's completion statement applied only to F-082 through
F-085 and was not final Task 6 acceptance.

## Findings closed

### F-086: Completed final metrics lacked full evidence validation

Final evaluation required the canonical metric name sequence but did not
validate the remaining evidence for completed runs. Metric rows containing
only names therefore passed.

Completed-run evaluation now requires the complete legacy or direct field set,
the plan-bound scientific and configuration identity, exact nonnegative
integer denominators, and either a finite float value with completed status or
a reason-coded unavailable row. The canonical metric set and order remain
unchanged.

### F-087: Public scaling accepted truncated fixed populations

The public scaling executor called a lower-level plan serializer that
deliberately supports small fixture populations. It did not independently
require the fixed production population before materialization.

The public executor now requires exact scaling plan, entry, configuration, and
contract types; four fixed sizes by five fixed methods; canonical size-method
order and ordinals; the established stochastic seed policy; and configuration
bindings. Generic lower-level fixture and result-store behavior remains
available for unit tests.

### F-088: Completed checkpoint counts could be incomplete

Checkpoint serialization treated the planned count only as an upper bound.
A completed checkpoint could therefore contain zero or partial records and no
dataset receipt.

Completed serialization now requires exactly the planned number of records.
Running checkpoints and synthetic completed replay fixtures with the complete
record population retain their established behavior.

### F-089: Derived frozen-list wrappers passed direct equality

Equality used subclass-admitting frozen-list checks even though freeze, thaw,
and encoding require the exact wrapper type. Both equality operands now pass
through the same exact frozen-list wrapper validator. Canonical frozen-list
and ordinary-list comparisons are unchanged.

## Test-first evidence

The consolidated final-metric RED result listed all six accepted mutations:

```text
names_only
missing_status
boolean_value
boolean_denominator
unavailable_without_reason
completed_with_reason
```

The three completed-checkpoint cases failed because serialization did not
raise. All four public scaling mutations reached dataset materialization
instead of failing at the public population boundary. Both derived
frozen-list operand positions compared without rejection.

After the minimal owning corrections:

```text
final metric mutation set: 1 passed in 28.80s
scaling/checkpoint/direct-list focused set: 9 passed
```

The complete three changed-owner suites reported:

```text
269 passed in 3362.21s (0:56:02)
```

## Exact-suite integration correction

The first exact fifteen-file Task 6 run reported:

```text
3 failed, 943 passed in 6500.93s (1:48:20)
```

The three failures were established downstream replay fixtures whose completed
synthetic checkpoint contained its full planned record population but no
dataset receipts. Requiring nonempty datasets in the generic serializer was
therefore an overconstraint. The fixed public execution boundary and real
result store independently own production dataset completeness.

A focused compatibility control first demonstrated:

```text
1 failed in 2.30s
```

The serializer was then narrowed to retain only the required completed
record-count equality. The two zero/partial-record mutations remain rejected.
The three prior downstream failures reported:

```text
3 passed in 2.07s
```

The complete affected owner suites then reported:

```text
243 passed in 2173.21s (0:36:13)
```

## Final verification

Ruff formatting and lint, scoped byte compilation, and `git diff --check` all
exited zero from the final production/test state. The exact sanitized
fifteen-file Task 6 acceptance suite then reported:

```text
946 passed in 6472.26s (1:47:52)
```

Production, tests, and documentation remained frozen throughout that
authoritative run. F-086 through F-089 are closed, and this correction is
accepted.
