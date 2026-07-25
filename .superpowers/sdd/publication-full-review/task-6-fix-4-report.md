# Task 6 v4 final-acceptance correction report

## Status

Complete.

Correction base:

```text
0fa2719
```

## Scope and constraints

This correction closes only the four important gaps returned by the latest
independent Task 6 review. It does not alter selected methods, final datasets,
the fixed 1,760 primary or 44 trajectory run population, model seeds, revision
allowances, estimands, execution behavior, or claim permissions.

No real scientific or comparator workload ran. No plan seal field was inspected
or changed by the correction, and no new content-summary, filesystem-hardening,
runtime-lock, or cyber-related mechanism was added.

## Findings closed

### F-082: Full-size typed plans remained structurally forgeable

Severity: Important.

The public evaluation gates required a full Cartesian population but derived
that population from the submitted plan itself and used subclass-admitting
type checks. A caller could therefore replace a complete dataset block,
coherently forge the trajectory dataset, or supply derived plan, entry, run, or
configuration authorities.

The public final and trajectory gates now require exact plan, entry, run,
configuration, and relevant nested authority types. They independently replay
the fixed method order and seed policy, primary 4 x 5 x 2 scientific
coordinates or registered trajectory identity, canonical run construction,
configuration/action/reason bindings, and planned preflight fields. Canonical
1,760-run and 44-run controls remain valid.

### F-083: Direct equality normalized malformed frozen objects

Severity: Important.

`direct_equal` converted tuple-backed frozen objects to dictionaries before
validating them. Duplicate keys and noncanonical order could disappear, while a
malformed pair could raise an incidental dictionary-conversion error.

Every frozen-object operand is now structurally validated before equality.
Duplicate keys, noncanonical order, malformed pair shapes, non-string keys, and
derived wrappers are rejected on either side. Valid frozen-object comparisons
retain their mapping/list compatibility.

### F-084: Score evidence admitted Boolean integer aliases

Severity: Important.

A Boolean false metric denominator compared equal to integer zero, and Boolean
true compared equal to reliability-bin ordinal one. Both fields now require the
exact built-in integer type before value or ordering checks.

### F-085: Selection and claim numeric coercion leaked `OverflowError`

Severity: Important.

Huge positive and negative Python integers reached unguarded `float`
conversions in `GateResult`, both selection finite-number helpers, final
analysis, and publication synthesis. The owning helpers now translate
unrepresentable numeric inputs to their documented `ValueError`,
`FinalAnalysisContractError`, or `PublicationSynthesisError` domain failure.
Valid finite numeric behavior is unchanged.

## Test-first evidence

The focused RED invocations reported:

```text
20 failed, 1 passed in 3.38s
12 failed, 2 passed in 30.61s
```

After the minimal shared corrections, the complete formatted focused set
reported:

```text
38 passed in 44.81s
```

The complete five changed and adjacent owner files reported:

```text
354 passed in 1569.69s (0:26:09)
```

After removing the only newly added comparison involving a seal-named field,
the exact affected focused plan set and complete final-runner owner reported:

```text
17 passed in 45.17s
154 passed in 753.58s (0:12:33)
```

## Static verification

Ruff formatting and lint, scoped byte compilation, and `git diff --check`
exited zero before the post-refinement owner rerun and again immediately before
the exact fifteen-file acceptance suite.

## Exact Task 6 acceptance suite

The exact sanitized fifteen-file Task 6 suite from the brief reported:

```text
936 passed in 6444.11s (1:47:24)
```

No production, test, or documentation file changed during that run. Only this
terminal evidence and the matching tracked ledger entry were recorded
afterward.

## Disposition

F-082 through F-085 are closed at the owning public population, direct-value,
score-evidence, and numeric coercion boundaries. The exact Task 6 acceptance
suite is green from the final production/test state.

## Subsequent review status

The status above was scoped to F-082 through F-085. It did not constitute
final Task 6 acceptance. A later independent review identified F-086 through
F-089, which are tracked in the v5 correction report and full-review ledger.
