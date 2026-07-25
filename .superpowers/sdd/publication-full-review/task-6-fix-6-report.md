# Task 6 v6 independent-review correction report

## Status

Complete.

The preceding v5 completion statement applied only to F-086 through F-089 and
did not constitute final Task 6 acceptance.

## Scope

This correction closes only the four important gaps returned by the latest
independent Task 6 review. It does not change selected methods, fixed final or
scaling populations, seeds, revision allowances, estimands, execution
behavior, or claim permissions. No scientific workload ran.

## Findings closed

### F-090: Final disposition drift was self-authorizing

The public final-population gate compared each entry only with the mutable
configuration embedded in the same plan. A coherent change to the observed
control's configuration and all forty rows could therefore replace execution
with a technical non-run.

The gate now independently derives the established disposition for every
method. Fixed non-runs retain their established reasons, unavailable
comparators retain their technical non-run, and all other methods must remain
executable. The canonical 1,480 executable and 280 non-run rows are unchanged.

### F-091: Terminal metric rows lacked complete bindings

Executable failures and planned non-runs previously checked only metric name,
terminal status, reason, null value, and a zero denominator. They did not apply
the complete row schema and plan bindings used for completed runs, and Boolean
false passed as integer zero.

Every terminal state now requires the established full row schema, exact
method, dataset, seed, and configuration bindings, canonical metric order, and
an exact built-in integer denominator. Terminal reason and value semantics are
unchanged.

### F-092: Public scaling trusted mutable typed schedule values

The public scaling gate closed the four-size by five-method shape but trusted
typed contract and entry values for the model seed, run labels, output scale,
resource ceilings, measurement provenance, and schedule policy.

Planning and public validation now share the existing fixed entry builder. The
public gate rebuilds the tracked contract and registry values and requires
every permitted contract and entry field to match that reconstruction.
Generic lower-level serialization and synthetic checkpoint behavior are
unchanged.

### F-093: Selection source schemas admitted Boolean one

The development panel, selection contract, calibration contract, ablation
registry, and development-search ledger used equality-only checks for their
source schema version. Each source boundary now requires the exact built-in
integer type. Evaluation-manifest behavior is unchanged.

## Test-first evidence

The focused RED run reported:

```text
26 failed in 115.63s (0:01:55)
```

It covered one coherent final disposition mutation, twelve terminal-row
mutations across both terminal families, eight scaling schedule mutations,
and five selection source-schema mutations. Each family first exercised a
canonical compatibility control.

After the bounded owning corrections, the formatted focused set reported:

```text
26 passed in 113.30s (0:01:53)
```

The complete changed-owner suites reported:

```text
355 passed in 3176.50s (0:52:56)
```

Ruff formatting and lint, scoped byte compilation, and `git diff --check`
passed before the exact acceptance run. The exact sanitized fifteen-file Task
6 suite then reported:

```text
972 passed in 6387.77s (1:46:27)
```

Production, tests, and documentation remained frozen throughout that run.
Post-suite static gates also passed. F-090 through F-093 are closed.
