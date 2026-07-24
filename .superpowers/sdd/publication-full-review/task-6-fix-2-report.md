# Task 6 final acceptance correction report

## Status

Complete.

Correction base:

```text
1da78e70cf3a029d0c6f8edf70c5bbbbd086f26d
```

## Scope and constraints

This correction closes only the four gaps returned by the final independent
Task 6 acceptance review. It does not change selected methods, scheduled or
numerical populations, final seeds, revision allowances, estimands, runtime
policy, concurrency behavior, or claim permissions. No real scientific
workload ran. No new provenance, content-validation, filesystem-hardening, or
cyber-related mechanism was added. Legacy outer provenance remains unchanged.

## Findings closed

### F-077: Direct reconstruction evidence retained caller-owned nested values

Severity: Important.

`DirectReconstructionEvidence` was a frozen dataclass but did not acquire its
own recursive snapshot. Caller mutation could therefore change nested plan,
input, record, or selected-configuration values after construction, including
through the revision loader. Construction now validates canonical direct
values and stores recursively detached immutable mappings and sequences.
Existing mapping access used by reconstruction consumers is preserved.

### F-078: Malformed frozen object wrappers bypassed structural validation

Severity: Important.

`FrozenDirectObject` is tuple-backed, and preconstructed malformed instances
could reach freeze and thaw operations without validating pair shape, key
type, uniqueness, or canonical key order. Direct-value boundaries now validate
the wrapper before freezing, thawing, or serializing it. Valid wrappers retain
their existing representation and round trip.

### F-079: Post-freeze integer authority fields accepted Boolean aliases

Severity: Important.

Persisted final execution manifests and transaction intents, downstream plans,
manifests, references and records, and publication scaling evidence used
integer equality at some boundaries without first requiring the exact built-in
integer type. Those owning loaders now require exact integers for schema
versions, counts, and ordinals before accepting their values.

### F-080: Unrepresentable integers leaked numeric conversion exceptions

Severity: Important.

Huge positive and negative Python integers could leak `OverflowError` from
endpoint values, null-DE alpha, and persisted scaling runtime or metric
validation. Each owning boundary now converts through a guarded finite-real
validator and raises its declared contract exception for invalid or
unrepresentable values.

## Test-first and compatibility evidence

The initial focused acceptance set reported:

```text
17 failed, 1 passed in 184.34s (0:03:04)
```

The passing persisted-plan case was rejected earlier by its existing binding;
the owning semantic schema boundary still lacked the exact type requirement
and was corrected with the other persisted authorities.

After the owning corrections, the focused set initially exposed a compatibility
error: recursively frozen records no longer supported the established nested
mapping access used by reconstruction selection. The exact development and
revision owner run reported:

```text
3 failed, 37 passed in 488.89s (0:08:08)
```

The immutable snapshot was adjusted to retain read-only mapping access without
retaining caller ownership. The three exact regressions then reported:

```text
3 passed in 95.16s (0:01:35)
```

The complete focused correction set reported:

```text
18 passed in 183.73s (0:03:03)
```

The complete five owning files reported:

```text
375 passed in 3379.70s (0:56:19)
```

Two adjacent direct-value/runner compatibility nodes reported:

```text
2 passed in 2.06s
```

Ruff formatting and lint, byte compilation, and `git diff --check` all exited
zero before the exact Task 6 suite.

## Exact Task 6 acceptance suite

The exact sanitized fifteen-file suite from the Task 6 brief reported:

```text
860 passed in 6211.36s (1:43:31)
```

No production or test file changed during that execution. Only this result and
the matching tracked ledger evidence were recorded afterward.

## Disposition

F-077 through F-080 are closed at their owning validation boundaries. These
bounded contract corrections do not establish empirical competitiveness,
external-runtime availability, or Genome Biology submission readiness.
