# Task 6 acceptance correction report

## Status

Complete.

Correction base:

```text
ec08b1f4c74b1224d57f9f71d63c0a650e301a52
```

## Scope and constraints

This correction closes only the four findings returned by the independent
Task 6 acceptance review. It does not change the selected methods, final
population, seeds, revision allowance, estimands, runtime behavior, or claim
permissions. No real scientific workload ran. No new checksum, fingerprint,
content-summary, filesystem-hardening, or cyber-related mechanism was added.
Legacy outer provenance is unchanged.

## Findings closed

### F-073: Final-manifest Boolean count aliases

Severity: Important.

`failed_count=False` passed equality with the required integer zero. The
manifest boundary now requires exact built-in integer types for
`independent_unit_count`, `completed_count`, and `failed_count`, while
preserving the canonical integer values.

### F-074: Open final non-run reason vocabulary

Severity: Important.

Typed final entries and frozen method authorities admitted any nonblank
non-run reason, and frozen applicability accepted arbitrary unavailable
reasons. The final authority now uses only the reason codes already emitted by
the publication freeze: fixed historical, matched-bulk, and registry
applicability reasons, plus the existing technical-unavailability form.
Validation is disposition-specific at direct constructors and frozen receipt
reconstruction. No scientific reason was invented.

### F-075: Nested direct keys were silently rewritten

Severity: Important.

Direct normalization converted a nested non-string mapping key to text before
the canonical-value freezer could reject it. Both direct serialization paths
now require string mapping keys before recursive conversion. Valid nested
snapshots remain detached and immutable.

### F-076: Endpoint integer conversion leaked `OverflowError`

Severity: Important.

Huge positive and negative Python integers reached `float(...)` in direct
endpoint value and multiple-testing-alpha validation and in persisted
downstream endpoint prevalidation. The owning boundaries now classify those
values as invalid and raise their declared `ValueError` or
`DownstreamEvidenceError`.

## Test-first evidence

The initial focused acceptance set reported:

```text
9 failed, 3 passed in 71.36s
```

The analogous family-alpha conversion was then restored to its original state
and separately demonstrated:

```text
2 failed in 2.01s
```

After the owning corrections and replacement of three ad-hoc test-only reason
strings with existing canonical reason codes, the complete focused set
reported:

```text
14 passed in 78.93s (0:01:18)
```

Owning and adjacent suites reported:

```text
tests/test_downstream_evaluation.py: 48 passed in 2.38s
tests/test_final_runner.py: 122 passed in 667.35s (0:11:07)
tests/test_downstream_evidence.py: 149 passed in 82.26s (0:01:22)
freeze/final-analysis/publication-synthesis/development-evaluation:
266 passed in 2717.82s (0:45:17)
```

After formatting and static checks, the exact sanitized fifteen-file Task 6
suite from the brief reported:

```text
843 passed in 6033.96s (1:40:33)
```

No production or test file changed during that execution. Only this result and
the matching tracked ledger evidence were recorded afterward.

## Disposition

F-073 through F-076 are closed at their owning validation boundaries. These
bounded contract checks do not establish empirical competitiveness, external
runtime availability, or Genome Biology submission readiness.
