# Task 6 report: Selection, revision, freeze, and post-freeze authority

## Status

Complete.

Review base:

```text
dbd766618e2b62b6c7a0295036ff8e02b9699dc0
```

Worktree:

```text
/home/marcinmaleclocal/Coding/MaskedImpute/.worktrees/publication-integration
```

## Scope and constraints

The audit traced persisted development evidence through direct-comparator
projection, selection and promotion, bounded revisions, publication freeze,
final execution planning, trajectory and downstream evaluation, scaling, and
publication synthesis. It inspected every production and test file assigned by
the Task 6 brief.

No real scientific or comparator workload ran. The review did not change the
selection objective, candidate set, revision allowance, final seed denominator,
scheduled or numerical populations, estimand, multiplicity family, runtime
lock, filesystem hardening, or legacy outer provenance. It added no new
checksum, fingerprint, or content-summary mechanism and made no empirical
competitiveness or publication-readiness claim.

## Authority and claim audit

The complete authority path was reviewed from persisted development scores to
the final claim surface:

- direct comparator evidence remains bound to the completed fixed competition
  plan and projected selection receipt;
- selected methods and reason-coded nonexecutions remain a complete,
  mutually-exclusive denominator through promotion and revision;
- revision activation remains bounded by the prespecified revision authority;
- freeze reconstruction remains commit-bound and rejects altered frozen
  evidence;
- final plans retain the exact action vocabulary, fixed seeds, and one-use
  lifecycle;
- trajectory, downstream, and scaling records remain plan-prefix-bound and
  population-complete;
- final analysis and publication synthesis retain scheduled-versus-numerical
  accounting, unavailable-estimand suppression, confidence-direction checks,
  and multiplicity permissions.

No development selector was found to consume final performance, downstream
endpoints, final data, or post-freeze results. No additional reachable
selection, revision, one-use execution, or claim-permission bypass remained
after the corrections below.

## Findings closed

### F-066: Masked downstream authorities could lose missingness during coercion

Severity: Important.

Dense masked method output, held-out counts, trajectory truth, and group-marker
masks were accepted because ordinary NumPy conversion discarded mask
semantics. A sparse matrix could carry the same problem in its stored data.

The downstream numeric boundary now rejects masks before dense conversion,
including masked sparse storage. This prevents missing evaluator truth or
method values from becoming ordinary completed evidence.

### F-067: Downstream float64 narrowing leaked floating-point exceptions

Severity: Important.

Finite extended-precision values outside the float64 range could raise
`FloatingPointError` during method-output or trajectory conversion, violating
the declared validation exception contract.

Both boundaries now perform narrowing under a local strict state and translate
unrepresentable values to stable `ValueError` failures. No value is clipped or
silently replaced.

### F-068: Held-out CP10k normalization overflowed on representable rows

Severity: Important.

The downstream held-out path summed and scaled extreme but finite float64 rows
directly. The intermediate library-size or multiplication could overflow even
when the normalized log2 result was representable.

The path now uses the already-reviewed count-equivalent conversion for nonzero
rows and preserves exact zero rows. Extreme representable controls complete
with the declared endpoints under strict floating-point state.

### F-069: Typed final authorities accepted states outside the closed domain

Severity: Important.

`FinalPlanEntry` and `FrozenPlanMethodAuthority` accepted unknown action text.
The frozen authority also lacked a complete typed check for safe method IDs,
exact seed tuples, duplicate seeds, and blank nonexecution reasons.

The typed constructors now admit only `execute` or `not_applicable`, require
safe method identity, require a nonempty exact tuple of unique nonnegative
integer or `None` seeds, and reject whitespace-only reasons. Existing execute
and reason-coded nonexecution populations are unchanged.

### F-070: Nested direct authority snapshots remained caller-mutable

Severity: Important.

Direct reconstruction evidence and frozen comparator nonexecution authority
used shallow mapping wrappers. A caller retaining a nested source dictionary or
list could mutate already-validated selected or nonexecution authority.

Both boundaries now recursively normalize and detach the direct values before
exposing immutable mappings. Mutation of the caller's original nested
structures no longer changes validated evidence.

### F-071: Boolean values could impersonate integer schema and endpoint fields

Severity: Important.

Python Boolean values compare equal to integers. Boolean schema versions were
therefore accepted by final-manifest, frozen-method-plan, scaling-checkpoint,
and scaling-dataset-receipt validators. Downstream endpoint records likewise
accepted Boolean independent counts and completed values, and persisted
downstream evidence accepted a Boolean independent count.

Each boundary now requires the exact integer schema/count type. Completed
endpoint values explicitly reject Python and NumPy Boolean values while
retaining genuine finite numeric scalar support. The frozen publication
validator was separately challenged and was already fail-closed through its
existing commit-bound reconstruction.

### F-072: Scaling lock regression depended on large-artifact storage speed

Severity: Minor.

The two-writer regression paused the first writer before large artifact writes
and replay validation, then required completion within 20 seconds after
release. Its intended lock assertion failed on the review host even though the
process remained CPU-bound and the product lock behaved correctly.

The pause now occurs at the already-validated publish boundary. The second
writer still proves it cannot enter transaction preparation while the first
writer owns the operation, and it still rejects its stale cached prefix after
release. The corrected isolated node passed in 183.04 seconds; no production
locking behavior changed.

## Test-first evidence

The initial focused mutation set produced ten expected failures:

```text
10 failed in 27.65s
```

After the owning corrections, the same ten nodes reported:

```text
10 passed in 27.66s
```

Adjacent audit nodes then demonstrated and closed the remaining gaps:

- Boolean endpoint count/value: two expected failures, then two passes.
- Nested selected-map mutation: one expected failure, then one pass.
- Boolean frozen final-plan receipt: one expected failure, then one pass.
- Trajectory narrowing exception translation: one expected failure, then one
  pass.
- Boolean scaling checkpoint/dataset receipt: two expected failures, then two
  passes.

The complete downstream evaluation owning file reported:

```text
43 passed in 2.17s
```

The first authoritative 15-file run exposed only F-072:

```text
1 failed, 829 passed in 7096.09s (1:58:16)
```

The same scaling concurrency node failed independently at the same 20-second
future deadline:

```text
1 failed in 133.94s (0:02:13)
```

After moving the test pause to the publish boundary, the isolated node
reported:

```text
1 passed in 183.04s (0:03:03)
```

The new clean final-state sanitized 15-file authority and post-freeze suite
then reported:

```text
830 passed in 7065.83s (1:57:45)
```

No production or test file changed during or after that final suite. Only this
report and the tracked review ledger were added afterward.

## Static verification

Before the final authoritative suite:

```text
Ruff check: All checks passed
Ruff format --check: 11 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The same static and whitespace gates were repeated after documentation was
recorded.

## Disposition

Task 6 is complete. Findings F-066 through F-072 are closed by bounded
corrections and regression coverage. The fixed selection, revision, freeze,
final, downstream, scaling, and synthesis populations remain unchanged.

These checks establish authority and claim contracts over bounded fixtures.
They do not establish empirical method competitiveness, external runtime
availability, real-data results, or Genome Biology submission readiness.
