# Task 7 first independent-review correction report

## Status

Implementation and self-verification are complete. Independent acceptance is
not claimed.

## Corrections

### F-098: Overlapping SAVER build destinations

The builder normalized both destinations but did not require them to be
disjoint before creating their parents. An equal destination or an ancestor
relationship could therefore make the staged library move target a directory
container, or allow the library move to complete before the receipt write
failed.

The builder now rejects normalized equality and either ancestor relationship
before source resolution, parent creation, or build work. Tests cover an equal
path, a normalized receipt below the library, the converse ancestor layout,
and fail-before-write behavior. The canonical sibling receipt remains valid.

### F-099: Existing receipt validation order

Source normalization previously ran before the existing-receipt check. A
missing source parent could therefore return an incidental path error instead
of the documented non-overwrite status.

Receipt normalization and the existing-receipt refusal now precede source
normalization. The regression confirms status 73, unchanged existing content,
and no library creation.

### F-100: Conditional revision CommonMark structure

The first conditional-revision fence used three spaces under ordered item 13,
although that item requires a four-space content indent. CommonMark therefore
closed the item before the v28 block and did not render the conditional
revision sequence as intended.

The v28 block now uses the same four-space continuation as the v29 block. A
structural regression locates ordered items 13 and 14, verifies every
intervening continuation belongs to item 13, recognizes exactly two nested
fenced blocks, and verifies v28 precedes v29.

## Test-first evidence

The initial focused run reported:

```text
4 failed, 1 passed in 0.11s
```

The sibling control passed while the ordering, equal-path, normalized
nested-path, and CommonMark regressions failed for the expected reasons. The
separately isolated converse ancestor case reported:

```text
1 failed in 0.07s
```

After the bounded corrections and formatting, the consolidated focused set
reported:

```text
7 passed in 0.10s
```

The complete changed and adjacent owners reported:

```text
168 passed in 234.25s (0:03:54)
```

The exact four-file Task 7 suite reported:

```text
230 passed, 1 skipped in 1020.79s (0:17:00)
```

Production, tests, and operator documentation remained frozen throughout the
exact run.

## Gates and boundaries

Before the exact suite, all active Python commands returned successful help,
all shell and R scripts parsed, every study and method-attempt JSON document
parsed, the historical archive count and non-runtime import boundary held, and
the repository retained no gitlinks, active submodules, or `.gitmodules` file.
Ruff formatting and lint, scoped Python compilation, and `git diff --check`
also passed. The same gates passed after recording the terminal evidence.

The correction preserves the public CLI argument surface, sibling receipt
layout, source and environment contracts, archive and migration boundaries,
scientific populations, selection decisions, seed policies, estimands,
revision permissions, and claim permissions. No real scientific workload ran.
