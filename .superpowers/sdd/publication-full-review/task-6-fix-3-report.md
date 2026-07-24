# Task 6 v3 final-acceptance correction report

## Status

Complete.

Correction base:

```text
df6c3b16820c28ba2397590be08f7251997ea4f5
```

## Scope

This correction completes F-079 across the reachable scaling, final,
trajectory, downstream, final-analysis, null-DE, and publication-synthesis
boundaries. It also requires nonempty, complete publication plan populations
at the public final and trajectory evaluation entries.

The correction does not change selected methods, final datasets, the fixed
1,760 primary-run population, the fixed 44 trajectory-run population, seeds,
revision allowances, estimands, execution behavior, or claim permissions. No
real scientific or comparator workload ran.

## Root cause and correction

Python Booleans are integer subclasses, so ordinary equality admitted `True`
as one and `False` as zero. Several reachable authority consumers compared
schema, count, or ordinal fields without first requiring the exact built-in
integer type.

The affected owners now reject those aliases before equality or ordering
checks. Scaling and trajectory plan payloads also require nonempty, contiguous
typed entries. Public final and trajectory evaluation independently require:

- exact typed plan, entry, run, and frozen-configuration objects;
- nonempty configurations and entries;
- unique method authorities;
- complete method-by-dataset-by-seed Cartesian membership;
- exact sequential integer ordinals;
- entry configuration identity, action, and reason agreement;
- the established 40-dataset, 1,760-run primary population and 44-run
  trajectory population.

Internal replay paths that first rebuild and validate the frozen typed plan use
the existing structural terminal-record validator. This keeps synthetic unit
fixtures valid without weakening either public production entry.

## Test-first evidence

The direct payload, loader, and population RED set reported:

```text
13 failed in 186.27s (0:03:06)
```

Seven additional adapter mutations demonstrated Boolean acceptance while
nineteen nearby controls already rejected their mutations. Corrected focused
sets reported:

```text
13 passed in 166.61s (0:02:46)
11 passed, 19 deselected in 2.15s
8 passed in 2.86s
```

Complete owner and compatibility suites reported:

```text
tests/test_downstream_evidence.py: 170 passed in 93.77s
tests/test_final_analysis.py: 53 passed in 382.85s (0:06:22)
tests/test_final_runner.py: 131 passed in 782.17s (0:13:02)
tests/test_scaling_panel.py: 66 passed in 2940.99s (0:49:00)
publication-synthesis and final-null-DE: 72 passed in 80.91s (0:01:20)
freeze and revision compatibility: 139 passed in 2314.41s (0:38:34)
```

The first exact run exposed redundant use of the strict public population gate
inside independently validated internal replays:

```text
57 failed, 844 passed in 7652.31s (2:07:32)
```

After correcting only those internal callers, all prior failures and the full
affected owners passed:

```text
53 passed, 376 deselected in 159.04s (0:02:39)
363 passed in 1246.67s (0:20:46)
```

Ruff formatting and lint, byte compilation, and `git diff --check` exited zero.
The second clean exact fifteen-file Task 6 suite then reported:

```text
901 passed in 7766.96s (2:09:26)
```

No production or test file changed during that successful exact run.

## Disposition

F-079 is closed across the remaining reachable integer authority boundaries,
and public publication evaluation now rejects empty, sliced, duplicated, or
expanded plan populations. This bounded contract correction does not itself
establish empirical competitiveness or Genome Biology submission readiness.
