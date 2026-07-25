# Task 7 second independent-review correction report

## Status

Implementation and self-verification are complete. Independent acceptance is
not claimed.

## Corrections

### F-101: CLI imports changed process-global bytecode state

Importing `scripts/run_external_reference_development.py`,
`scripts/freeze_publication_round.py`, or `scripts/studyctl.py` unconditionally
set `sys.dont_write_bytecode` to true. The assignment escaped the imported
module and changed later in-process behavior.

Each command now enables the setting at module-load time only when Python is
executing that file as the command entry point. This retains the existing
no-bytecode behavior before production imports during direct execution.
Importing any of the three files under a module name preserves both possible
initial states.

Each public `main(argv)` also scopes the setting around command execution and
restores the caller's prior state in a `finally` block. Direct entry-point
execution restores the state that existed before the script began, including
when command execution exits exceptionally. Public arguments, return statuses,
and structured error behavior are unchanged.

### F-102: Task 7 shell syntax command did not discover files safely

The Task 7 plan passed two shell glob classes to one `bash -n` invocation. With
no matching file, Bash attempted to open the literal glob and returned status
127. With multiple expanded paths, Bash parsed only the first path as its
script and treated the remainder as positional arguments.

The plan now discovers regular shell files under both active script roots with
null-delimited paths and invokes `bash -n` once for each discovered file. The
loop succeeds when no optional class contains a shell file and propagates any
individual syntax failure. A regression extracts and executes the literal
documented command against an empty tree, a valid script, and a malformed
script in the optional simulator class.

## Test-first evidence

The focused RED run reported:

```text
7 failed, 3 passed, 43 deselected in 0.73s
```

The three false-state imports, all three scoped command calls, and the empty
shell discovery failed for the expected reasons. The three true-state import
controls passed.

After the bounded corrections, the focused GREEN run reported:

```text
10 passed, 43 deselected in 0.84s
```

The post-format focused rerun reported:

```text
10 passed, 43 deselected in 0.80s
```

The complete external-reference, publication-freeze, study-lifecycle, and
repository-hygiene owners reported:

```text
293 passed, 1 skipped in 2124.57s (0:35:24)
```

The exact four-file Task 7 suite reported:

```text
240 passed, 1 skipped in 1002.41s (0:16:42)
```

Production, tests, and the Task 7 plan remained frozen throughout both complete
test runs.

## Gates and boundaries

Before the exact suite:

- All 29 active Python commands returned successful help without scientific
  execution.
- The active shell script and both R simulator drivers parsed.
- All 17 study and method-attempt JSON documents parsed.
- The 1,056-file historical archive and its non-runtime import boundary held.
- `.gitmodules`, gitlinks, and active submodules remained absent.
- Ruff formatting and lint passed across all 164 Python files.
- Scoped Python compilation and `git diff --check` passed.

The correction does not change command arguments, scientific dependencies,
selected methods, configurations, dataset populations, seed policies,
estimands, revision allowances, archive contents, or claim permissions. No
real scientific workload ran.
