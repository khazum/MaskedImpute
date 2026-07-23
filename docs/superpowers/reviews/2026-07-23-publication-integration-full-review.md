# Publication Integration Full Review

## Scope and constraints

This ledger covers the full `codex/publication-integration` branch relative to
`main`, beginning at branch head
`391efcb3bf1b1b07a2bb159fade049528fad149b`. Review work is restricted to
`.worktrees/publication-integration`.

The review will not perform cyber-related work, add or reinstate content
fingerprints, add parent-directory race hardening, remove legitimate outer
provenance, run the real scientific workload, claim empirical superiority, or
invent human declarations, identifiers, results, figures, licenses, or
submission authorization. Bounded collection, static analysis, tests, dry
runs, planning, and manuscript compilation are permitted.

## Baseline

The worktree was clean before this report was created. The exact identity
commands and outputs were:

```text
$ pwd
/home/marcinmaleclocal/Coding/MaskedImpute/.worktrees/publication-integration
$ git status --short --branch
## codex/publication-integration
$ git branch --show-current
codex/publication-integration
$ git merge-base main HEAD
3813a7e4a665b6e2a5c6e33d53e45c4290eb7f1f
$ git rev-parse HEAD
391efcb3bf1b1b07a2bb159fade049528fad149b
$ git rev-list --count main..HEAD
278
```

The exact path inventories were created successfully with:

```text
$ git -c diff.renameLimit=20000 diff --name-status main...HEAD > /tmp/publication-integration-name-status.txt
$ git -c diff.renameLimit=20000 diff --numstat main...HEAD > /tmp/publication-integration-numstat.txt
```

Both commands exited zero and each inventory contains exactly 7,336 records.
The canonical name-status inventory contains 246 added paths, 3 modified paths,
6,034 deleted paths, and 1,053 paths renamed at 100% similarity. The numstat
inventory records 207,365 added lines, 1,764,787 deleted lines, and 319 binary
paths.

`git diff --summary main...HEAD` exited zero and emitted 7,333 summary records.
It also emitted these exact diagnostics:

```text
warning: exhaustive rename detection was skipped due to too many files.
warning: you may want to set your diff.renameLimit variable to at least 6034 and retry the command.
```

Because the required high-limit name-status command completed successfully, its
7,336 records are the authoritative changed-path inventory; the summary command
is retained only as supplemental mode-change and migration evidence.

The remaining Step 2 commands exited zero with empty output:

```text
$ git submodule status
$ git ls-files -d
$ git ls-files -o --exclude-standard
```

Thus the branch head has no active submodules, and the initial worktree had no
tracked-but-missing or untracked non-ignored paths. Inspection also established
that `.gitmodules` is absent at branch head, while `main` contains entries for
`AutoClass` and `MAGIC`; the six gitlinks `AutoClass`, `MAGIC`, `SAVER`,
`rds2py`, `splatter`, and `temp/single-cell-3prime-paper` are deleted by the
branch and remain structural-review items.

Repository metadata inspected in this task:

- `.gitignore`: present; excludes Python/R caches, local environments, build
  artifacts, data outputs, and `.worktrees/`.
- `.gitattributes`: present; classifies the two Springer Nature template files
  as binary.
- `.gitmodules`: deleted on the branch; no active submodule status entries.
- `pyproject.toml`: present; requires Python 3.10 or later, defines runtime and
  test dependencies, and limits pytest discovery to `tests`.

The invoking shell initially resolved `python` to Python 3.13.9 without the
declared test tools. An initial collection attempt in the existing
`masked_imputation` environment reproduced 28 import errors: its enabled Python
3.10 user site placed NumPy 2.1.2 ahead of the environment's NumPy 1.26.4 and
loaded pandas 2.0.3 binaries built for the older ABI. With user-site packages
disabled, the environment's NumPy 1.26.4 and pandas 2.1.4 import together, but
that environment does not contain `anndata`.

The authoritative checks therefore use the existing `magic311` environment
with `PYTHONNOUSERSITE=1`. It provides Python 3.11.14, pytest 8.4.2, the
project's declared `anndata` range, and Torch. Ruff 0.14.4 was installed only
into `/tmp/publication-integration-review-tools-task1` and exposed through
`PYTHONPATH`; neither repository content nor an existing environment was
changed.

## Changed-path coverage

| Scope | Total | Added | Modified | Deleted | Exact rename | Planned review |
|---|---:|---:|---:|---:|---:|---|
| repository root | 10 | 2 | 2 | 6 | 0 | Tasks 2 and 7 |
| `.superpowers` | 8 | 8 | 0 | 0 | 0 | Task 7 |
| `.venv_scvi` | 6,019 | 0 | 0 | 6,019 | 0 | Task 7 |
| `DenseLayerPack` | 8 | 0 | 0 | 8 | 0 | Task 7 |
| `docs` | 48 | 48 | 0 | 0 | 0 | Tasks 7 and 9 |
| `environments` | 4 | 4 | 0 | 0 | 0 | Task 5 |
| `historical` | 1,056 | 3 | 0 | 0 | 1,053 | Task 7 |
| `maskimpute` | 13 | 13 | 0 | 0 | 0 | Task 2 |
| `maskimpute_benchmark` | 61 | 61 | 0 | 0 | 0 | Tasks 3–6 |
| `paper` | 7 | 6 | 1 | 0 | 0 | Task 9 |
| `scripts` | 32 | 32 | 0 | 0 | 0 | Tasks 3–6 |
| `study` | 17 | 17 | 0 | 0 | 0 | Tasks 3–6 |
| `temp` | 1 | 0 | 0 | 1 | 0 | Task 7 |
| `tests` | 52 | 52 | 0 | 0 | 0 | Tasks 2–7 |
| **Total** | **7,336** | **246** | **3** | **6,034** | **1,053** | **Tasks 2–9** |

Every changed path is assigned to a later manual or automated review scope.
Task 10 owns final whole-branch verification and disposition.

## Findings

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| O-001 | Minor observation | The invoking shell lacks the declared test tools; the first alternate environment leaked incompatible user-site NumPy/pandas packages. | Diagnosed as environment selection rather than a repository defect. Authoritative collection passed in isolated Python 3.11. Recheck the intended release environment in Task 10. |
| F-001 | Minor | Repository-wide Ruff lint exits 1 with 261 violations: 259 in `historical/v26_neurips` and 2 in unchanged `DenseLayerPack` source. | Open for Task 8. The active integration packages, scripts, and tests produced no lint violation. Diagnose whether archival/legacy scope should be explicitly excluded or made compliant; no baseline-task fix. |
| F-002 | Minor | Ruff formatting exits 1: 59 files would be reformatted (53 historical files, `masked_imputation26.py`, and five `maskimpute` modules). | Open for Task 8. Diagnose and correct the active formatting surface without rewriting preserved history unless review policy explicitly requires it; no baseline-task fix. |

### Critical

None identified by the Task 1 baseline and static checks.

### Important

None identified by the Task 1 baseline and static checks.

### Minor

- O-001: environment-selection variance described in the findings table. No
  repository correction is warranted in this task.
- F-001: the repository-wide Ruff lint gate is not clean.
- F-002: the repository-wide Ruff formatting gate is not clean.

### Human and scientific blockers

- Empirical competitiveness remains unknown until the separately authorized
  real benchmark, frozen analysis, and review workflow is completed.
- Author names, affiliations, contributions, funding, ethics determinations,
  competing interests, acknowledgements, licenses, repository/archive
  identifiers, dataset accessions, final figures, and numerical results require
  human or scientific authority.
- Genome Biology submission and release readiness require human authorization.

## Corrections

Task 1 creates only this review ledger. It makes no production, test,
manuscript, dependency, or environment correction.

## Verification ledger

| Command | Environment | Result |
|---|---|---|
| `python -m pytest --collect-only -q` | `masked_imputation`, Python 3.10.19 | Exit 2: 1,065 tests collected before 28 imports failed with one NumPy/pandas ABI mismatch; diagnosed as user-site leakage. |
| `python -m pytest --collect-only -q` | isolated `magic311`, Python 3.11.14, pytest 8.4.2 | Exit 0: 2,782 tests collected in 30.10 seconds. |
| `python -m ruff check .` | isolated `magic311`, Ruff 0.14.4 | Exit 1: 261 violations across 39 files; 259 violations are historical and 2 are in unchanged `DenseLayerPack` source. Finding F-001. |
| `python -m ruff format --check .` | isolated `magic311`, Ruff 0.14.4 | Exit 1: 59 files would be reformatted and 158 are already formatted; 53 affected files are historical and 6 are active. Finding F-002. |
| `python -m compileall -q maskimpute maskimpute_benchmark scripts tests` | isolated `magic311`, Python 3.11.14 | Exit 0 with empty output. |
| `git diff --check main...HEAD` | worktree | Exit 0 with no whitespace errors; Git repeated the rename-limit warning recorded above. |
| `git -c diff.renameLimit=20000 diff --check main...HEAD` | worktree | Supplemental exit 0 with empty output. |

## Final disposition

Task 1 established the exact branch baseline and changed-path ledger. Test
collection, Python compilation, and whitespace checks pass in the diagnosed
environment. Findings F-001 and F-002 remain open for Task 8, so the branch
does not yet satisfy its complete static-check acceptance criteria.

No conclusion about whole-branch correctness, scientific competitiveness, or
Genome Biology submission readiness is made here.
