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

The exact raw path inventories are retained as tracked review artifacts:

- [baseline name-status inventory](2026-07-23-publication-integration-baseline-name-status.txt)
- [baseline numstat inventory](2026-07-23-publication-integration-baseline-numstat.txt)

They were generated directly by Git from the fixed merge base through baseline
commit `391efcb3bf1b1b07a2bb159fade049528fad149b`:

```text
$ git -c diff.renameLimit=20000 diff --name-status 3813a7e4a665b6e2a5c6e33d53e45c4290eb7f1f..391efcb3bf1b1b07a2bb159fade049528fad149b > docs/superpowers/reviews/2026-07-23-publication-integration-baseline-name-status.txt
$ git -c diff.renameLimit=20000 diff --numstat 3813a7e4a665b6e2a5c6e33d53e45c4290eb7f1f..391efcb3bf1b1b07a2bb159fade049528fad149b > docs/superpowers/reviews/2026-07-23-publication-integration-baseline-numstat.txt
```

Both commands exited zero. Each retained inventory contains the original 7,336
records.
The canonical name-status inventory contains 246 added paths, 3 modified paths,
6,034 deleted paths, and 1,053 paths renamed at 100% similarity. The numstat
inventory records 207,365 added lines, 1,764,787 deleted lines, and 319 binary
paths.

One prohibited checksum diagnostic was run transiently against the temporary
inventories during the original Task 1 investigation. This is closed as
procedural deviation P-001: every resulting value was discarded immediately,
none was used as review evidence, and no value or artifact was retained.

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

The authoritative checks therefore use the existing `magic311` environment.
Every isolated Python invocation in the verification ledger spells out this
exact prefix:

```text
PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m
```

It provides Python 3.11.14, pytest 8.4.2, the project's declared `anndata`
range, and Torch. Ruff 0.14.4 was installed only into
`/tmp/publication-integration-review-tools-task1` and exposed through
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
| O-001 | Minor | The invoking shell lacks the declared test tools; the first alternate environment leaked incompatible user-site NumPy/pandas packages. | Diagnosed as environment selection rather than a repository defect. Authoritative collection passed in isolated Python 3.11. Recheck the intended release environment in Task 10. |
| F-001 | Important | Repository-wide Ruff lint exited 1 with 261 violations: 259 in preserved `historical/v26_neurips` material and 2 in active `DenseLayerPack` source. | Closed before deeper review. `pyproject.toml` excludes only `historical`; the unused DenseLayerPack import and dead local computation were removed. Repository-wide Ruff lint now exits zero. |
| F-002 | Important | Ruff formatting exited 1: 59 files would be reformatted (53 historical files and six active files). | Closed before deeper review. Only `masked_imputation26.py` and the five identified `maskimpute` modules were mechanically formatted; preserved history was not rewritten. Repository-wide Ruff format checking now exits zero. |
| F-003 | Minor | Once tracked, the exact raw inventories caused the active-terminology hygiene test to interpret archived path names as active prose. | Closed test-first. The two exact evidence artifacts are excluded by exact path from that terminology scan; their raw content remains unchanged. |
| P-001 | Minor | A prohibited transient checksum diagnostic was run against temporary inventories during the original baseline investigation. | Closed procedural deviation. Resulting values were discarded, never used, and no value or artifact was retained. The authoritative evidence is the two tracked raw Git inventories linked above. |

### Critical

None identified by the Task 1 baseline and static checks.

### Important

- F-001 and F-002 blocked the repository-wide static gate. Both are closed by
  the corrections and exact verification recorded below.

### Minor

- O-001: environment-selection variance described in the findings table. No
  repository correction is warranted in this task.
- F-003: the exact-path terminology-test exclusion preserves both the hygiene
  contract and the raw inventory evidence.
- P-001: the transient procedural deviation is recorded and closed with no
  retained value or artifact.

### Human and scientific blockers

- Empirical competitiveness remains unknown until the separately authorized
  real benchmark, frozen analysis, and review workflow is completed.
- Author names, affiliations, contributions, funding, ethics determinations,
  competing interests, acknowledgements, licenses, repository/archive
  identifiers, dataset accessions, final figures, and numerical results require
  human or scientific authority.
- Genome Biology submission and release readiness require human authorization.

## Corrections

- Retained the exact raw baseline name-status and numstat inventories as the
  two linked tracked review artifacts.
- Excluded only preserved `historical` material from repository-wide Ruff
  discovery after isolating all archival lint and formatting noise there.
- Removed the unused `torch` import from `DenseLayerPack/DenseLayer.py` and the
  dead `base_output` computation plus now-unused functional import from
  `DenseLayerPack/WavKAN.py`.
- Mechanically formatted only `masked_imputation26.py`,
  `maskimpute/calibration.py`, `maskimpute/impute.py`,
  `maskimpute/nb_model.py`, `maskimpute/structure.py`, and
  `maskimpute/train.py`.
- Corrected the full-review plan so baseline failures route to Task 9 in the
  current 10-task plan.
- Excluded the two raw inventory artifacts by exact path from the active
  terminology scan after the newly tracked files produced the expected RED
  hygiene failure.
- Closed F-001, F-002, F-003, and P-001 before deeper review. No historical
  content, manuscript, dependency, existing environment, scientific result, or
  human metadata was changed.

## Verification ledger

| Exact command | Phase | Result |
|---|---|---|
| `/home/marcinmaleclocal/miniconda3/envs/masked_imputation/bin/python -m pytest --collect-only -q` | Baseline diagnosis | Exit 2: 1,065 tests collected before 28 imports failed with one NumPy/pandas ABI mismatch caused by enabled user-site packages. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest --collect-only -q` | Baseline at `391efcb3bf1b1b07a2bb159fade049528fad149b` | Exit 0: 2,782 tests collected in 30.10 seconds. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m ruff check .` | Baseline at `391efcb3bf1b1b07a2bb159fade049528fad149b` | Exit 1: 261 violations; 259 were in preserved history and 2 were in active DenseLayerPack source. Finding F-001. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m ruff format --check .` | Baseline at `391efcb3bf1b1b07a2bb159fade049528fad149b` | Exit 1: 59 files would be reformatted; 53 were historical and 6 were active. Finding F-002. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m ruff format masked_imputation26.py maskimpute/calibration.py maskimpute/impute.py maskimpute/nb_model.py maskimpute/structure.py maskimpute/train.py` | Correction | Exit 0: exactly six active files reformatted. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m ruff check .` | Correction verification | Exit 0: all checks passed. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m ruff format --check .` | Correction verification | Exit 0: 164 files already formatted. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m compileall -q DenseLayerPack masked_imputation26.py maskimpute maskimpute_benchmark scripts tests` | Correction verification | Exit 0 with empty output. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest -q tests/test_obsolete_terms.py tests/test_maskimpute_v28.py tests/test_maskimpute_v29.py tests/test_prezero_calibration.py tests/test_maskimpute_v27.py::test_public_imputation_signature_has_no_evaluator_side_channels tests/test_maskimpute_v27.py::test_inference_marks_exactly_natural_observed_zeros_unavailable tests/test_maskimpute_v27.py::test_power_complement_gate_is_monotone_in_pre_zero_probability tests/test_maskimpute_v27.py::test_power_complement_gate_rejects_ambiguous_direct_inputs` | Correction verification | Exit 0: 151 focused tests passed in 6.96 seconds. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest -q tests/test_obsolete_terms.py tests/test_maskimpute_v28.py tests/test_maskimpute_v29.py tests/test_prezero_calibration.py tests/test_maskimpute_v27.py::test_public_imputation_signature_has_no_evaluator_side_channels tests/test_maskimpute_v27.py::test_inference_marks_exactly_natural_observed_zeros_unavailable tests/test_maskimpute_v27.py::test_power_complement_gate_is_monotone_in_pre_zero_probability tests/test_maskimpute_v27.py::test_power_complement_gate_rejects_ambiguous_direct_inputs` | F-003 RED at the first committed head | Exit 1: 1 failed and 150 passed in 6.96 seconds; both exact evidence artifacts contained archived path names. |
| `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/publication-integration-review-tools-task1 /home/marcinmaleclocal/miniconda3/envs/magic311/bin/python -m pytest -q tests/test_obsolete_terms.py` | F-003 GREEN | Exit 0: 2 tests passed in 0.76 seconds. |
| `git diff --check` | Correction verification | Exit 0 with empty output. |

## Final disposition

Task 1 established the exact branch baseline and changed-path ledger, now with
the exact raw inventories retained as tracked artifacts. Test collection,
Python compilation, repository-wide Ruff lint and format checks, and whitespace
checks pass in the diagnosed environment. F-001, F-002, F-003, and P-001 are
closed; deeper review may proceed under Tasks 2–10.

No conclusion about whole-branch correctness, scientific competitiveness, or
Genome Biology submission readiness is made here.
