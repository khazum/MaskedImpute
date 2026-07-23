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

## Task 2 model and numerical correctness audit

### Environment and baseline

Task 2 started from clean commit
`36f4ca09a05b1c79df12759929fcd8e7ff6a7c03` in the linked
`publication-integration` worktree. No dependency was installed or changed.

The first focused-suite attempt used the existing `masked_imputation`
interpreter and stopped during collection. Its enabled Python 3.10 user site
loaded an `anndata`/`pandas` binary incompatible with the user-site NumPy ABI.
Disabling the user site exposed that environment's NumPy 1.26.4, below the
declared NumPy 2 floor, and removed pytest and `anndata`. This was classified as
an environment failure rather than a model defect.

The authoritative Task 2 interpreter was the existing
`/tmp/maskimpute-scziva312/bin/python`: Python 3.12.7, NumPy 2.2.6, SciPy
1.13.1, Torch 2.4.1, and pytest 8.4.2. It satisfies the declared Python,
NumPy, SciPy, Torch, and pytest ranges. The unmodified focused baseline was:

```text
572 passed, 2 skipped in 24.65s
```

### Path and invariant coverage

Every production path assigned by the Task 2 brief was inspected end to end:

| Path | Reviewed model/numerical responsibilities | Disposition |
|---|---|---|
| `maskimpute/__init__.py` | Public exports and lazy Torch-backed API loading | Valid; no change |
| `maskimpute/ablations.py` | Prespecified component changes, capacity parity, score resolution, gating, selective/full output, inference mode | F-005 finite-output correction and shared F-010/F-011 validation-mask coercion closed |
| `maskimpute/calibration.py` | Probability/target/weight domains, monotone transforms, stable metrics, immutable retained artifact | Valid; no change |
| `maskimpute/config.py` | Immutable finite hyperparameters, positive dimensions, bounded seed | Valid; no change |
| `maskimpute/count_model.py` | Raw-count validation, dense/sparse parity, deterministic cross-fitting, stable NB zero probability, immutable score arrays | Valid; F-011 consolidated its dense path onto the shared coercion boundary without changing the sparse path |
| `maskimpute/impute.py` | Input-to-result flow, calibrated score support, eval/no-grad inference, finite outputs, observed-positive preservation | Valid; no change |
| `maskimpute/model.py` | Explicit availability semantics, learned mask token, aligned shapes/devices, nonnegative decoder | Valid; no change |
| `maskimpute/nb_model.py` | Float64 NB likelihood, dispersion estimation, library-size offset, zero-library behavior | F-006 and shared F-010/F-011 mask coercion closed |
| `maskimpute/prezero.py` | Stable NB/Poisson Bayes posterior, broadcast/support validation, observed-positive zero score | Valid; no change |
| `maskimpute/result.py` | Shape/domain validation and defensive immutable dense/sparse/diagnostic snapshots | Valid; no change |
| `maskimpute/sparse_input.py` | Exact supported sparse storage, coordinate bounds/uniqueness, lossless snapshotting, shared dense mask-preserving coercion | Sparse path remains valid and unchanged; F-010 introduced the dense helper and F-011 made its nested protocol traversal cycle-safe and single-snapshot |
| `maskimpute/structure.py` | Observed-only variable-gene/neighborhood authority and finite differentiable penalties | Valid; no change |
| `maskimpute/train.py` | Dense/sparse conversion, normalization/inverse, mask construction, seed propagation, RNG restoration, train/eval transitions | F-004, F-005, and F-007 through F-011 closed |
| `masked_imputation26.py` | Migration-only legacy wrapper that does not execute archived code | Valid; no change |

Every assigned test path was also inspected and executed:

| Path | Coverage classification |
|---|---|
| `tests/test_count_model.py` | Count-model domains, dense/sparse equivalence, array-protocol single-snapshot behavior, cross-fit invariants, stability, immutability |
| `tests/test_maskimpute_ablations.py` | Capacity/mask/gate/output controls and verified score/calibration execution; F-005, F-010, and F-011 regressions |
| `tests/test_maskimpute_api.py` | Config/result/pre-zero shape, support, sparse, dtype, finite, and immutability contracts |
| `tests/test_maskimpute_v27.py` | Normalization, masks, seeds, RNG scope, train/eval behavior, zero libraries, public result invariants; F-004, F-007 through F-011 regressions |
| `tests/test_maskimpute_v28.py` | NB likelihood oracle, float64/device contract, dispersion, zero-library gradients; F-006, F-010, and F-011 regressions |
| `tests/test_maskimpute_v29.py` | Observed-only structure authority, differentiability, fixed validation exclusion |
| `tests/test_prezero_calibration.py` | Monotone calibration, stable metrics, retention semantics, immutable artifact |
| `tests/test_prezero_evidence.py` | Realized-score evidence binding, defensive matrices, checkpoint/recovery dispositions |

The requested cross-cutting classifications are:

| Invariant | Evidence and disposition |
|---|---|
| Shape validation | Two-dimensional/nonempty count and score boundaries, aligned Torch shapes, result row/shape contracts; covered and valid |
| Sparse/dense parity | Shared coordinate snapshot boundary and equivalent count/score behavior; covered and valid |
| Dtype/device changes | Exact numeric input checks, explicit float32 model tensors, float64 NB likelihood, aligned Torch devices, post-cast finite checks; F-005, F-008, and F-009 closed |
| Finite values | Checked at external arrays and scalars, fitted parameters, losses, decoder/latent/count outputs, and result construction; F-004, F-005, F-008, and F-009 closed |
| Zero library sizes | Preserved as all-zero normalized/count rows and excluded safely from exposure gradients; covered and valid |
| Mask meaning | Observed positives are available, natural zeros unavailable, validation/training masks select observed positives only, unavailable payload is irrelevant; F-006/F-007 cover original direct masked vectors, F-010 covers root protocol-produced masks, and F-011 closes nested protocol-produced masks at every demonstrated dense model boundary |
| Seed propagation | Bounded config seed, independent NumPy seed streams, scoped Torch deterministic state, caller RNG restoration; covered and valid |
| Train/eval transitions | Training mode per epoch, eval/no-grad validation and inference, best checkpoint restored in eval mode; covered and valid |
| Result immutability | Dense, sparse, latent, probability, and nested diagnostics are snapshotted and freshly materialized read-only views; covered and valid |

### Task 2 findings and dispositions

| ID | Severity | Finding | Root cause | Disposition |
|---|---|---|---|---|
| O-002 | Minor | The first Task 2 interpreter stopped in collection with a NumPy/pandas ABI mismatch and proved below the declared NumPy floor when isolated. | Enabled user-site packages overrode an older environment and mixed incompatible binaries. | Closed as environmental; authoritative tests use the existing declared-compatible Python 3.12 environment. |
| F-004 | Important | Both observed-count and corrupted-availability normalization could return `inf` for a valid finite normalization target. | Computing `target / library` first can round upward; multiplying the result by the count then overflowed even though `count / library * target` is mathematically bounded by `target`. | Closed test-first by evaluating proportions before multiplying by the target. No hyperparameter or estimand changed. |
| F-005 | Important | A finite extended-precision candidate outside float64 range could be cast to `inf` and returned by the full-ungated ablation output. | `_numeric_matrix_to_dense` checked finiteness only before its float64 conversion. | Closed test-first with a post-conversion finite check at the shared dense boundary. |
| F-006 | Important | NB dispersion estimation silently accepted a directly masked `library_sizes` vector. | `np.asarray` discarded the mask before validation. | The original direct-input regression remains closed. F-010 supersedes its boundary-local guard with shared mask-preserving coercion and separately covers protocol-produced masks. |
| F-007 | Important | Inverse normalization silently accepted a directly masked `library_sizes` vector. | `invert_observed_normalization` called `np.asarray` before checking mask semantics, which erased the mask. | The original direct-input regression remains closed. F-010 supersedes its boundary-local guard with shared mask-preserving coercion and separately covers both inverse inputs. |
| F-008 | Minor | Both direct forward-normalization helpers accepted a finite extended-precision target outside float64 range and returned `inf`. | The helpers checked finiteness before converting the scalar to float64, but did not validate the converted scalar before arithmetic. | Closed test-first with a post-conversion finite check before normalization arithmetic; the valid float64 maximum remains accepted. |
| F-009 | Minor | Inverse normalization accepted a finite extended-precision target outside float64 range after it converted to `inf`. | `invert_observed_normalization` validated the original scalar but used `float(target)` without validating that conversion. | Closed test-first with the same post-conversion finite-and-positive guard used at all three normalization boundaries. |
| F-010 | Important | Root dense array-like inputs whose array protocol produced masked data bypassed boundary-local direct-mask guards; inverse normalized expression also accepted a direct masked array. Reachable affected inputs were inverse normalized expression and library sizes, dispersion library sizes and estimation mask, and epoch/ablation validation masks. | Ordinary `np.asarray` coercion discarded `MaskedArray` semantics before the owning boundary validated them. | Closed test-first for direct and root protocol inputs with one private shared coercion boundary. Its original claim to cover arbitrary nested protocol objects was incorrect and is superseded by F-011. |
| F-011 | Important | Protocol objects nested in accepted Python sequences could return masked rows that were erased by the outer NumPy conversion. Reachable acceptance was demonstrated for observed counts, available-input normalization, inverse normalized expression, dispersion estimation masks, and epoch/ablation validation masks. | The F-010 pre-scan recursed into containers but stopped at arbitrary nested protocol objects. A first attempt that merely inspected each protocol would also have invoked stateful objects again during outer coercion. | Closed test-first by making the one private recursive boundary replace every nested protocol object with one validated ordinary-ndarray snapshot, reuse repeated-object snapshots, reject masks and cycles before outer coercion, and leave root ndarray/protocol and exact sparse paths unchanged. |

### Regression evidence

F-004 through F-006 were run alone during their original TDD cycles. The
original F-007/F-008 fix wave instead ran its three RED nodes together, so the
earlier ledger statement that every regression had run alone before correction
was inaccurate. Isolated RED evidence for those three nodes was reconstructed
later in a detached checkout of pre-fix production commit `93105ac`, with only
the 41-line test addition from `4728945` applied. `git diff --name-only` listed
only `tests/test_maskimpute_v27.py`, and each node then failed alone because
the expected exception was not raised. F-009 used a conventional isolated
RED/GREEN cycle at the integration head. F-010 and F-011 also used
conventional isolated RED/GREEN cycles for every defect-regression node.

| Pytest node | Isolated RED provenance and result | Isolated GREEN |
|---|---|---|
| `tests/test_maskimpute_v27.py::test_observed_normalization_keeps_maximum_finite_target_finite` | Original TDD: actual `inf`, expected finite `log1p(target)` | 1 passed |
| `tests/test_maskimpute_v27.py::test_available_normalization_keeps_maximum_finite_target_finite` | Original TDD: actual `inf`, expected finite `log1p(target)` | 1 passed |
| `tests/test_maskimpute_ablations.py::test_ablation_output_rejects_finite_candidates_not_representable_in_float64` | Original TDD: did not raise and returned post-cast `inf` | 1 passed |
| `tests/test_maskimpute_v28.py::test_gene_dispersion_rejects_masked_library_sizes` | Original TDD: did not raise after mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_inverse_normalization_rejects_masked_library_sizes_before_coercion` | Reconstructed at `93105ac`: did not raise after mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_observed_normalization_rejects_finite_target_outside_float64_range` | Reconstructed at `93105ac`: did not raise after float64 overflow | 1 passed |
| `tests/test_maskimpute_v27.py::test_available_normalization_rejects_finite_target_outside_float64_range` | Reconstructed at `93105ac`: did not raise after float64 overflow | 1 passed |
| `tests/test_maskimpute_v27.py::test_inverse_normalization_rejects_finite_target_outside_float64_range` | Original TDD at `4728945`: did not raise after float64 overflow | 1 passed |
| `tests/test_maskimpute_v27.py::test_inverse_normalization_rejects_masked_normalized_expression` | Original TDD at the F-010 parent: did not raise after direct mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_inverse_normalization_rejects_protocol_masked_normalized_expression` | Original TDD at the F-010 parent: did not raise after protocol-produced mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_inverse_normalization_rejects_protocol_masked_library_sizes` | Original TDD at the F-010 parent: did not raise after protocol-produced mask erasure | 1 passed |
| `tests/test_maskimpute_v28.py::test_gene_dispersion_rejects_protocol_masked_library_sizes` | Original TDD at the F-010 parent: did not raise after protocol-produced mask erasure | 1 passed |
| `tests/test_maskimpute_v28.py::test_gene_dispersion_rejects_protocol_masked_estimation_mask` | Original TDD at the F-010 parent: did not raise after protocol-produced mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_epoch_mask_rejects_protocol_masked_validation_mask` | Original TDD at the F-010 parent: did not raise after protocol-produced mask erasure | 1 passed |
| `tests/test_maskimpute_ablations.py::test_uniform_masking_rejects_protocol_masked_validation_mask` | Original TDD at the F-010 parent: did not raise after protocol-produced mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_observed_counts_reject_nested_protocol_masked_row` | Original TDD at the F-011 parent: did not raise after nested protocol mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_available_normalization_rejects_nested_protocol_masked_row` | Original TDD at the F-011 parent: did not raise after nested protocol mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_inverse_normalization_rejects_nested_protocol_masked_row` | Original TDD at the F-011 parent: did not raise after nested protocol mask erasure | 1 passed |
| `tests/test_maskimpute_v28.py::test_gene_dispersion_rejects_nested_protocol_masked_estimation_row` | Original TDD at the F-011 parent: did not raise after nested protocol mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_epoch_mask_rejects_nested_protocol_masked_validation_row` | Original TDD at the F-011 parent: did not raise after nested protocol mask erasure | 1 passed |
| `tests/test_maskimpute_ablations.py::test_uniform_masking_rejects_nested_protocol_masked_validation_row` | Original TDD at the F-011 parent: did not raise after nested protocol mask erasure | 1 passed |
| `tests/test_maskimpute_v27.py::test_observed_counts_snapshot_nested_stateful_protocol_once` | Original TDD at the F-011 parent: a second protocol call supplied different invalid counts | 1 passed with one protocol call |
| `tests/test_maskimpute_v27.py::test_observed_counts_reject_cyclic_nested_sequence_before_coercion` | Original TDD at the F-011 parent: NumPy reached its 64-dimension recursion bound instead of boundary rejection | 1 passed with bounded pre-coercion rejection |

The complete post-correction Task 2 suite passed:

```text
576 passed, 2 skipped in 29.14s
```

After the Task 2 review fix wave and the bounded in-place algebra refactor, the
same complete suite passed:

```text
579 passed, 2 skipped in 27.53s
```

After F-009 and the evidence correction, the same complete suite passed:

```text
580 passed, 2 skipped in 26.66s
```

After F-010 centralized the masked-aware coercion boundary, the same complete
suite passed:

```text
587 passed, 2 skipped in 28.02s
Final pre-commit rerun: 587 passed, 2 skipped in 23.66s
```

After F-011 corrected nested protocol snapshotting, the same complete suite
passed:

```text
Initial post-fix run: 595 passed, 2 skipped in 27.18s
Final pre-commit run with the mapping-value compatibility check:
596 passed, 2 skipped in 21.21s
```

The two skips are the existing CUDA smoke checks on a host where CUDA is
unavailable. No real scientific workload, empirical comparison, parameter
retuning, new provenance mechanism, or human/publication metadata was added.

## Task 3: Study schemas, datasets, and simulator contracts

### Environment and bounded execution

The audit ran in the isolated `codex/publication-integration` worktree from
base `3e61a88ffc288bbb7d3343c63bafdc39e8577a7b` with
`/tmp/maskimpute-scziva312/bin/python`. The requested Ruff commands used
`PYTHONPATH=/tmp/publication-integration-review-tools-task1`. No package was
installed and no real dataset, simulator, or scientific workload was run.

The exact requested pre-change suite passed:

```text
586 passed, 5 skipped in 250.35s (0:04:10)
```

The five skips are the suite's existing real-runtime smoke tests. They were
not enabled because this review was limited to JSON parsing and bounded
fixtures.

The Step 1 parser loaded all 17 tracked study documents exactly once. A second
strict parse rejected duplicate object keys and non-finite JSON constants; all
17 documents also passed that stricter parse.

### Path and invariant coverage

Every assigned production path was inspected:

| Path | Coverage and disposition |
|---|---|
| `maskimpute_benchmark/schema.py` | Dataset/truth schema construction, required arrays, dimensions, finite and integer count domains, axis labels, optional metadata, and legacy outer provenance compatibility; covered and valid |
| `maskimpute_benchmark/protocol.py` | Required and exact keys, schema version, registry agreement, namespace identity, mechanisms, draws, views, seeds, non-empty identifiers, primary metric uniqueness, and Splatter exclusion; F-012 closed |
| `maskimpute_benchmark/study.py` | Study roots, atomic JSON state, ledger/result-journal schemas, dataset/source/view/mechanism binding, append-only identity, ordering, and recovery behavior; F-015 and F-016 closed |
| `maskimpute_benchmark/datasets.py` | Development panel, dataset registry, pair receipts, status documents, scaling/calibration declarations, dimensions, mechanisms, views, paths, and bool-versus-int rejection; F-013 closed |
| `maskimpute_benchmark/sources.py` | Exact source document schema, unique source IDs, allowed roles, repository-relative paths, runtime declarations, and source/runtime agreement; covered and valid |
| `maskimpute_benchmark/trajectory_dataset.py` | Separate trajectory authority, exact schema, duplicate-key handling, dimensions, identifiers, truth agreement, finite values, and output immutability; F-014 closed |
| `maskimpute_benchmark/simulators/__init__.py` | Public simulator adapter surface; covered and valid |
| `maskimpute_benchmark/simulators/base.py` | Contract dataclasses, exact file populations, moderate/severe view populations, paired seeds/truth, input domains, and receipt agreement; covered and valid |
| `maskimpute_benchmark/simulators/native.py` | Native simulator bounded fixture and truth contract; covered and valid |
| `maskimpute_benchmark/simulators/runtime_assets.py` | Runtime asset declarations, exact source IDs/roles and receipt schema types, repository paths, environment agreement, and missing-asset rejection; F-017 closed |
| `maskimpute_benchmark/simulators/semisynthetic.py` | Semi-synthetic source selection, deterministic bounded generation, view population, truth preservation, and receipt construction; covered and valid |
| `maskimpute_benchmark/simulators/sergio.py` | SERGIO adapter inputs, runtime/source binding, exact output population, paired truth/seeds, and fail-closed fixture boundaries; covered and valid |
| `maskimpute_benchmark/simulators/sparsim.py` | SPARSim adapter inputs, runtime/source binding, exact output population, paired truth/seeds, and fail-closed fixture boundaries; covered and valid |
| `maskimpute_benchmark/simulators/symsim.py` | SymSim adapter inputs, runtime/source binding, exact output population, paired truth/seeds, and fail-closed fixture boundaries; covered and valid |

Every assigned test path was inspected and executed:

| Path | Coverage classification |
|---|---|
| `tests/test_dataset_registry.py` | Development-panel, registry, pair-receipt, status, path, mechanism/view/draw, and schema-version contracts; F-013 regressions |
| `tests/test_dataset_schema.py` | Dataset/truth shapes, domains, labels, metadata, immutability, and failure boundaries |
| `tests/test_protocol.py` | Exact protocol schema, registry coverage, namespaces, mechanisms, views, draws, seeds, metrics, and excluded final simulators; F-012 regressions |
| `tests/test_sources.py` | Exact source ledger, source IDs, roles, paths, and runtime agreement |
| `tests/test_study_state.py` | Study roots, ledgers, result journal, recovery, identity binding, ordering, and exact integer schema/sequence fields; F-015 and F-016 regressions |
| `tests/test_trajectory_dataset.py` | Trajectory authority schema, JSON parsing, dimensions, truth, values, immutability, and duplicate keys; F-014 regression |
| `tests/test_semisynthetic_adapter.py` | Deterministic semi-synthetic fixture, exact output population, truth, views, and receipts |
| `tests/test_sergio_adapter.py` | SERGIO source/runtime binding, paired outputs, seeds, truth, populations, and invalid outputs |
| `tests/test_simulator_contract.py` | Shared simulator inputs, exact file sets, dimensions, seeds, truth, views, and receipt contract |
| `tests/test_simulator_runtime_assets.py` | Runtime declarations, source binding, receipt schema types, repository paths, environments, and missing assets; F-017 regression |
| `tests/test_sparsim_adapter.py` | SPARSim source/runtime binding, paired outputs, seeds, truth, populations, and invalid outputs |
| `tests/test_symsim_adapter.py` | SymSim source/runtime binding, paired outputs, seeds, truth, populations, and invalid outputs |

The 17 parsed JSON paths were:

```text
study/ablations.json
study/calibration_contract.json
study/comparator_tuning.json
study/development_panel.json
study/development_search.json
study/method-attempts/scgimpute.json
study/method-attempts/sczn.json
study/methods.json
study/protocol.json
study/scaling_panel.json
study/selection_contract.json
study/simulator_r_environment.lock.json
study/simulator_runtime_assets.json
study/sources.json
study/trajectory_panel.json
study/v28_revision.json
study/v29_revision.json
```

Cross-document checks established exact agreement between the protocol and
development registries for all four mechanisms; the `moderate` and `severe`
technical views; the two declared development draws; the four source bindings
with source IDs (`baron-pancreas-umi`, `sergio`, `sparsim`, and `symsim`) and
two role values (`semisynthetic_source` and `mechanism`); and the final
trajectory dimensions. List identity fields were non-empty and unique for
ablation variants, comparator configurations, development-search
configurations, methods, comparator method/environment bindings, orthogonal
endpoints, and source IDs. Declared tracked inputs existed. Generated artifact,
cache, and receipt paths were absent before workloads, as expected, and were
not treated as defects.

### Task 3 findings and dispositions

| ID | Severity | Finding | Root cause | Disposition |
|---|---|---|---|---|
| F-012 | Important | Protocol authorities accepted unknown top-level, development, and final fields; arbitrary development/final namespaces; and a duplicate primary metric. | The parser checked required fields and individual value domains but did not enforce version-1 exact field populations, fixed execution namespaces, or metric uniqueness. | Closed test-first with exact version-1 fields, `dev`/`final` namespace agreement, and unique primary metrics. The established explicit Splatter rejection remains prior to structural validation so its public error remains stable. |
| F-013 | Important | Development-panel, dataset-status, and dataset-pair-receipt authorities accepted boolean schema versions because `True == 1`. | These three parsers used equality without first requiring an exact integer type. | Closed test-first by requiring `type(schema_version) is int` at each owning boundary. |
| F-014 | Important | A trajectory authority accepted a duplicate JSON key when both occurrences had the same value. | The loader used ordinary `json.loads`, which silently retained the last duplicate member before schema validation. | Closed test-first with duplicate-member rejection during trajectory JSON decoding. |
| F-015 | Important | Result-journal entries accepted a boolean schema version when the entry's existing binding fields were updated consistently. | Journal validation used equality without first requiring an exact integer type. | Closed test-first by requiring an exact integer schema version before accepting an entry. |
| F-016 | Important | A first result-journal entry accepted boolean `sequence: true` when its existing authority binding was updated consistently. | The journal validator compared the value with sequence 1 before requiring an exact integer type. | Closed at the journal validator with an exact integer type check before sequence equality; the regression enters through the public final-claim loader. |
| F-017 | Important | A recorded runtime source receipt accepted boolean `schema_version: true`, and later mapping equality treated it as equal to integer 1. | The semantic runtime-receipt validator copied the field without first requiring its exact type and version. | Closed at the semantic receipt owner with an exact integer type check before the loader's later equality check; the regression enters through the public runtime-assets loader. |

No source declaration, study JSON document, runtime asset declaration,
simulator adapter, scientific parameter, estimand, legacy outer provenance
field, or human/publication metadata was changed.

### Exact mutation evidence

The original RED invocation grouped the eight new test functions and their 11
parameterized nodes. Before production changes it reported:

```text
11 failed in 6.93s
```

Because that first run was grouped, it is not described as isolated RED
evidence. Isolation was reconstructed afterward in a temporary detached
worktree at the exact review base
`3e61a88ffc288bbb7d3343c63bafdc39e8577a7b`. Only the four changed test files
were applied there; production remained at the base. Each node below was then
invoked by itself and exited 1 because the expected exception was not raised.
The temporary worktree was removed after the reconstruction. Each node was
also invoked by itself in the corrected tree and exited 0:

| Pytest node | Reconstructed isolated RED | Corrected isolated GREEN |
|---|---|---|
| `tests/test_protocol.py::test_protocol_rejects_unknown_fields[None]` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_protocol.py::test_protocol_rejects_unknown_fields[development]` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_protocol.py::test_protocol_rejects_unknown_fields[final]` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_protocol.py::test_protocol_namespaces_match_the_execution_contract[development-development]` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_protocol.py::test_protocol_namespaces_match_the_execution_contract[final-publication-final]` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_protocol.py::test_protocol_primary_metrics_are_unique` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_dataset_registry.py::test_development_panel_rejects_boolean_schema_version` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_dataset_registry.py::test_dataset_status_rejects_boolean_schema_version` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_dataset_registry.py::test_dataset_pair_receipt_rejects_boolean_schema_version` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_trajectory_dataset.py::test_trajectory_authority_rejects_duplicate_json_keys` | 1 failed; exit 1 | 1 passed; exit 0 |
| `tests/test_study_state.py::test_result_journal_rejects_boolean_schema_version` | 1 failed; exit 1 | 1 passed; exit 0 |

The first complete post-fix suite exposed one compatibility regression:
`test_protocol_rejects_splatter_as_final` received the new structural error
before its established explicit Splatter error:

```text
1 failed, 596 passed, 5 skipped in 240.15s
```

The cause was validation precedence, not missing strictness. Moving the
explicit excluded-simulator check ahead of exact-field validation preserved
the established error while retaining every new fail-closed check. The
focused protocol check then passed:

```text
7 passed in 0.04s
```

The fresh complete Task 3 rerun passed:

```text
597 passed, 5 skipped in 249.14s (0:04:09)
```

Targeted Ruff checking and formatting, scoped byte compilation, and
`git diff --check` also passed. These results establish the reviewed schema
and bounded-fixture contracts only; they do not establish real simulator
availability or scientific readiness.

### Reviewer follow-up evidence

The two follow-up regressions were each run alone before and after the owning
validator change:

| Pytest node | Isolated RED | Corrected isolated GREEN |
|---|---|---|
| `tests/test_study_state.py::test_result_journal_rejects_boolean_sequence_at_public_boundary` | 1 failed in 3.28s; exit 1 because no exception was raised | 1 passed in 2.99s; exit 0 |
| `tests/test_simulator_runtime_assets.py::test_loader_rejects_boolean_runtime_source_receipt_schema_version` | 1 failed in 1.74s; exit 1 because no exception was raised | 1 passed in 1.71s; exit 0 |

All 11 earlier Task 3 mutation nodes were then invoked as separate pytest
processes and passed. The two adjacent test files passed together:

```text
106 passed in 57.78s
```

The complete fresh Task 3 suite passed:

```text
599 passed, 5 skipped in 250.72s (0:04:10)
```

Strict duplicate-aware and non-finite-aware parsing accepted exactly the 17
tracked study JSON documents. Full `maskimpute_benchmark` and `tests` byte
compilation exited 0. Ruff check, Ruff format checking of the four changed
Python files, and `git diff --check` passed.

The Task 3 brief was corrected to remove the nonexistent
`maskimpute_benchmark/config.py` responsibility from its file list, inspection
command, and staging command. This is a plan correction, not a missing
production-file finding. No real dataset, simulator, or scientific workload
was run.

## Task 4 method-adapter and fair-comparator execution audit

### Environment and baseline

Task 4 began from a clean linked worktree at the expected
`codex/publication-integration` head. The existing
`/tmp/maskimpute-scziva312/bin/python` interpreter was used without installing
or changing dependencies.

The exact eleven-file suite named in the Task 4 brief initially reported:

```text
21 failed, 489 passed, 44 skipped in 573.59s
```

All 21 failures were in `tests/test_runtime_environments.py`. The invoking
shell inherited this CUDA library path:

```text
/usr/local/cuda/lib64:/usr/local/cuda/lib64:
```

Both entries resolve through symlinks, so the declared runtime validator
rejected them. This is an environment-selection concern, not a repository
defect: the production operational-environment builder already removes the
inherited variable. A representative failing runtime test passed unchanged
when invoked without that inherited setting:

```text
1 passed in 182.31s
```

No runtime-lock, path-hardening, dependency, or operational-environment code
was changed.

### Registry, adapter, and execution coverage

All 20 canonical registry entries were cross-checked for execution scope,
integration status, input/output scale, stochastic and seed policy,
environment, CPU/GPU mode, timeout, memory ceiling, and observed-positive
policy. They comprise three in-tree/control methods, ten scheduled same-input
comparators, two historical methods, three unavailable methods, and two
external-reference-only methods.

The 34 smoke rows bind exactly the scheduled comparator configurations:
1 ALRA, 4 MAGIC, 4 DCA, 4 scVI, 1 SAVER, 4 scZiva, 4 afMF, 4 BiAEImpute,
4 scCR, and 4 scSDAE. Their typed configuration decoding, canonical ordering,
fixed truth-free fixture, complete binding projection, and exact 34-row
denominator were inspected.

Every adapter was compared with the shared input/output boundary for declared
domain, preprocessing, seed propagation, output orientation, observed-value
policy, runtime invocation, and failure translation:

- ALRA and MAGIC use their declared log-normalized cells-by-genes domain and
  the shared inverse to count equivalents.
- DCA and scVI consume raw cells-by-genes counts; scVI uses only the allowed
  batch covariate and rescales frequencies with observed library sizes.
- SAVER, afMF, scCR, and scSDAE retain their declared native normalized output
  and explicit evaluator inverses.
- afMF and scSDAE explicitly bridge their upstream genes-by-cells contracts
  back to the benchmark cells-by-genes orientation.
- scZiva and BiAEImpute propagate required framework seeds and retain their
  declared non-preservation policy without a post-hoc observed-value rewrite.
- MaskImpute and its matched control bind the exact model seed and raw-count
  output policies; the separately declared full-denoising behavior remains
  explicit.
- D3Impute and SCTSI remain external-reference-only with exact matched-bulk,
  orientation, and scale handling. Preserved legacy provenance was not
  modified.

The direct competition plan has 2,896 scheduled rows: 16 observed controls,
48 matched controls, 1,200 MaskImpute rows, and 1,632 comparator rows. Every
identity includes the complete method/configuration binding and dataset,
mechanism, biological, technical-view, seed, and draw coordinates. Structural
validation re-resolves those identities against the authorities.

Direct dispatch uses only the typed configuration for the named adapter,
rejects configuration mutation, enforces method resources, and maps
unavailable, timeout, memory, infrastructure, and unexpected adapter failures
to the closed terminal vocabulary. Completed rows carry exactly six metrics.
Non-completed rows carry one consistent terminal status/reason with zero
metric denominators. Checkpoint replay reconstructs those semantics, requires
an exact plan prefix, revalidates bindings and budget states, and retains
blockers in the scheduled denominator. No comparator fallback or fabricated
output was found.

### Task 4 findings and dispositions

| ID | Severity | Finding | Root cause | Disposition |
|---|---|---|---|---|
| F-018 | Important | D3Impute accepted a method specification whose stochastic flag or seed policy drifted from its declared deterministic/no-seed contract. | Its adapter-specific specification guard checked identity, track, and scales but omitted the seed contract enforced by the sibling SCTSI external-reference adapter. | Closed test-first. The D3Impute boundary now rejects independent or combined stochastic/seed-policy drift before finalization or execution. |
| F-019 | Important | Smoke execution granted all methods the global 14 GiB GPU ceiling, and smoke-receipt validation applied the same global ceiling. A CPU-only method could consume GPU memory and still satisfy the readiness gate despite its exact zero-GPU binding. | Request construction and receipt validation used authority-wide maxima instead of the `ResourceSpec` attached to the canonical method. | Closed test-first. Spawned smoke requests now use the method's exact timeout, RSS, and GPU limits; receipt validation enforces the exact method RSS/GPU limits, including zero GPU bytes for CPU-only methods. |
| O-003 | Minor | The inherited CUDA library path caused the initial runtime-environment suite failures. | The two inherited entries resolve through symlinks, which the runtime declaration intentionally rejects. | No code change. The authoritative rerun used the same sanitized environment constructed by the operational boundary. |

No other adapter, tuning, plan, execution, checkpoint, or operational
environment defect was demonstrated. The method-specific observed-value
differences for scZiva, BiAEImpute, scSDAE, and full denoising are intentional
declared policies covered by tests.

### Exact test-first and verification evidence

The focused pre-fix invocation covered three D3Impute seed-contract cases, a
CPU smoke-receipt GPU overage, a spawned CPU smoke request, and the existing
RSS-overage control:

```text
5 failed, 1 passed in 2.40s
```

The same invocation after the minimal fixes passed:

```text
6 passed in 1.97s
```

The complete two changed test files then passed:

```text
153 passed, 8 skipped in 204.47s
```

The authoritative complete eleven-file suite was rerun without the inherited
CUDA library path and passed:

```text
513 passed, 44 skipped in 1730.33s (0:28:50)
```

Targeted Ruff lint reported all checks passed, and Ruff format checking
reported all four changed Python files already formatted. Byte compilation of
all scoped production modules and test files exited zero with no output.
`git diff --check` also exited zero with no output.

These checks establish the reviewed method binding, adapter, fixed-smoke,
direct execution, and checkpoint contracts only. No real comparator workload
was run, no external runtime availability claim was made, and no empirical
performance or publication-readiness conclusion is drawn.
