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
  orientation, scale, exact-boolean stochastic flags, and deterministic
  no-seed contracts. Preserved legacy provenance was not modified.

The direct competition plan has 2,896 scheduled rows: 16 observed controls,
48 matched controls, 1,200 MaskImpute rows, and 1,632 comparator rows. Every
identity includes the complete method/configuration binding and dataset,
mechanism, biological, technical-view, seed, and draw coordinates. Structural
validation re-resolves those identities against the authorities.

Direct dispatch uses only the typed configuration for the named adapter,
rejects configuration mutation, enforces method resources, and maps
unavailable, timeout, memory, infrastructure, and unexpected adapter failures
to the closed terminal vocabulary. Completed rows carry exactly six metrics.
Ready comparator-smoke evidence requires both canonical process-tree RSS
provenance and canonical process-tree GPU provenance on every completed row.
Non-completed rows carry one consistent terminal status/reason with zero
metric denominators. Checkpoint replay reconstructs those semantics, requires
an exact plan prefix, revalidates bindings and budget states, and retains
blockers in the scheduled denominator. No comparator fallback or fabricated
output was found.

### Task 4 findings and dispositions

| ID | Severity | Finding | Root cause | Disposition |
|---|---|---|---|---|
| F-018 | Important | D3Impute accepted a method specification whose stochastic flag or seed policy drifted from its declared deterministic/no-seed contract. | Its adapter-specific specification guard checked identity, track, and scales but omitted the seed contract enforced by the sibling SCTSI external-reference adapter. | The initial correction rejected ordinary boolean/policy drift. F-021 completed the closure by requiring an exact boolean and auditing the same invariant across external-reference and shared same-input adapter guards. |
| F-019 | Important | Smoke execution granted all methods the global 14 GiB GPU ceiling, and smoke-receipt validation applied the same global ceiling. A CPU-only method could consume GPU memory and still satisfy the readiness gate despite its exact zero-GPU binding. | Request construction and receipt validation used authority-wide maxima instead of the `ResourceSpec` attached to the canonical method. | The initial correction bound requests and receipt caps to each method's `ResourceSpec`. F-020 completed the closure by requiring live independent GPU telemetry and authoritative receipt evidence for the zero-GPU policy. |
| F-020 | Important | The corrected zero-byte cap was not actually observed for CPU-declared smoke methods: the spawned runner skipped GPU sampling, initialized a synthetic zero, and allowed imported completed outcomes to label that zero unavailable or not applicable. | The sampler's `gpu_required` switch was also used as the decision to measure GPU use, conflating a method's scheduling mode with whether the smoke authority needed evidence that forbidden use was zero. Receipt validation required only a nonempty measurement string. | The first test-first correction required independent GPU measurement for every smoke row, retained the CPU path's 50 ms cadence, enforced nonzero forbidden use against the zero cap, and required canonical completed-receipt GPU evidence. It detected telemetry that remained unavailable at the terminal check but did not retain an earlier transient gap; F-024 completes that live enforcement. |
| F-021 | Important | Directly constructed `MethodSpec` values could bypass declared seed contracts. Integer zero passed the D3Impute and SCTSI truthiness checks, while observed and ALRA accepted self-consistent stochastic/seed-policy drift; all same-input adapters shared the latter incomplete guard. | External-reference guards used truthiness instead of an exact boolean check, and the shared same-input guard checked identity, track, and scales without checking the canonical seed contract. | Closed test-first. D3Impute and SCTSI require an exact false boolean with `not_applicable`; the one shared same-input guard now enforces the exact canonical stochastic/seed-policy pair for observed, the two in-tree learned methods, and all ten implemented same-input comparator adapters. |
| F-022 | Minor | A directly constructed method resource specification with fractional GiB limits produced floating-point byte limits in the smoke request, unlike other direct request paths. | Smoke request construction multiplied GiB by bytes-per-GiB without applying the established integer normalization. | Closed test-first by normalizing both RSS and GPU limits to integer bytes at direct smoke request construction. |
| F-023 | Important | A completed 34-row comparator-smoke population could use unavailable, not-applicable, or arbitrary nonempty RSS provenance and still produce ready imported evidence. | Receipt construction canonicalized GPU provenance after F-020 but continued to validate RSS provenance only as a nonempty string. | Closed test-first. Every completed smoke row must now use the canonical `linux_proc_process_tree_rss` measurement code. Synthetic completed fixtures use that same valid provenance, and receipt loading inherits the check through full receipt recomputation. |
| F-024 | Important | A required telemetry observation could fail, return a non-`ResourceSample`, omit RSS or GPU, or carry noncanonical required provenance and then be followed by a valid canonical sample; the measured spawn could still complete or report a later resource overage. | The parent loop retained only accumulated peaks and the most recent provenance. It had no run-level record that a required observation interval was unmeasured or unauthoritative, so later values erased the terminal evidence of the gap. | Closed test-first. Any required RSS gap, required GPU gap, sampler exception, or invalid sample now sets a sticky run-level telemetry failure. Later samples may continue collecting peaks but cannot restore authority; `resource_telemetry_unavailable` takes precedence over a later cap result or child outcome. Exact process-tree provenance is required for live RSS and for GPU whenever GPU telemetry is required. GPU-unrequired non-smoke CPU execution remains valid, and CPU smoke sampling retains its 50 ms cadence. |
| O-003 | Minor | The inherited CUDA library path caused the initial runtime-environment suite failures. | The two inherited entries resolve through symlinks, which the runtime declaration intentionally rejects. | No code change. The authoritative rerun used the same sanitized environment constructed by the operational boundary. |

After F-018 through F-024, no unresolved adapter, tuning, plan, execution,
checkpoint, or operational-environment defect was demonstrated. The
method-specific observed-value differences for scZiva, BiAEImpute, scSDAE,
and full denoising are intentional declared policies covered by tests.

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

The independent review exposed the incomplete live-measurement, exact-boolean,
shared seed-contract, and fractional-byte cases recorded as F-020 through
F-022. The focused reviewer-correction invocation reported:

```text
14 failed, 3 passed in 3.44s
```

The failures were the expected missing seed-contract exceptions, unsupported
required GPU-measurement mode, accepted unavailable/not-applicable completed
CPU evidence, absent smoke measurement option, and floating-point byte limits.
The identical invocation after the minimal corrections passed:

```text
17 passed in 4.57s
```

The complete five owning test files then passed:

```text
364 passed, 31 skipped in 1117.17s (0:18:37)
```

After self-review preserved the CPU-forbidden path's 50 ms sampling cadence,
the affected telemetry and smoke nodes passed:

```text
6 passed in 7.36s
```

The complete eleven-file suite first passed after the reviewer corrections
with the inherited CUDA library path removed:

```text
524 passed, 44 skipped in 1728.03s (0:28:48)
```

After the one-line cadence refinement, the exact same suite passed again at
that correction state:

```text
524 passed, 44 skipped in 1719.54s (0:28:39)
```

The final independent re-review then exposed the remaining RSS provenance
gap recorded as F-023. Three completed 34-row receipt cases using
`rss_measurement_unavailable`, `not_applicable`, and an arbitrary nonempty
label all failed to raise before the production correction:

```text
3 failed in 2.06s
```

After requiring the canonical process-tree RSS code, the identical focused
node passed all three parameter cases:

```text
3 passed in 2.17s
```

The focused smoke builder, GPU/RSS provenance, create-only, and loader set
then passed:

```text
10 passed in 2.77s
```

The complete owning comparator-tuning file passed:

```text
125 passed in 204.56s (0:03:24)
```

The next independent review exposed the transient telemetry recovery bypass
recorded as F-024. Six cases covered sampler exceptions, invalid sample
objects, missing required RSS or GPU values, and noncanonical required RSS or
GPU provenance, each followed by valid canonical observations. A seventh case
combined an initial telemetry gap with a later resource overage; the
GPU-unrequired non-smoke CPU path served as the passing control. Before the
production correction, the focused invocation reported:

```text
7 failed, 1 passed in 21.61s
```

After adding the sticky required-telemetry failure state, the identical
invocation reported:

```text
8 passed in 21.18s
```

The expanded focused runner set, including the existing completion, timeout,
RSS/GPU-cap, required-GPU, and direct-revision cases, then reported:

```text
19 passed in 33.82s
```

The complete owning runner and comparator-tuning files passed at the final
formatted production state:

```text
281 passed in 1186.85s (0:19:46)
```

The first post-F-024 exact-suite attempt then exposed one stale test fixture:
`_DirectFixedResourceSampler` still reported synthetic parent provenance, so
the corrected live boundary properly returned `infrastructure_error` instead
of the fixture's expected timeout. The otherwise complete invocation reported:

```text
1 failed, 526 passed, 44 skipped in 1769.07s (0:29:29)
```

The fixture was changed to the same canonical process-tree labels required of
the other measured-spawn test samplers. Its timeout and required-GPU-missing
nodes then reported:

```text
2 passed in 4.25s
```

The exact sanitized eleven-file Task 4 suite passed at the final formatted
F-024 state:

```text
527 passed, 44 skipped in 1702.67s (0:28:22)
```

Targeted Ruff lint and formatting, byte compilation of all final changed
production and test files, and `git diff --check` passed.

These checks establish the reviewed method binding, adapter, fixed-smoke,
direct execution, and checkpoint contracts only. No real comparator workload
was run, no external runtime availability claim was made, and no empirical
performance or publication-readiness conclusion is drawn.

## Task 5: Metrics, statistics, and evaluation orchestration

The audit covered the complete Task 5 production and test scope: metric
domains and denominators, reason-coded unavailable states, biological-draw
inference, duplicate/conflicting result identities, common-scale conversion,
plan and checkpoint evidence, calibration separation, external-reference
truth isolation, pre-zero score semantics, and evaluation-manifest
reconstruction.

The biological draw remains the independent inferential unit. Technical views
and model seeds remain repeated measurements. The hierarchical bootstrap,
variance components, interval construction, win/tie/loss definition, and Holm
multiplicity family are unchanged. External reference remains fixed to the
D3Impute/SCTSI denominator on the declared Tung single-cell and matched-bulk
inputs, with truth excluded from method execution. Evaluation-manifest direct
and orthogonal evidence is reconstructed independently from typed authority.

### Task 5 findings and dispositions

| ID | Severity | Finding | Root cause | Disposition |
|---|---|---|---|---|
| F-025 | Important | Direct and nested dense masks at metric inputs, a common-scale converter result, and stored pre-zero authoritative targets were silently accepted as ordinary dense values. | Ordinary NumPy conversion erased masked-array semantics before the owning dense boundary validated the resulting matrix. | Closed test-first for direct, nested, and protocol-produced dense masks. Metrics, evaluator targets and converter output, and stored-score targets reuse the approved unmasked dense snapshot boundary. F-030 separately closes masked storage inside an already sparse container. |
| F-026 | Important | A finite extended-precision matrix outside float64 range became infinity after validation, while finite float64 inputs could overflow derived metric arithmetic and abort the complete metric row. | Finiteness was checked only before float64 conversion, and the strict public scalar dataclass received a nonfinite derived value directly. | Closed test-first for post-conversion finiteness and complete reason-coded metric rows. Its first arithmetic correction classified nonfinite intermediates as `nonfinite_metric` even when a scaled final estimand was representable; F-031 completes that numerical guarantee. |
| F-027 | Minor | Exported tie-aware grouping accepted zero/negative maximum group counts and lacked explicit empty, dimensional, numeric, and finite input checks. | The public helper assumed already validated internal callers. | Closed test-first by enforcing a nonempty finite one-dimensional real vector and an exact positive integer maximum. The stable tie-preserving partition is unchanged. |
| F-028 | Minor | Whitespace-only result statuses passed nonempty validation and canonicalized to an empty status. | Validation preceded `strip().lower()`. | Closed test-first by requiring every identity/status string to contain a non-whitespace character before canonicalization. |
| F-029 | Important | Pre-zero outer/policy schema versions and metric/overall denominators could accept Boolean integer aliases. | Equality checks relied on Python's Boolean/integer equivalence. | Closed test-first with exact built-in integer checks. Existing reliability-bin and stratum exact-type checks remain. No checksum or provenance mechanism changed. |
| F-030 | Important | A SciPy sparse evaluator target whose internal `.data` was a `MaskedArray` reached `toarray()` before mask validation; densification silently returned the underlying stored values. Completed evaluation and terminal-attempt pre-zero evidence could therefore consume a masked authority as ordinary data. | The F-025 dense snapshot guard was applied only after the sparse/dense dispatch, while SciPy's densifier erased the sparse storage mask. | Closed test-first. The evaluator inspects sparse storage with the foundational mask boundary before any `toarray()` call. Direct conversion, completed public evaluation, and terminal-attempt pre-zero target extraction reject the masked CSR; valid sparse behavior and import layering are unchanged. |
| F-031 | Important | The F-026 fallback mislabeled representable endpoints as unavailable or zero: the published gNRMSE probe returned zero instead of `7.866824069956793e-309`, opposite-sign finite inputs overflowed during subtraction, standard deviation, norms, and reductions, and a raw difference, square, or sum could overflow even when the final mean or ratio fit float64. | Reconstruction materialized raw float64 intermediates before division, averaging, cancellation, or normalization established the range of the final estimand. Suppressing the warning did not recover the lost value. | The first test-first correction introduced normalized mantissa/exponent terms and closed the demonstrated overflow cases. Its claim that the scaled terms completed the numerical guarantee was too broad: F-032 and F-034 close the underflow and exact-legacy branches, while F-036 closes cancellation between separately rounded scaled endpoints. |
| F-032 | Important | The conservative ordinary gNRMSE and correlation branches did not distinguish underflow from safe execution. A two-cell error at `finfo(float64).tiny` raised or collapsed to zero instead of the positive representable `1.5733648139913584e-300`; tiny nonconstant correlations and mixed-scale scaled reductions could raise under `np.errstate(all="raise")`. | The ordinary branch proved only overflow bounds, then executed underflow-prone squares and covariance reductions. The scaled branch also allowed expected normalization underflow to inherit a caller's strict floating-point state. | Closed test-first. Every ordinary endpoint is attempted under a strict local error state and accepted only when it completes finite; otherwise the scaled route is used. Scaling permits underflow only at the specific dimensionless normalization, square, and centering operations where terms below relative float64 precision cannot alter the representable reduction. Tiny gNRMSE remains positive at the required value, tiny correlations complete, and mixed-scale routes do not leak floating-point exceptions. |
| F-033 | Important | `log1p_cp10k` and `count_equivalent_to_log2_cp10k` summed raw nonnegative rows before normalization. `[DBL_MAX, DBL_MAX]` therefore overflowed and could warn or become two zeros instead of two `log1p(5000)` or `log2(5001)` values. The same raw sum in `observed_library_sizes` leaked the arithmetic failure. | The evaluator materialized an unrepresentable total even though CP10k needs only row proportions. The observed-library owner did not translate a genuinely unrepresentable total into its declared adapter failure contract. | The first test-first correction preserved the two equal maximum entries through maximum-scaled proportions and retained exact legacy safe rows. F-041 completes that path for entries whose proportion underflows before multiplication by 10,000. Observed library sizes retain the exact legacy finite sum, fail explicitly with `unrepresentable_library_size` when the total itself cannot fit, and preserve `zero_library_cell` precedence even when another row overflows. |
| F-034 | Important | Stable arithmetic changed ordinary finite metric serialization broadly: deterministic bounded probes differed from the legacy NumPy result for MSE, MAE, mean and variance distortion, Wasserstein distance, and pairwise distance. Unconditional `fsum` also changed library-size tie membership from `[[0, 2], [1], [3]]` to `[[2], [0, 1], [3]]` in a representable legacy reduction. | The scaled and compensated algorithms ran even when the exact established NumPy formula had completed without an overflow, underflow, invalid, or divide signal and yielded a finite endpoint. Conservative magnitude gates covered only gNRMSE and correlation. | The first correction restored the established NumPy formula as the strict first branch and retained the exact legacy tie group. F-047 refines only mean, variance, and pairwise distance: a finite ordinary candidate is now retained when conservative joint bounds certify its final binary64 cell, while uncertified cancellation uses the already reviewed bounded/refined path. MSE, MAE, gNRMSE, correlation, Wasserstein, and row-library behavior remain unchanged. |
| F-035 | Minor | Finite extended-precision values outside float64 range leaked `RuntimeWarning` or `FloatingPointError` from evaluator-target, common-scale-output, and stored pre-zero target casts under warnings-as-errors. | The boundaries caught Python conversion exceptions but not NumPy's floating-point signal from an overflowing cast. | Closed test-first with narrow `over`/`invalid` cast trapping. Evaluator targets enter `RunnerContractError`, common-scale output enters the existing reason-coded unavailable path, and both stored pre-zero target roles enter `PreZeroEvidenceError`; no warning or floating-point exception crosses the public contract. |
| F-036 | Important | The scaled variance and pairwise-distance fallbacks could still lose a representable final difference by independently rounding two unrepresentable or nearly equal endpoints before subtraction. For adjacent extreme profiles, an exact population-variance distortion of `0x1.8p+1023` became unavailable; an exact cell-distance distortion of `0x1.6a09e667f3bcdp+971` became `0x1p+972`. | Variances were formed by separately squaring two rounded scaled standard deviations, and pair distances by separately rounding two scaled norms. Subtracting those rounded endpoints discarded the low-order difference the metric is defined to retain. | Closed test-first in both operand orders. The strict established NumPy branches remain unchanged. The variance fallback now common-scales each aligned gene profile and sums centered square differences directly through `(a-b)*(a+b)` using exact rational arithmetic in linear time. The distance fallback common-scales both pair-difference vectors and evaluates `abs(sum(u²-v²))/(\|\|u\|\|+\|\|v\|\|)` before either norm is rounded. Exact-hex warnings-as-errors regressions lock both reviewed endpoints, and the obsolete subtract-after-rounded helpers were removed. |
| F-037 | Minor | The native-output matrix boundary leaked `FloatingPointError` or a promoted `RuntimeWarning` when a finite `longdouble` lay outside float64 range. | Unlike the other reviewed conversion boundaries, `_validated_native_matrix` performed its narrowing cast outside a local NumPy error-translation boundary. | Closed test-first for both raw-count and log1p-CP10k native-output converters. Only `over` and `invalid` signals from the narrowing cast are translated to the established `ValueError` contract; ordinary float64 conversion and subsequent nonnegative/finite validation are unchanged. |
| F-038 | Important | The F-036 joint fallbacks still classified two positive, representable minimum-subnormal endpoints as unavailable. The exact variance difference `2^-1075 + 2^-1128` and a pairwise norm difference slightly above one half of the minimum subnormal both had to round upward, but instead became `nonfinite_metric` in both operand orders. | Each completed exact or high-precision endpoint was converted to a float mantissa before `ldexp` restored its scale. That first rounding erased the increment above the half-subnormal boundary, so the second rounding tied to zero. | Closed test-first in both operand orders. The reviewed variance and pairwise results remain exact or high precision through their completed aggregates and are converted directly once. Positive endpoints that round to zero and overflowing endpoints remain reason-coded unavailable; exact zero remains available. F-039 replaces the initial whole-endpoint exact implementation with bounded wider blocks while retaining exact `Fraction`/`Decimal` state for every ambiguity. The obsolete exact-rational-to-scaled-term path remains removed. |
| F-039 | Important | One floating-point signal in pairwise distance switched the complete endpoint to exact `Fraction`/`Decimal` work for every cell pair and gene, then retained every pair value until the mean. The 2,700-cell by 1,200-gene final design would therefore require about 3.64 million exact pair constructions and 4.37 billion Python-level component operations. The analogous variance route exact-recomputed every gene after one signal. | F-036 and F-038 correctly preserved cancellation and single rounding, but used exact arithmetic as the whole-endpoint fallback rather than as a narrow ambiguity resolver. The pairwise implementation also materialized an all-pair `Decimal` list. | Closed test-first on wider-`longdouble` platforms without changing the ordinary NumPy branch or either exact endpoint. Those platforms common-scale bounded pair/gene blocks, form cancellation-preserving extended-precision intervals, and use the existing exact pair/gene helper only for ambiguous intervals or unresolved final rounding. The original portable path remained bounded-memory but exact-recomputed every pair or gene; F-042 completes the same approximate/interval/escalation architecture for float64-only platforms. |
| F-040 | Important | The variance interval accepted an inaccurate near-constant gene as safe. In the exact reviewed two-gene fixture, the first gene was rounded to `0x1.c71c71c00000bp-101` while a minimum-subnormal second gene forced fallback; the required aggregate is `0x1.c71c71c71c71cp-101` in both operand orders. | Its error bound was relative only to the two already-computed variances. It omitted absolute normalization, mean, centering, division, squaring, and reduction error, which dominates when both profiles are nearly constant. | Closed test-first with an absolute common-scale-and-cell-count bound covering every fallback stage. Ambiguous genes remain exact through final rounding, while certified genes retain bounded vectorized evaluation. The public two-order fixture and a 48-gene, 320-digit/Fraction near-constant oracle are exact. |
| F-041 | Important | CP10k fallback turned the third entry of `[DBL_MAX, DBL_MAX, 2^-51]` into zero. The representable log1p and log2 results are respectively `0x0.00000000009c4p-1022` and `0x0.0000000000e17p-1022`. | The first fallback divided each count by the maximum and total before multiplying by 10,000. The tiny entry underflowed during that intermediate proportion even though its final CP10k value is representable. The log2 expression also formed `1 + x`, which discards subnormal `x`. | The first test-first correction retained each nonzero count as a mantissa/exponent term through multiplication by 10,000 and used `log1p(x) / log(2)`, closing the demonstrated fixture. It still materialized the completed CP10k term as float64 before the logarithm; F-045 completes the rounding guarantee at the smaller threshold where the log-base conversion changes the final float64 result. |
| F-042 | Important | On platforms where `longdouble` is not genuinely wider than float64, one unsafe endpoint still exact-recomputed every pair and every variance component. This made the portable route computationally incompatible with the bounded final design even though the wider-platform route was vectorized. | F-039 deliberately left the portable branch as an incremental exact memory fix; dispatch still bypassed its interval architecture entirely. | The first test-first correction added common-scaled float64 blocks, conservative float64-epsilon pair intervals, and exact escalation for cancellation. It closed the reviewed 300-by-96 and 900-by-8 pair performance probes, the 100-by-128 variance probe, the reviewed adjacent-extreme, centering, and subnormal endpoints, and bounded-memory behavior. The pairwise aggregate nevertheless discarded every safe pair's lower/upper bounds before final rounding; F-044 completes that final certification and extends the zero-exact-call performance lock to 1,200 by 64. |
| F-043 | Minor | A finite out-of-range `longdouble` p-pre-zero matrix leaked `FloatingPointError` or a promoted cast warning instead of the declared `PreZeroEvidenceError`. | `_probability_matrix` caught ordinary conversion exceptions but performed its narrowing cast outside a local floating-point translation boundary. | Closed test-first by trapping only cast overflow/invalid signals and translating them to the existing representability error. Shape, finite-range, and probability-domain validation are unchanged. |
| F-044 | Important | With `_WIDE_LONGDOUBLE=False`, the portable pairwise fallback returned `0x1.5ae5a5656fbbbp+755` for the reviewed two-cell fixture instead of the exact `0x1.5ae5a5656fbbcp+755`. A seeded 1,000-fixture exponent audit had 169 analogous exact-hex mismatches. | The per-pair float64 fallback computed conservative lower and upper bounds, but the aggregate discarded both bounds, summed only the approximate estimate, multiplied by the common scale, and converted that approximate mean directly to float64. | Closed test-first. The portable path streams outward lower/upper totals with compensated block accumulation, combines them with exact cancellation-only contributions as exact `Decimal` endpoints, and certifies that the complete mean occupies one binary64 rounding cell. If not certified, one bounded second pass retains at most 1,024 widest/most influential unresolved intervals, exact-refines them in batches of 64, updates the aggregate bounds, and repeats certification. Whole-endpoint exact evaluation is the final bounded-memory correctness path only after that cap. A post-suite diff audit rejected an initially overbroad exact-unit shortcut: fixed unit contributions now require original-coordinate proof of one `{0, ±common_scale}` axis and an exactly unchanged opposing vector. The reviewed fixture certifies after exactly one pair refinement; 1,024 seeded small exponent fixtures match a 320-digit whole oracle exactly; 300-by-96, 900-by-8, and 1,200-by-64 probes use zero exact pair calls; and the portable memory bound remains closed. |
| F-045 | Important | For `[DBL_MAX, DBL_MAX, 0x1.41418faa57d52p-64]`, the third log2(CP10k+1) result was zero, although the exact endpoint rounds to the minimum float64 subnormal; the exact natural-log endpoint correctly rounds to zero. A 20,000-case threshold audit had 5,547 log-base rounding mismatches. | The scaled fallback rounded the CP10k term to float64 before evaluating either logarithm. Information below the CP10k float64 threshold was therefore unavailable to the division by `ln(2)`, even when that completed logarithm was representable. | The first correction retained exact binary-rational counts and totals through dynamically precise logarithms for unsafe rows. F-046 extends certified `log1p(x)/ln(2)` evaluation to vulnerable finite rows and adds outward rounding bounds. The reviewed subnormal endpoints remain exact, and established moderate rows retain their legacy serialization whenever the wider interval certifies that cell. |
| F-046 | Important | Finite, representable log2-CP10k rows still lost or misrounded small nonzero terms. `[1e300, 1e280]` returned zero instead of `0x1.4ca9af0f3becep-53`; `[1, 2^-60]` returned `0x1.c21ef034d0783p-47` instead of `0x1.c2d79a6ff9d45p-47`. | The exact logarithm route was entered only after a floating-point signal. A finite row therefore retained `log2(1+x)` even when the addition discarded or rounded the small CP10k term before the logarithm. | Closed test-first. Every nonzero finite log2 cell is evaluated through cancellation-safe `log1p(x)/ln(2)` in wider precision with conservative outward bounds. A cell occupying one binary64 rounding cell is converted once; ambiguity escalates to an adaptive exact-binary-rational Decimal interval. Certified moderate legacy values remain unchanged. A 3,000-row/6,000-endpoint independent 350-digit audit found zero mismatches. |
| F-047 | Important | Finite ordinary mean, population-variance, and pairwise-distance paths accepted small nonzero cancellation as zero or the wrong adjacent float. The reviewed endpoints were respectively `0x1p-53`, `0x1.8p-53`, and `0x1.6a09e667f3bcdp-53`, but the ordinary formulas returned zero, `0x1p-52`, and zero. | Each ordinary path accepted a finite scalar without certifying the joint subtraction and final rounding cell; this affected small nonzero cancellation, not only endpoints that had already become zero. | Closed test-first in both operand orders. Mean distortion now forms a joint wider interval and uses an exact linear-time Fraction fallback only when its final cell is uncertified. Variance and pairwise distance submit every finite ordinary candidate to the existing bounded wider/portable interval and ambiguity-refinement machinery, returning the legacy scalar only when the certified result agrees. Outward bounds cover interval and accumulated rounding. Sixty-four committed randomized adjacent cases per order and an independent 5,000-case public-metric audit match exact Fraction/320-digit oracles. A 160-by-48 safe fixture uses zero exact mean/pair calls and one ambiguous-gene refinement in under ten seconds. |
| F-048 | Important | Inverse log1p-CP10k conversion multiplied before dividing by 10,000. With observed library `DBL_MAX` and stored native `log1p(5000)`, the intermediate overflowed and the representable endpoint was rejected; reordering in float64 would still produce the one-ULP-low `0x1.0000000000001p+1023` instead of `0x1.0000000000002p+1023`. | The converter treated intermediate float64 overflow as endpoint overflow and had no single-rounding path for a mathematically representable scaled product. | Closed test-first with a shared observed-library inverse boundary. Safe products retain the exact legacy operation order. Only multiplication-risk cells use adaptive Decimal `expm1(native) * library / target`, outward decimal bounds, and one final binary64 conversion. The analogous scSDAE log1p-CPM endpoint now uses the same boundary. A 20,000-case independent 250-digit risky-endpoint audit found zero mismatches. |
| F-049 | Important | Unsafe but finite reconstruction fallbacks rounded before the completed estimand. MAE and empirical Wasserstein distance for `[DBL_MAX, predecessor(DBL_MAX)]` versus zero returned `DBL_MAX` rather than `0x1.ffffffffffffep+1023`; an extreme gNRMSE returned `0x1.6a09e667f3bccp+0` rather than `0x1.6a09e667f3bcdp+0`. | `_scaled_signed_differences` rounded normalized per-entry differences, then later reductions and scale restoration rounded again. The same architecture could erase a tiny residual while separately forming overflowed row means. | Closed test-first. Exceptional MSE, MAE, Wasserstein, and mean endpoints retain exact binary-rational sums through the completed estimator and convert once. Exceptional gNRMSE retains exact squared-error and variance ratios and uses directed, adaptive Decimal square-root means until one binary64 cell is certified. The inaccurate scaled-difference path and all consumers were removed. Ordinary finite branches remain unchanged. Both operand orders retain the reviewed `2e-301` mean residual, and bounded unsafe 300-by-96 and 900-by-8 probes complete with bounded memory. |
| F-050 | Important | Identical matrices containing only the minimum float64 subnormal produced unavailable mean distortion instead of exact available zero. | The certified mean interval widened a zero estimate by an underflow-scale uncertainty and then classified the positive upper endpoint rounding to zero as an unavailable positive underflow. It did not recognize an exactly identical input before approximate interval construction. | Closed test-first with an exact array-identity shortcut at the mean-distortion owner. Identical finite matrices return available zero without approximate arithmetic; nonidentical overflowed means continue through the exact endpoint path and retain the reviewed tiny residual. |
| F-051 | Important | Protocol-scale log2-CP10k conversion repeated a row copy, wider row sum, and certification for every nonzero cell; conservative ambiguity then invoked adaptive Decimal 120,000 times for an all-ones 100-by-1,200 matrix. The 300-by-96 and 900-by-8 bounded probes made 28,800 and 7,200 analogous calls. | Certification was cell-oriented even though every cell shared one row denominator, and ambiguous equal values were recomputed independently. | Closed test-first. Each row is copied and certified in one vectorized wider operation. Only unresolved nonzero cells use the exact route; their row denominator is constructed once and equal values share one adaptive Decimal result. The 100-by-1,200, 300-by-96, and 900-by-8 regressions cap exact work at one call per row, retain exact expected serialization, and enforce bounded runtime and memory. |
| F-052 | Minor | `scsdae_to_evaluator_counts` leaked `FloatingPointError` or a promoted cast warning when a finite `longdouble` native output lay outside float64 range. | Its direct narrowing cast did not use the already established local native-boundary signal translation. | Closed test-first. The scSDAE native cast now traps only overflow and invalid narrowing signals and translates them to the existing `ValueError` representability contract. Shape, numeric, finite, nonnegative, and inverse-scale semantics are unchanged. |
| F-053 | Important | The shared observed-library inverse rejected every native log1p value above 1,000 even when a minimum-subnormal observed library made the completed CP10k or CPM endpoint finite. Native value 1,001 must produce `0x1.cd72a103bf4cdp+356` at target 10,000 and `0x1.27539a3fd6979p+350` at target 1,000,000. | A native-value-only cutoff assumed `exp(value)` alone determined endpoint representability and ignored the library and target factors. | Closed test-first. A conservative combined-log lower bound rejects only endpoints proven far above the binary64 overflow boundary before any Decimal exponential. Remaining risky cells use adaptive, outward Decimal `expm1(value) * library / target` bounds and convert once after both bounds occupy one binary64 cell. Both exact endpoints, 128 randomized large-native high-precision oracles, and bounded rejection of a 32-by-32 absurd-native fixture pass. |
| F-054 | Important | The exact pairwise distance fallback returned `0x1.a1a92b184e548p+1021` for the reviewed one-pair fixture instead of `0x1.a1a92b184e547p+1021` in both wider and portable modes. | Exact rational operands entered Decimal square roots and division under a fixed 120-digit context. The exact result lies unusually close to a binary64 boundary, and the outer 120-digit accumulator re-rounded a higher-precision correction. | Closed test-first. Rational radicands, numerator, and scale now enter directed lower/upper Decimal square-root and division intervals whose precision doubles until both endpoints share one binary64 cell. Exact Decimal additions preserve every certified digit, final division is adaptive, and exact-only wider/portable aggregates avoid the former outer-context re-rounding. Both execution modes retain the required exact hexadecimal result, and the existing 1,024-fixture whole-Decimal oracle remains exact. |
| F-055 | Important | Exact identity still sent a 100-by-48 extreme matrix through pair generation and 4,950 exact norm-difference calls before returning zero. | Pairwise distance had no exact identity dispatch before its ordinary and fallback arithmetic. | Closed test-first. After the pair denominator is established, exact array identity returns available zero before any distance or exact helper work. The public evaluator has already converted and mask-validated both operands, so the shortcut does not invoke an input protocol or bypass the approved dense boundary. The 100-by-48 extreme fixture makes zero exact helper calls; existing identity checks for mean and all other public reconstruction endpoints remain available and exact. |
| F-056 | Important | Under a caller `np.errstate(all="raise")`, probabilities `[minimum subnormal, nextafter(1, 0)]` leaked `FloatingPointError` from calibration instead of returning a complete score row with reason-coded `calibration_fit_failed` coefficients. | The numerically stable sigmoid intentionally permits an exponential to underflow to zero, and IRLS weight/information products can intentionally underflow, but both inherited the caller's floating-point policy. | Closed test-first. Underflow is ignored only around the sign-split sigmoid exponential and the IRLS weight/information products where zero is the limiting mathematical value. Other floating-point signals retain their strict behavior, the singular optimizer enters the existing reason-coded failure path, and Brier, log loss, reliability, and the complete score record remain available without a warning or exception. |
| F-057 | Important | Three technical-view effects `[DBL_MAX, -DBL_MAX, 2^-52]` collapsed to zero before biological-draw inference; their exact mean is `0x1.5555555555555p-54`. The observed median, every bootstrap replicate, and both interval endpoints therefore lost a representable residual. | `_finite_mean` divided every input by the largest magnitude before summation, which underflowed the small residual before the maximum terms cancelled. Some input orders also make `math.fsum` raise on an intermediate even when exact cancellation leaves a representable raw sum. | Closed test-first. Stable aggregation first attempts raw `math.fsum`, uses exact binary-rational cancellation when that sum is finite or an intermediate overflow may conceal a representable raw total, and uses the established scaled fallback only when the exact raw sum is truly unrepresentable. The public three-view result retains the exact median, bootstrap distribution, and interval values under strict caller state; probability and win/tie/loss semantics remain unchanged. A 256-fixture randomized order/exponent oracle and ordinary equality controls pass. |
| F-058 | Important | A five-cell, one-gene pairwise fixture returned `0x1.128ba744285dbp+994` instead of the whole-estimand result `0x1.128ba744285dcp+994` in public, wider, portable, and nominally exact modes. | Every exact pair was individually certified, replaced by a Decimal midpoint, and only then summed. Per-pair float-cell certification did not certify the rounding cell of their completed mean. | Closed test-first. Directed lower and upper pair endpoints now survive exact accumulation and combination with wider or portable approximate bounds. If those complete endpoints do not occupy one binary64 cell, precision doubles over the whole bounded-memory endpoint; pair midpoints are not summed. Existing zero-exact-call, refinement-count, runtime, and memory regressions remain green. |
| F-059 | Important | Under strict caller floating-point state, the reviewed two-entry probability row aborted while squaring the minimum-subnormal Brier residual, then again in the calibration gradient. The representable Brier result is `0x1p-107` with denominator two, while calibration must be reason-coded `calibration_fit_failed`. | Expected limiting underflow was localized for the sigmoid and information matrix but not for Brier squaring or the IRLS gradient product. | Closed test-first. Brier ignores underflow only while forming bounded squared residuals and exact-recomputes the completed mean when a nonzero residual is lost. The calibration gradient joins the already narrow underflow-localized IRLS operations. Brier, log loss, ECE, and reliability complete; calibration retains its declared failure reason. |
| F-060 | Important | Stable means with an unrepresentable raw sum rounded twice: every permutation of `[DBL_MAX, DBL_MAX, 0]` returned `0x1.5555555555554p+1023` instead of `0x1.5555555555555p+1023`. The mean and even median of `DBL_MAX` and its predecessor rounded upward to `DBL_MAX` instead of ties-to-even `0x1.ffffffffffffep+1023`. | The remaining overflow path divided scaled binary64 inputs and multiplied the rounded normalized mean by the scale. | Closed test-first by retaining exact binary-rational inputs through the completed sum and division, then converting once. This is the same exact route already used for ordinary and cancellation means, so established exactly rounded ordinary values are unchanged. A 100,000-fixture signed exponent/order audit found zero mean or median oracle mismatches. |
| F-061 | Important | Paired relative effects could round the subtraction or ratio before the completed estimand. The reviewed finite pair returned one ULP low instead of `-0x1.fffe840932394p-1`. | Same-sign and opposite-sign branches used different staged float64 formulas, neither of which guaranteed one rounding of `(method-comparator)/abs(comparator)`. | Closed test-first with exact binary-rational subtraction and division followed by one float64 conversion. Positive and negative comparator directions, zero method, zero comparator exclusion, exact ties, nonrepresentable effects, and 100,000 randomized signed finite pairs retain their declared policies with zero oracle mismatches. |
| F-062 | Important | Overflow-safe quantile interpolation and sample variance still double-rounded completed estimands. The reviewed 2.5% quantile was one ULP low and the reviewed three-value sample variance was one ULP high. | Quantiles interpolated scaled rounded endpoints; sample variance rounded the normalized mean, centered squares, normalized variance, and restored scale separately. | Closed test-first. Linear interpolation now retains exact binary-rational endpoints and its float64 interpolation weight through one final conversion. Sample variance retains the exact centered sum and degrees-of-freedom division through one final conversion, returning `None` only for positive underflow or overflow as before. Strict-state endpoints, exact zero, unrepresentable variance, and 100,000 randomized quantile and variance oracles pass. |
| F-063 | Important | The exceptional gNRMSE route could falsely certify the upper adjacent binary64 value for an exact square root lying just below their midpoint. For `a=nextafter(2^200,+inf)`, `b=nextafter(a,+inf)`, and squared ratio `((a+b)/2)^2-1`, its nominal floor and ceiling endpoints were the same half-even Decimal square root and the 120-digit route returned `b`; a 2,000-digit oracle rounds to `a`. If the fixed precision sequence remained ambiguous, the route also returned an uncertified midpoint. | Python Decimal square root is correctly rounded with `ROUND_HALF_EVEN` regardless of the context's `ROUND_FLOOR` or `ROUND_CEILING` setting. The code treated those identical half-even results as directed bounds, then had a midpoint fallback without a proved binary64 cell. | Closed test-first. Each exact rational radicand is first rounded in the requested direction, each half-even square root is stepped one Decimal value outward, and the roots and completed mean are accumulated and divided under the same direction. Precision now doubles through the completed mean until both endpoints certify one binary64 cell; exhaustion raises instead of serializing an uncertified midpoint. A 1,000-fixture multi-term, broad-exponent oracle audit found zero enclosure or rounding mismatches. The only analogous production Decimal-square-root path already used explicit predecessor/successor bounds. |
| F-064 | Minor | `MetricValue(True, 1, None)` was accepted as the numeric value `True`, and whitespace-only unavailable reasons were accepted as meaningful schema states. | Python Boolean values inherit from `int`, while unavailable-reason validation checked only string length before any whitespace semantics. | Closed test-first. Python and NumPy Boolean metric values are rejected explicitly, and unavailable reasons must contain at least one non-whitespace character. Valid finite numeric values and nonblank reason-coded unavailable states are unchanged. |
| F-065 | Minor | `_fraction_to_decimal` and `_decimal_divide_adaptive` remained as unowned private arithmetic helpers with no production or test callers. | Earlier numerical corrections superseded both helpers without removing the obsolete definitions. | Closed by deleting the two unreachable private helpers after repository-wide caller search. No owning arithmetic path was redirected through them. |
| O-004 | Minor | The inherited CUDA library path caused the five baseline runtime-environment failures; one later temporary-venv inventory rebuild fluctuated once in the excluded transient-runtime-swap test. | The shell path resolves through intentionally rejected symlinks; the isolated temporary-runtime inventory changed between its two probes on one attempt. | No code change. The exact transient node passed unchanged on immediate isolated rerun, and the authoritative suite passed with only the inherited CUDA path removed. |

No unresolved metric-domain, statistical-independence, evaluation-row,
external-reference, manifest, or pre-zero evidence defect was demonstrated
after F-025 through F-065. Legacy runtime-lock, filesystem-hardening, and
outer-provenance mechanisms were not redesigned or extended.

### Task 5 test-first and verification evidence

The focused pre-correction nodes produced 13 expected failures and one
downstream-type-error control pass. After the minimal corrections and final
formatting, the combined focused set reported:

```text
14 passed in 2.40s
```

The overflow regression passed with runtime warnings promoted to errors. The
complete metrics and statistics files reported:

```text
80 passed in 0.83s
```

The exact transient-runtime-swap node passed unchanged after the one
fluctuating full-suite attempt:

```text
1 passed in 193.68s (0:03:13)
```

The exact sanitized five-file Task 5 suite passed at the final formatted
state:

```text
335 passed, 1 skipped in 976.13s (0:16:16)
```

The independent correction began with six focused RED nodes for sparse
storage and stable derived arithmetic:

```text
6 failed in 2.57s
```

The identical focused set then passed:

```text
6 passed in 1.62s
```

The warnings-as-errors metrics, pre-zero, and public sparse-boundary set
passed at the final production state:

```text
131 passed in 16.74s
```

The first exact correction suite exposed only an ordinary-data checkpoint
rounding difference after all new semantic nodes had passed:

```text
1 failed, 342 passed, 1 skipped in 971.34s (0:16:11)
```

The safe ordinary gNRMSE/correlation branch restored the established literals,
and the previously failing checkpoint node passed:

```text
1 passed in 144.89s (0:02:24)
```

The exact sanitized five-file Task 5 suite then passed at the final formatted
correction state:

```text
343 passed, 1 skipped in 973.31s (0:16:13)
```

A second independent review demonstrated F-032 through F-035. Its focused
pre-correction invocation produced all eleven expected failures: three
underflow/strict-state failures, ordinary-value and tie drift, two CP10k
library failures, two runner cast failures, and both stored-target cast
failures. After the exact-first correction, the expanded focused set reported:

```text
13 passed in 2.57s
```

The complete metrics, observed-method, and pre-zero owning files reported:

```text
185 passed, 14 skipped in 18.88s
```

The exact sanitized five-file Task 5 suite passed at the final formatted
production state:

```text
353 passed, 1 skipped in 969.12s (0:16:09)
```

Targeted Ruff lint and formatting, byte compilation of all changed production
and test files, and `git diff --check` passed.

A final independent numerical review then demonstrated F-036 and F-037. Its
exact-hex, warnings-as-errors focused invocation produced all six expected
failures: both operand orders for variance cancellation, both operand orders
for norm cancellation, and both native-output conversion routes. After the
joint-cancellation and narrow cast corrections, the identical focused set
reported:

```text
6 passed in 1.94s
```

The complete metrics, core-method adapter, and pre-zero owning files then
reported:

```text
191 passed, 14 skipped in 21.54s
```

The single final exact sanitized five-file Task 5 suite, run after formatting
and all scoped static checks, reported:

```text
357 passed, 1 skipped in 958.56s (0:15:58)
```

Before that final suite, scoped Ruff lint and formatting, byte compilation,
and `git diff --check` all passed. No production or test file changed after
those gates or the exact suite.

A final narrow double-rounding review then demonstrated F-038. Its exact
minimum-subnormal regressions produced all four expected failures under
warnings-as-errors: both operand orders for variance distortion and both
operand orders for pairwise distance distortion. After preserving exact or
high-precision state through the completed aggregate, the identical set
reported:

```text
4 passed in 0.23s
```

The complete metrics, core-method adapter, and pre-zero owning files then
reported:

```text
195 passed, 14 skipped in 12.89s
```

After formatting and all scoped static checks, the final correctly sanitized
five-file Task 5 suite reported:

```text
361 passed, 1 skipped in 1196.70s (0:19:56)
```

Two earlier broad commands were stopped after the four publication-spawn
tests correctly rejected command-level nondefault Python warning/no-user-site
flags. Neither attempt demonstrated a production or test defect. The final
command retained default interpreter flags and removed only the diagnosed
inherited CUDA library path.

The final performance review then demonstrated F-039. Its focused pre-change
run produced all six expected failures in 27.33 seconds: the interval
classifier was absent, the 20-by-128, 60-by-16, 100-by-16, and 300-by-96
unsafe pairwise probes made 190, 1,770, 4,950, and 44,850 exact calls, and the
unsafe variance probe exact-recomputed all 128 genes. The identical focused
set passed after bounded block evaluation and ambiguity-only escalation:

```text
6 passed in 0.55s
```

The prior large/subnormal endpoints in both operand orders, ordinary exact
serialization, warnings-as-errors extremes, and the hand-calculated endpoint
then passed together:

```text
17 passed in 0.82s
```

The complete metric file passed:

```text
73 passed in 1.02s
```

A separate portable-route memory regression reconstructed the removed
all-pair collection at a 1,549,047-byte traced peak and failed its 256,000-byte
bound. The incremental exact route passed the same node in 4.69 seconds.

Formatting and static gates were completed before the single final suite:

```text
Ruff check: All checks passed
Ruff format --check: 2 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The metrics, statistics, core-method adapter, and pre-zero owning files then
passed:

```text
234 passed, 14 skipped in 26.05s
```

The single final correctly sanitized five-file Task 5 suite reported:

```text
368 passed, 1 skipped in 980.11s (0:16:20)
```

No production or test file changed after the formatting/static gates or this
exact final suite.

A final portability and interval review then demonstrated F-040 through F-043.
After correcting one test-only missing import, the focused pre-change
invocations produced all eight expected behavioral failures: the two variance
operand orders, the near-constant interval oracle, two portable pairwise
sizes, portable variance exact-call scaling, the CP10k tiny term, and the
p-pre-zero cast signal. The adjacent-extreme portable pair control already
escalated correctly.

The corrected variance interval contains the exact result for every one of 48
random near-constant genes under both exact `Fraction` comparison and a
320-digit decimal projection. Both reviewed fixture orders produce
`0x1.c71c71c71c71cp-101`. The portable 300-by-96 and 900-by-8 pair probes use
no exact pair calls; the 100-by-128 variance probe uses at most one exact gene
while retaining the exact oracle. Portable adjacent-extreme, centering, and
subnormal probes remain exact under warnings-as-errors.

The CP10k regression retains the tiny count through the final conversion and
produces the exact required log1p and log2 hexadecimal values. The public
p-pre-zero evaluator translates a strict out-of-range narrowing signal to its
declared error. The combined exact-oracle, ordinary-serialization, mask, cast,
portable-performance, and escalation acceptance set reported:

```text
30 passed in 2.94s
```

The metrics, core-method adapter, and pre-zero owning files reported:

```text
214 passed, 14 skipped in 20.43s
```

Formatting and static gates were completed before the single final suite:

```text
Ruff check: All checks passed
Ruff format --check: 6 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The single final correctly sanitized five-file Task 5 suite reported:

```text
408 passed, 14 skipped in 968.76s (0:16:08)
```

No production or test file changed after those formatting/static gates or the
final suite.

A final portable-aggregation and transform-threshold review then demonstrated
F-044 and F-045. The focused pre-change invocation produced the four expected
failures and one already-correct aggregation control pass: the reviewed
portable pair endpoint was one ULP low, the seeded portable oracle stopped at
its first mismatch, and both the exact log2 threshold fixture and randomized
threshold oracle exposed premature zero. After the initial corrections, the reviewed
fixture, 1,024 seeded 320-digit whole pairwise oracles, aggregation control,
exact log-base threshold, and 1,024 seeded 800-digit threshold oracles
reported:

```text
5 passed in 27.29s
```

The complete metric file, including portable bounded-memory and zero-exact-call
performance probes at 300-by-96, 900-by-8, and 1,200-by-64, reported:

```text
88 passed in 8.04s
```

The expanded metrics, core-method adapter, and pre-zero owning set initially
reported:

```text
220 passed, 14 skipped in 43.92s
```

An independent 20,000-case threshold audit compared both transforms against an
800-digit decimal oracle in bounded 1,000-row chunks and reported:

```text
20,000 audited, 0 mismatches
```

Formatting and the scoped static gates were completed before the one final
sanitized suite:

```text
Ruff check: All checks passed
Ruff format --check: 4 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The first sanitized five-file Task 5 suite then reported:

```text
383 passed, 1 skipped in 977.91s (0:16:17)
```

Mandatory post-suite diff inspection then demonstrated that the new exact-unit
performance shortcut was too broad. For original imputed coordinates
`[1.5, -3*2^-54]` and an unchanged zero truth vector, scaled subtraction rounds
to exactly one although the original-coordinate distance must round upward to
`0x1.8000000000001p+0`. The isolated test produced the expected failure:

```text
1 failed in 0.45s
```

The shortcut now requires proof from original coordinates that the changing
vector has exactly one `{0, ±common_scale}` axis and that the opposing vector
is exactly unchanged. The corrected exact-unit node, reviewer fixture,
1,024-fixture whole oracle, aggregation control, and both threshold-log tests
reported:

```text
6 passed in 25.21s
```

The corrected expanded owning set reported:

```text
221 passed, 14 skipped in 46.50s
```

Formatting and scoped static gates were repeated after the exact-unit
correction and before the replacement authoritative suite:

```text
Ruff check: All checks passed
Ruff format --check: 4 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The replacement final-state sanitized five-file Task 5 suite reported:

```text
384 passed, 1 skipped in 973.79s (0:16:13)
```

No production or test file changed after those replacement formatting/static
gates or the final-state suite. Only this ledger evidence and the ignored
scratch correction report followed.

A definitive finite-row and ordinary-cancellation review then demonstrated
F-046 through F-048. The exact public RED set produced eleven expected
failures and one already-correct protocol-sized control pass: two finite-row
log2 endpoints, the risky inverse CP10k endpoint, and mean, variance, and
pairwise cancellation in both operand orders plus the two randomized-order
nodes. The analogous scSDAE inverse endpoint then failed separately before it
was moved to the shared boundary.

The corrected exact public set, including 64 committed randomized adjacent
fixtures in each operand order and the protocol-sized refinement bound,
reported:

```text
13 passed in 2.94s
```

Three independent high-precision audits then reported:

```text
3,000 finite rows / 6,000 log2 endpoints, 0 mismatches
5,000 adjacent ordinary public metric fixtures, 0 mismatches
20,000 risky inverse-log1p endpoints, 0 mismatches
```

The expanded metric and core-method adapter owning files reported:

```text
156 passed, 14 skipped in 32.75s
```

Formatting and static gates were completed before the single replacement
authoritative suite:

```text
Ruff check: All checks passed
Ruff format --check: 5 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The final-state sanitized five-file Task 5 suite reported:

```text
393 passed, 1 skipped in 983.42s (0:16:23)
```

No production or test file changed after those formatting/static gates or the
final suite. Only this ledger evidence and the ignored fix-8 report followed.

A final endpoint and protocol-scale review then demonstrated F-049 through
F-052. The exact focused pre-change invocation produced eight expected
failures and two already-correct residual controls: both MAE/Wasserstein
operand orders, the extreme gNRMSE endpoint, exact-zero minimum-subnormal mean
distortion, all three protocol-size CP10k call-count nodes, and the scSDAE
native cast boundary. After the exact/certified endpoint corrections,
vectorized row certification, and narrow cast translation, the identical set
reported:

```text
10 passed in 3.72s
```

The complete metric and core-method adapter owning files then reported:

```text
166 passed, 14 skipped in 35.24s
```

Independent numerical checks retained every one of 5,000 ordinary endpoint
tuples, matched 3,000 finite log2-CP10k rows and all 6,000 of their endpoints
against 350-digit oracles, and matched 20,000 risky inverse-log1p endpoints
against 250-digit oracles. The unsafe 300-by-96 and 900-by-8 exact-fallback
probes completed in 1.766 and 0.567 seconds with traced peaks below one
megabyte; the direct 100-by-1,200 all-ones CP10k probe completed in one second
including interpreter startup.

Formatting was applied before the final verification sequence. Scoped Ruff
lint and format checking, byte compilation, and `git diff --check` all exited
zero. The single final correctly sanitized five-file Task 5 suite then
reported:

```text
399 passed, 1 skipped in 979.32s (0:16:19)
```

The post-format CP10k oracle/performance and scSDAE cast set separately
reported 10 passes in 21.97 seconds. No production or test file changed after
the formatting/static gates or final five-file suite; only this ledger
evidence and the ignored fix-9 report followed.

A tenth independent numerical review then demonstrated F-053 through F-057.
The corrected inverse fixtures first failed independently with two `NaN`
results. In the initial combined invocation, the two first-draft inverse tests
were rejected earlier by the established integer-count fixture contract and
were corrected before production work; the remaining five exact behavioral
nodes all failed for their expected reasons: both pairwise modes were one ULP
high, exact identity made 4,950 helper calls, calibration leaked
`FloatingPointError`, and the three-view effect became zero. After correcting
the five owning boundaries, the complete focused set reported:

```text
7 passed in 1.87s
```

The 128-fixture large-native inverse oracle and bounded absurd-native
rejection, the existing 1,024-fixture whole-Decimal pairwise oracle, ordinary
reconstruction serialization, and protocol-size exact-refinement controls
passed. A 256-fixture randomized cancellation audit then exposed one
input-order case where `math.fsum` raised before exact cancellation; after the
exact representable-raw-sum fallback, the expanded oracle and ordinary-control
set reported:

```text
6 passed in 10.14s
```

The complete metric, statistics, and core-method adapter owning files then
reported:

```text
207 passed, 14 skipped in 38.93s
```

Formatting and the scoped static gates completed before the one authoritative
suite:

```text
Ruff check: All checks passed
Ruff format --check: 6 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The single final correctly sanitized five-file Task 5 suite reported:

```text
405 passed, 1 skipped in 980.72s (0:16:20)
```

No production or test file changed after those static gates or the final
five-file suite; only this ledger evidence and the ignored fix-10 report
followed.

An eleventh independent numerical review then demonstrated F-058 through
F-062. The exact focused pre-change invocation produced thirteen expected
failures: all four pairwise modes, the strict Brier/calibration row, three
unrepresentable-sum mean permutations, the extreme even mean/median, the
reviewed relative effect and randomized signed oracle, and the reviewed
quantile and sample-variance endpoints. After correcting the five owning
arithmetic boundaries, the post-format focused set reported:

```text
13 passed in 0.44s
```

The complete metric and statistics owning files, including the existing
pairwise exact-call, runtime, and traced-memory bounds, then reported:

```text
155 passed in 12.82s
```

Four independent 100,000-fixture signed exponent audits compared mean/median,
relative effect, quantile, and sample variance against exact binary-rational
oracles. Each reported zero mismatches; their elapsed times were respectively
9.386, 5.598, 6.986, and 15.443 seconds.

Formatting and scoped static gates completed before the one authoritative
suite:

```text
Ruff check: All checks passed
Ruff format --check: 4 files already formatted
Scoped compileall: exit 0
git diff --check: exit 0
```

The single final correctly sanitized five-file Task 5 suite reported:

```text
418 passed, 1 skipped in 983.96s (0:16:23)
```

No production or test file changed after those static gates or the final
five-file suite; only this ledger evidence and the ignored fix-11 report
followed.

A twelfth independent numerical and API review then demonstrated F-063
through F-065. The exact focused pre-change invocation reported four expected
failures and one existing-control pass: Python Boolean metric values,
whitespace-only unavailable reasons, the `2^200` square-root boundary, and the
uncertified gNRMSE midpoint failed; the NumPy Boolean parameter already entered
the existing numeric-type rejection. After the owning corrections, the same
parameterized set reported:

```text
5 passed in 0.32s
```

The complete metrics and statistics owning files, including the established
ordinary-serialization, exact endpoint, strict floating-point-state, and
performance controls, reported:

```text
160 passed in 12.96s
```

A separate 1,000-fixture audit formed multi-term exact binary-rational
radicands over exponents from -500 through 500, compared both directed bounds
against 1,000-digit Decimal oracles, and compared each certified binary64
result. It reported zero enclosure or rounding mismatches. The repository-wide
square-root search confirmed that the pairwise Decimal path already steps its
half-even square roots outward. The ordinary gNRMSE branch and the reviewed
log1p-CP10k, safe Brier, and inverse-log1p formulas were not changed.

Formatting and scoped static gates completed before the authoritative suite:

```text
Ruff check: All checks passed
Ruff format --check: 2 files already formatted
Scoped compileall: exit 0
Dead-helper search: no callers or definitions remain
git diff --check: exit 0
```

The single final correctly sanitized five-file Task 5 suite reported:

```text
423 passed, 1 skipped in 989.64s (0:16:29)
```

No production or test file changed after those static gates or the final
five-file suite; only this tracked ledger evidence and the ignored fix-12
report followed.

These checks establish bounded-fixture evaluation contracts only. No real
scientific or comparator workload ran, and no empirical competitiveness,
external-runtime availability, or publication-readiness claim is made.

## Task 6 selection, revision, freeze, and post-freeze authority audit

Task 6 began from clean commit
`dbd766618e2b62b6c7a0295036ff8e02b9699dc0`. It traced direct-comparator
evidence through selection, promotion, bounded revision, freeze, final
planning, trajectory and downstream evaluation, scaling, final analysis, and
publication synthesis. Fixed scheduled and numerical populations, revision
limits, final seed denominators, one-use lifecycle, unavailable-estimand
suppression, confidence direction, and multiplicity permissions remain
unchanged.

No development selector was found to consume final performance, downstream
endpoints, final data, or post-freeze results. After the corrections below, no
additional reachable selection, revision, execution, or claim-permission
bypass remained.

### Task 6 findings

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-066 | Important | Dense and sparse masked values could cross downstream method-output, held-out truth, trajectory, and group-marker boundaries after coercion erased missingness. | Closed test-first by rejecting masked containers before coercion, including sparse stored data. |
| F-067 | Important | Finite extended-precision method or trajectory values outside float64 range leaked `FloatingPointError` rather than the declared validation failure. | Closed by strict local narrowing with stable `ValueError` translation and no clipping. |
| F-068 | Important | Direct held-out CP10k normalization overflowed on extreme finite rows even when the normalized output was representable. | Closed by using the reviewed count-equivalent conversion for nonzero rows while preserving zero rows. |
| F-069 | Important | Typed final authorities admitted unknown action text and incompletely constrained method IDs, seed denominators, and nonexecution reasons. | Closed by enforcing the exact two-action vocabulary, safe IDs, exact unique seed tuples, and nonblank reasons. |
| F-070 | Important | Selected direct evidence and frozen comparator nonexecution mappings were only shallow snapshots and remained mutable through caller-owned nested values. | Closed by recursively normalizing and detaching nested direct values before exposing immutable mappings. |
| F-071 | Important | Boolean values could impersonate integer schema versions, downstream independent counts, and completed endpoint values. | Closed with exact integer schema/count checks and explicit Python/NumPy Boolean value rejection. Existing frozen-publication reconstruction was separately confirmed fail-closed. |
| F-072 | Minor | The two-writer scaling regression coupled its lock assertion to large-artifact write and replay speed, producing a 20-second timeout while the process was CPU-bound. | Closed in the test by pausing at the already-validated publish boundary. Cross-instance serialization and stale-prefix rejection remain unchanged; no production lock behavior changed. |

### Task 6 verification

The initial focused mutation set reported:

```text
10 failed in 27.65s
```

After the owning corrections, the same set reported:

```text
10 passed in 27.66s
```

Adjacent RED/GREEN review separately covered Boolean endpoint count/value,
nested selected-map mutation, Boolean frozen final-plan receipt, trajectory
narrowing, and Boolean scaling checkpoint/dataset-receipt schemas. The complete
downstream evaluation owning file reported:

```text
43 passed in 2.17s
```

The first authoritative 15-file run reported:

```text
1 failed, 829 passed in 7096.09s (1:58:16)
```

Its sole failure was F-072. The same node reproduced independently at the same
future deadline:

```text
1 failed in 133.94s (0:02:13)
```

After the test-only pause-point correction, the isolated node reported:

```text
1 passed in 183.04s (0:03:03)
```

Scoped Ruff lint and format checking, byte compilation, and
`git diff --check` all exited zero before the new clean final-state suite. The
sanitized 15-file authority and post-freeze suite then reported:

```text
830 passed in 7065.83s (1:57:45)
```

No production or test file changed during or after that final suite; only this
ledger evidence and the ignored Task 6 report followed. The fixed populations,
estimands, revision allowance, execution lifecycle, runtime lock, legacy outer
provenance, and existing content-validation mechanisms were not redesigned.

These checks establish authority and claim contracts over bounded fixtures.
No real scientific or comparator workload ran, and no empirical
competitiveness, external-runtime availability, or Genome Biology submission
readiness claim is made.

### Task 6 independent-acceptance corrections

An independent acceptance review of Task 6 at
`ec08b1f4c74b1224d57f9f71d63c0a650e301a52` demonstrated four additional
important validation gaps.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-073 | Important | The final manifest accepted `failed_count=False` because Boolean zero compared equal to integer zero. | Closed test-first by requiring exact integer types for every manifest count field. |
| F-074 | Important | Typed final authorities and frozen applicability accepted arbitrary nonblank non-run reasons. | Closed test-first with the existing publication-freeze reason vocabulary enforced according to method/configuration disposition at constructors and receipt reconstruction. |
| F-075 | Important | Recursive direct normalization silently changed a nested integer mapping key into text before freezing it. | Closed test-first by rejecting non-string mapping keys before direct serialization or thawing; valid immutable snapshots are unchanged. |
| F-076 | Important | Huge positive or negative Python integers leaked `OverflowError` from direct endpoint and persisted endpoint validation. | Closed test-first with declared `ValueError` and `DownstreamEvidenceError` translation at the owning value/alpha and persisted-row boundaries. |

The first focused acceptance set reported `9 failed, 3 passed in 71.36s`.
The adjacent family-alpha case separately reported `2 failed in 2.01s`.
After correction, the complete focused set reported
`14 passed in 78.93s (0:01:18)`.

Complete owning files reported:

```text
tests/test_downstream_evaluation.py: 48 passed in 2.38s
tests/test_final_runner.py: 122 passed in 667.35s (0:11:07)
tests/test_downstream_evidence.py: 149 passed in 82.26s (0:01:22)
```

The adjacent freeze, final-analysis, publication-synthesis, and
development-evaluation files reported:

```text
266 passed in 2717.82s (0:45:17)
```

After formatting and static checks, the exact sanitized fifteen-file Task 6
suite from the brief reported:

```text
843 passed in 6033.96s (1:40:33)
```

No production or test file changed during that execution. Only this ledger
evidence and the matching Task 6 correction report were updated afterward.
No selected method, final population, seed denominator, revision allowance,
estimand, runtime behavior, or claim permission changed. No real scientific
workload ran, and no new checksum, fingerprint, content-summary,
filesystem-hardening, or cyber-related mechanism was added.

### Task 6 final-acceptance corrections

A final independent acceptance review of Task 6 at
`1da78e70cf3a029d0c6f8edf70c5bbbbd086f26d` demonstrated four additional
important contract gaps.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-077 | Important | `DirectReconstructionEvidence` retained caller-owned nested plan, input, record, and selected-configuration values, including through revision reconstruction. | Closed test-first with constructor-owned recursive canonical snapshots that retain established read-only mapping access. |
| F-078 | Important | A malformed tuple-backed `FrozenDirectObject` could reach freeze, thaw, or serialization without validating pair shape, string keys, uniqueness, and canonical order. | Closed test-first by validating the wrapper at every direct-value boundary while preserving valid wrappers. |
| F-079 | Important | Persisted final, downstream, and scaling authorities accepted Boolean aliases for some integer schema, count, and ordinal fields. | Closed test-first with exact built-in integer validation at the owning loaders. |
| F-080 | Important | Huge positive or negative Python integers leaked `OverflowError` from endpoint, null-DE alpha, and persisted scaling numeric validation. | Closed test-first by translating invalid or unrepresentable values to each boundary's declared contract exception. |

The initial focused acceptance set reported:

```text
17 failed, 1 passed in 184.34s (0:03:04)
```

After correcting the owning boundaries and preserving established mapping
access for immutable reconstruction records, the focused set reported:

```text
18 passed in 183.73s (0:03:03)
```

The complete five owning files reported:

```text
375 passed in 3379.70s (0:56:19)
```

Two adjacent direct-value/runner compatibility nodes reported
`2 passed in 2.06s`. Ruff formatting and lint, byte compilation, and
`git diff --check` all exited zero before the exact final suite.

The exact sanitized fifteen-file Task 6 suite reported:

```text
860 passed in 6211.36s (1:43:31)
```

No production or test file changed during that execution. Only this tracked
ledger evidence and the matching correction report were updated afterward. No
selected method, population, seed denominator, revision allowance, estimand,
concurrency behavior, or claim permission changed. No real scientific
workload ran, and no new provenance, content-validation,
filesystem-hardening, or cyber-related mechanism was added.

### Task 6 v3 final-acceptance correction

A further independent review of Task 6 at
`df6c3b16820c28ba2397590be08f7251997ea4f5` found that F-079 had not been
closed at every reachable publication boundary.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-081 | Important | Additional scaling, final, trajectory, downstream, final-analysis, null-DE, and publication-synthesis authorities accepted Boolean aliases for integer schema, count, or ordinal fields. Public final and trajectory evaluation also trusted empty, sliced, or coherently expanded typed plan populations. | Closed test-first with exact built-in integer checks at each owner, nonempty and complete Cartesian typed-plan validation at the public evaluation entries, and fixed production totals of 1,760 primary runs and 44 trajectory runs. Independently rebuilt internal typed plans use the existing structural terminal-record replay after their owning authority validation. |

The direct payload, loader, and population RED set reported:

```text
13 failed in 186.27s (0:03:06)
```

Adapter and source-owner mutation probes demonstrated seven additional Boolean
acceptance gaps while retaining nineteen already-rejecting controls. After
correction, the focused direct, adapter, and source-owner sets reported:

```text
13 passed in 166.61s (0:02:46)
11 passed, 19 deselected in 2.15s
8 passed in 2.86s
```

Complete changed and adjacent owner suites reported:

```text
tests/test_downstream_evidence.py: 170 passed in 93.77s
tests/test_final_analysis.py: 53 passed in 382.85s (0:06:22)
tests/test_final_runner.py: 131 passed in 782.17s (0:13:02)
tests/test_scaling_panel.py: 66 passed in 2940.99s (0:49:00)
publication-synthesis and final-null-DE: 72 passed in 80.91s (0:01:20)
freeze and revision compatibility: 139 passed in 2314.41s (0:38:34)
```

The first exact fifteen-file run exposed a test/consumer integration mistake:
the strict public production-population gate had also been used by internal
replay paths after those paths had independently rebuilt and validated a typed
plan. It reported:

```text
57 failed, 844 passed in 7652.31s (2:07:32)
```

The public gate remained strict. Only independently validated internal replays
were returned to the existing structural terminal-record validator. All 57
previous failures then passed, and the complete affected owners reported:

```text
53 passed, 376 deselected in 159.04s (0:02:39)
363 passed in 1246.67s (0:20:46)
```

Ruff formatting and lint, byte compilation, and `git diff --check` all exited
zero before the second clean exact run. The exact fifteen-file Task 6 suite
then reported:

```text
901 passed in 7766.96s (2:09:26)
```

No production or test file changed during that successful exact run. The
selected methods, final dataset population, 1,760/44 run totals, seed
denominator, revision allowance, estimands, execution behavior, and claim
permissions remain unchanged. No real scientific or comparator workload ran,
and no unrelated mechanism was added.

### Task 6 v4 final-acceptance correction

A subsequent independent review of Task 6 at `0fa2719` demonstrated four
remaining important validation gaps.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-082 | Important | Public final and trajectory evaluation accepted structurally coherent full-size forged populations and derived plan, entry, run, configuration, and nested authority types. | Closed test-first with exact typed authorities and independent reconstruction of the fixed 20-method order, established method seed policy, 4 x 5 x 2 primary scientific coordinates or registered trajectory identity, canonical run construction, action/reason bindings, and planned preflight fields. The fixed 1,760/44 populations are unchanged. |
| F-083 | Important | `direct_equal` converted a `FrozenDirectObject` to a mapping before validating it, so duplicate keys, noncanonical order, malformed pairs, and non-string keys could be silently normalized or leak an incidental conversion error on either operand. | Closed test-first by validating every frozen-object operand before equality. Valid nested mapping/list comparisons are unchanged. |
| F-084 | Important | Final score evidence accepted Boolean aliases for a zero metric denominator and reliability-bin ordinal one. | Closed test-first by requiring exact built-in integers for every score metric denominator and reliability-bin ordinal. |
| F-085 | Important | Huge positive and negative Python integers leaked `OverflowError` from `GateResult`, both selection finite-number helpers, final-analysis numeric evidence, and publication synthesis. | Closed test-first by translating unrepresentable real coercions to each boundary's documented domain exception while preserving valid finite numerics. |

The focused RED invocations reported:

```text
20 failed, 1 passed in 3.38s
12 failed, 2 passed in 30.61s
```

They covered complete final and trajectory dataset-block substitutions, five
scientific/preflight mutations, five derived authority layers, both
frozen-object operand positions across four malformed structures, exact score
integer aliases, and both signs of unrepresentable Python integers. The three
passing controls established the canonical full populations and valid direct
comparison behavior before production changes.

After the bounded corrections, the formatted combined focused set reported:

```text
38 passed in 44.81s
```

The complete five changed and adjacent owning files then reported:

```text
354 passed in 1569.69s (0:26:09)
```

The population reconstruction was subsequently narrowed to the explicitly
permitted non-seal authorities. The exact affected focused set and complete
final-runner owner were repeated from that final production state:

```text
17 passed in 45.17s
154 passed in 753.58s (0:12:33)
```

The correction added no new plan seal, content-summary, filesystem-hardening,
or runtime-lock mechanism. It did not change selected methods, dataset
populations, seeds, revision allowances, estimands, execution behavior, or
claim permissions. No real scientific or comparator workload ran.

Fresh Ruff formatting and lint, scoped byte compilation, and
`git diff --check` all exited zero from the final production/test state. The
exact sanitized fifteen-file Task 6 suite then reported:

```text
936 passed in 6444.11s (1:47:24)
```

No production, test, or documentation file changed during that authoritative
run. Only this terminal evidence and the matching correction report were
recorded afterward.

### Task 6 v5 independent-review correction

The next independent review of Task 6 at `a3b1374` found four remaining
important validation gaps. The preceding v4 completion statement applied only
to F-082 through F-085 and did not constitute final Task 6 acceptance.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-086 | Important | Final evaluation accepted completed runs whose metric rows contained only the canonical metric names, without the established identity, status, value, denominator, and reason evidence. | Closed test-first by requiring the complete legacy or direct metric field set, exact plan-bound identity, exact integer denominators, finite float values, and consistent completed or reason-coded unavailable rows. |
| F-087 | Important | Public scaling execution accepted a coherent three-size, 15-entry plan because its only entry validation used the deliberately generic lower-level plan serializer. | Closed test-first at the public executor with the fixed four-size by five-method population, exact typed authorities, canonical size-method order, sequential ordinals, established seed policy, and configuration bindings. Generic lower-level fixture compatibility is unchanged. |
| F-088 | Important | A checkpoint labeled completed could serialize with a positive planned count but no complete record population. | Closed test-first by requiring every completed checkpoint to contain exactly its planned record count. Running-prefix behavior and valid synthetic replay fixtures with complete records are unchanged. |
| F-089 | Important | `direct_equal` accepted derived `FrozenDirectList` wrappers even though the direct freeze and encoding boundaries reject them. | Closed test-first by applying the shared exact frozen-list wrapper validation to both equality operands while preserving canonical frozen-list/list comparison behavior. |

The focused RED evidence showed all six malformed completed-metric variants
were accepted, all three incomplete completed checkpoints serialized, all
four malformed public scaling populations reached execution-stage
materialization, and both derived frozen-list operand positions compared
without rejection. After the bounded owning corrections, the consolidated
metric test and the nine scaling, checkpoint, and direct-list cases were
green.

The complete changed-owner suites reported:

```text
269 passed in 3362.21s (0:56:02)
```

The first exact fifteen-file run exposed an integration overconstraint in the
generic checkpoint serializer:

```text
3 failed, 943 passed in 6500.93s (1:48:20)
```

All three failures used established downstream replay fixtures whose completed
synthetic checkpoint had its full planned record count but no dataset receipts.
Production dataset completeness is owned by the fixed public executor and real
result store, while the generic serializer historically supports these
complete-record synthetic fixtures. A focused compatibility regression first
failed in `2.30s`. The serializer was then narrowed to preserve empty dataset
fixtures while retaining the exact completed record-count invariant. The three
failed downstream nodes passed in `2.07s`, the zero- and partial-record
completed mutations remained rejected, and the complete affected owners
reported:

```text
243 passed in 2173.21s (0:36:13)
```

Fresh Ruff formatting and lint, scoped byte compilation, and
`git diff --check` all exited zero from the final production/test state. The
exact sanitized fifteen-file Task 6 acceptance suite was then repeated and
reported:

```text
946 passed in 6472.26s (1:47:52)
```

No production, test, or documentation file changed during that authoritative
run. Only this terminal evidence and the matching correction report were
recorded afterward.

These corrections do not alter selected methods, the fixed scaling sizes or
methods, final datasets, seed policies, revision allowances, estimands,
execution behavior, or claim permissions. No scientific workload ran.

### Task 6 v6 independent-review correction

The next independent review found four remaining important validation gaps.
The preceding v5 completion statement applied only to F-086 through F-089 and
was not final Task 6 acceptance.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-090 | Important | A coherent change to the observed control's configuration and all forty final rows could replace execution with a technical non-run. | Closed test-first by independently deriving each canonical method disposition and reason at the public final-population gate. The established 1,480 executable and 280 non-run rows are unchanged. |
| F-091 | Important | Executable failures and planned non-runs accepted five-field metric rows without complete method, dataset, seed, or configuration bindings, and accepted Boolean false as denominator zero. | Closed test-first by applying the established full row schema and plan bindings to every terminal state, requiring canonical metric order and exact built-in integer denominators. |
| F-092 | Important | The fixed public scaling gate trusted mutable typed values for its model seed, run labels, output scale, resource ceilings, measurement provenance, and schedule policy. | Closed test-first by sharing the existing fixed entry builder between planning and public validation, rebuilding tracked contract and registry values, and requiring exact permitted contract and entry fields. Generic serialization and synthetic checkpoints are unchanged. |
| F-093 | Important | Five selection-owned source schema versions admitted Boolean true as integer one. | Closed test-first with exact built-in integer checks for the development panel, selection contract, calibration contract, ablation registry, and development-search ledger. Evaluation-manifest behavior is unchanged. |

The focused RED run reported:

```text
26 failed in 115.63s (0:01:55)
```

After the bounded corrections, the formatted focused set reported:

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
Post-suite static gates also passed.

These corrections do not alter selected methods, fixed final or scaling
populations, seed policies, revision allowances, estimands, execution
behavior, or claim permissions. No scientific workload ran.

## Task 7: CLI, study-document, and migration audit

The audit covered every active Python, shell, and native simulator entry point;
all tracked study JSON and method-attempt documents; the development operator
workflow; the 6,034-path deletion surface; the preserved historical boundary;
and repository submodule state. The exact four-file baseline completed before
any tracked change:

```text
193 passed, 1 skipped in 989.46s (0:16:29)
```

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-094 | Important | `scripts/finalize_development_authority.py --help` ignored the help request and entered development-artifact validation; expected authority failures also escaped as a traceback rather than the CLI's structured failure result. | Independently accepted unchanged. The parser consumes help before the action and translates the bounded expected error to status 2 without adding scientific or path options. |
| F-095 | Important | `scripts/simulators/run_sergio.py --help` treated help as invalid positional arity and exited 1 on the required CLI discovery boundary. | Independently accepted unchanged. Help remains side-effect-free and the three-positional execution contract is unchanged. |
| F-096 | Important | `scripts/build_saver_r_environment.sh` refused to replace an existing library but could overwrite an existing caller-named build receipt. | The original overwrite defect remains closed. Independent review found adjacent destination-relationship and validation-order gaps, tracked separately as F-098 and F-099. |
| F-097 | Important | `docs/development-selection-workflow.md` omitted the score-preparation and authority-finalization prerequisites, the downstream step required before base promotion, the build/downstream/promotion/selection sequence for both conditional revisions, the external-reference prerequisite, and post-final analysis order. | The corrected command content and ordering were independently accepted. Independent review found a CommonMark nesting defect in the conditional revision block, tracked separately as F-100. |
| F-098 | Important | The SAVER environment builder accepted equal or ancestor-related normalized library and receipt destinations. Parent creation could turn the library destination into a container, or the final receipt write could follow a successful library move into the same path, producing a false success or a partial non-retryable outcome. | Closed test-first by rejecting equality and either ancestor relationship before source resolution, parent creation, or build work. The canonical sibling receipt remains valid, and every rejected case leaves both destinations absent. |
| F-099 | Minor | Source normalization preceded the existing-receipt refusal, so an invalid source parent could mask the documented non-overwrite status. | Closed test-first by normalizing and checking the receipt before source normalization. An existing receipt now always returns status 73 at this boundary and remains byte-for-byte unchanged. |
| F-100 | Minor | The first conditional revision fence used a three-space continuation under ordered item 13, so CommonMark rendered the v28 and v29 material outside the intended list item. | Closed with four-space item continuation indentation and a structural regression that verifies both fenced blocks remain children of ordered step 13 in v28-before-v29 order. |

The focused RED invocation reported:

```text
4 failed, 27 passed in 27.51s
```

After the minimal corrections, the same focused set reported:

```text
31 passed in 26.04s
```

The complete selection, SERGIO-adapter, and repository-hygiene owner set then
reported:

```text
197 passed, 1 skipped in 304.50s (0:05:04)
```

Study registry, protocol, workflow-order, and generic study-CLI compatibility
checks reported:

```text
73 passed, 1 skipped in 3.40s
```

Every active Python CLI subsequently returned successful help without
scientific execution. Shell and R scripts parsed, all 17 study and
method-attempt JSON documents parsed, `.gitmodules` remained absent, and no
tracked gitlink or active submodule remained. The historical tree retains 1,056
tracked files behind its explicit non-runtime boundary. Active matches from the
deleted-path reference scan were either current method names, review/design
evidence, or repository-hygiene ignore sentinels; no active import or runtime
consumer targets a deleted root.

Ruff formatting and lint, scoped Python compilation, shell syntax, R parsing,
and `git diff --check` passed before the final exact Task 7 suite. No real
scientific workload ran, no archived material was rewritten or restored, and
no selected method, configuration, dataset population, seed policy, estimand,
revision allowance, or claim permission changed.

The exact four-file Task 7 suite then reported:

```text
224 passed, 1 skipped in 998.65s (0:16:38)
```

Production, tests, and operator documentation remained frozen throughout that
run. Only this terminal evidence and the matching Task 7 report were updated
afterward.

### Task 7 independent-review correction

The first independent review accepted the F-094 and F-095 corrections and the
scientific command content added for F-097. It found one important
environment-builder destination defect and two minor validation/documentation
defects, recorded as F-098 through F-100 above.

Focused regressions first reported:

```text
4 failed, 1 passed in 0.11s
```

The failures covered the existing-receipt ordering, equal destinations, a
normalized receipt nested below the library, and the malformed ordered-list
structure. The canonical sibling receipt control passed. A separate converse
ancestor regression then reported:

```text
1 failed in 0.07s
```

After the bounded corrections and formatting, the consolidated focused set
reported:

```text
7 passed in 0.10s
```

The complete repository-hygiene and comparator-workflow owners reported:

```text
168 passed in 234.25s (0:03:54)
```

Active CLI help, shell and R parsing, study JSON parsing, archive and submodule
boundaries, Ruff formatting and lint, scoped Python compilation, and
`git diff --check` passed before the exact suite. The exact four-file Task 7
suite then reported:

```text
230 passed, 1 skipped in 1020.79s (0:17:00)
```

Production, tests, and operator documentation remained frozen throughout that
run. The same static and structural gates passed again after this terminal
evidence was recorded.

The corrections do not change scientific dependencies, selected methods,
configurations, dataset populations, seeds, estimands, revision allowances,
archive contents, or claim permissions. No scientific workload ran.

### Task 7 second independent-review correction

The final Task 7 review retained two nonblocking minor findings. Both were real
reproducibility and state-isolation defects and are now closed without changing
the accepted F-094 through F-100 dispositions.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-101 | Minor | Importing three active command modules unconditionally set `sys.dont_write_bytecode` to true and changed later in-process behavior. | Closed test-first. Module imports now preserve either initial state. Direct entry points still enable the setting before production imports, while every public `main(argv)` and direct entry-point exit restores the caller's prior state in `finally`. |
| F-102 | Minor | The literal Task 7 shell syntax command returned status 127 when its glob classes were empty and did not parse every expanded path independently. | Closed test-first. The plan now discovers null-delimited regular shell files and parses each separately, succeeds with no discovered scripts, and propagates a malformed optional-script failure. |

The focused RED run reported:

```text
7 failed, 3 passed, 43 deselected in 0.73s
```

After the bounded corrections, focused GREEN and the post-format rerun
reported:

```text
10 passed, 43 deselected in 0.84s
10 passed, 43 deselected in 0.80s
```

The complete affected owners reported:

```text
293 passed, 1 skipped in 2124.57s (0:35:24)
```

All 29 active Python help paths, the active shell script, both R drivers, all
17 study JSON documents, the 1,056-file archive boundary, migration and
submodule checks, Ruff formatting and lint over 164 Python files, scoped
compilation, and `git diff --check` passed before the exact suite.

The exact four-file Task 7 suite then reported:

```text
240 passed, 1 skipped in 1002.41s (0:16:42)
```

Production, tests, and the Task 7 plan remained frozen throughout both complete
runs. These corrections do not change scientific dependencies, selected
methods, configurations, dataset populations, seeds, estimands, revision
allowances, archive contents, or claim permissions. No real scientific
workload ran. Independent acceptance is not claimed.

## Task 8: Genome Biology package review

The Genome Biology general and Methodology instructions were rechecked against
the official pages verified on 23 July 2026. The package is a compilable,
fail-closed Methodology draft, not a submission-ready article and not evidence
of empirical competitiveness.

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-103 | Important | The rendered title page silently omitted all required author, affiliation, email, and corresponding-author metadata instead of making the blocking omission visible. | Closed with explicit red author-input blockers in the title-page commands. No person or institution was invented. |
| F-104 | Important | The abstract and external-reference Results prose described unexecuted analyses or evidence as complete. | Closed with a 62-word placeholder-bearing abstract that makes no empirical advance claim and future-tense external-reference evidence. |
| F-105 | Important | The compact checklist marked the state-of-the-art advancement exercise and final abstract constraint complete despite unavailable results and final wording. | Closed by separating checked structural facts from unchecked clear-advance, same-dataset, known-truth, real-data, and final-abstract evidence requirements. |
| F-106 | Minor | The manuscript lacked the project-required Abbreviations section and did not expand several terms at first use. | Partially closed by the initial correction: the section and identified prose expansions were added, but the first rendered `IQR` remained unexpanded and absent from the section. The residual defect is tracked as F-111. |
| F-107 | Minor | Both submission checklists retained excluded template-integrity validation gates, and the paper README referenced the compact instance. | Closed as explicitly required. The active gates and stale README reference were removed while ordinary template provenance, upstream notices, licensing separation, and publisher-redistribution caveats remain. No replacement integrity mechanism was added. |
| F-108 | Minor | Static DOI-issuing deposition was phrased as mandatory although the current Methodology instructions recommend it. | Partially closed by the initial correction: the recommendation and author decision were stated, but unconditional archive-identifier inputs remained in the declaration and checklists. The residual defect is tracked as F-110. |
| F-109 | Minor | One author name in the DCA BibTeX entry used a malformed TeX accent escape. | Closed. All twelve cited DOI, title, year, and venue records matched registry metadata and the final bibliography resolved. |
| F-110 | Minor | The manuscript declaration, compact author-input checklist, and full availability checklist still required a static software archive or identifier regardless of the authors' deposition decision. | Closed test-first. Public source and data access and the OSI-license requirement remain unconditional. Static deposition is conditional on the author decision; any created archive must have a persistent identifier and citation and identify the manuscript release. |
| F-111 | Minor | The first rendered use of `IQR` remained unexpanded in a pending Results marker, while the Abbreviations section omitted it and the compact checklist claimed compliance. | Closed test-first by expanding interquartile range (IQR) at first rendered use and adding its canonical expansion to the Abbreviations section. |

Focused venue-contract RED reported:

```text
4 failed, 53 deselected in 0.11s
```

Focused GREEN reported:

```text
4 passed, 53 deselected in 0.06s
```

The new checks plus the existing manuscript contracts then reported:

```text
7 passed, 175 deselected in 2.01s
```

Follow-up F-110/F-111 focused RED reported:

```text
2 failed in 0.09s
```

The same focused tests then reported:

```text
2 passed in 0.03s
```

The complete repository-hygiene owner reported:

```text
59 passed in 26.11s
```

Existing manuscript/comparator documentation checks reported:

```text
3 passed in 2.16s
```

The obsolete-term checks reported:

```text
2 passed in 0.80s
```

A clean `pdflatex`, `bibtex`, `pdflatex`, `pdflatex` build exited zero. The
final PDF had 14 A4 pages; build-log inspection found no unresolved citation,
unresolved reference, fatal error, emergency stop, or overfull box. Extracted
text showed the visible title-page blockers, 62-word unstructured abstract, six
keywords, required main order, the first-use interquartile range (IQR)
expansion, the canonical IQR abbreviation entry, the conditional static-archive
input, all seven Declarations, and References.

Ruff formatting and lint passed across all 164 Python files, scoped Python
compilation and `git diff --check` exited zero, and generated manuscript and
cache products were removed after inspection.

Submission remains blocked on real frozen evidence, same-dataset
state-of-the-art advance, known-truth and real-data utility, author metadata and
approval, all declarations, the author-approved LLM disclosure, public data and
source access, an OSI-compliant project license, accessions, the static-archive
decision and any resulting identifier, figures/tables/supplement, Minimum
Standards, cover letter, preprint and redistribution decisions, and final
authorization. No scientific result, human metadata, license, identifier, or
authorization was fabricated. No real scientific workload ran, and no
estimand, population, selection rule, method configuration, seed policy, or
claim permission changed. Independent acceptance is not claimed.

## Task 9: Cross-cutting and test-order review

The exact 52-file combined focused suite from Tasks 2 through 8 ran in one
pytest process, in the prescribed order, with default pytest flags and the
supported sanitized interpreter:

```text
3125 passed, 52 skipped in 9436.35s (2:37:16)
```

The command exited zero. Production, tests, and documentation remained frozen
throughout the run. No unexpected failure, collection error, warning summary,
or test-order defect occurred, so no failure minimization, new RED regression,
production patch, owner-adjacency rerun, or corrective combined-suite rerun was
required.

The finding ledger contains F-001 through F-111 exactly once, without a gap or
duplicate. All are fixed or terminate in a later fixed superseding finding:
the F-006/F-007/F-010 chain terminates in F-011; the numerical refinement
chains terminate by F-065; F-094/F-095 corrected behavior was independently
accepted unchanged; F-096/F-097 terminate through F-098/F-099/F-100; F-106
terminates in F-111; and F-108 terminates in F-110. No F-series finding is
assigned `not reproducible with evidence`, `human/scientific blocker`, or
`minor excluded by standing scope`, and no Critical or Important finding is
open.

The separate human and scientific submission blockers remain explicit and
unchanged. O-002 through O-004 remain evidence-backed environmental
observations; O-001 remains the existing Minor release-environment recheck
assigned to Task 10. No real scientific workload ran, no scientific or human
content was fabricated, and no estimand, population, selection rule,
configuration, seed policy, provenance, legacy compatibility, or claim
permission changed. The standing exclusions were honored. Independent
acceptance is not claimed.

A focused post-suite state-isolation gate passed 13 tests covering CLI
bytecode-state restoration, final/development environment isolation including
libc-only loader state, and caller RNG restoration. Ruff 0.14.4 reported all
164 files formatted and all checks passing. Scoped Python compilation and
`git diff --check` exited zero. The supported pytest interpreter's attempted
Ruff module invocation stopped before analysis because that environment does
not contain Ruff; rerunning the same checks with the already installed Ruff
0.14.4 executable produced the recorded clean results. This was tool placement,
not a repository finding.

## Task 10: Independent whole-range review correction

The independent review of the 40 commits from the fixed review baseline
through the Task 9 candidate accepted the range with zero Critical and zero
Important findings. It retained one Minor exception-normalization defect:

| ID | Severity | Summary | Disposition |
|---|---|---|---|
| F-112 | Minor | Two finite extended-precision inputs outside the float64 range were rejected with a NumPy-state-dependent `RuntimeWarning` or `FloatingPointError` instead of the documented `ValueError`. | Closed test-first. The two float64 conversion boundaries now locally suppress conversion overflow signalling, then reject the resulting nonfinite values through their existing `ValueError` contracts. Both global `np.seterr(over="raise")` and warning-as-error states are covered. |

The focused RED run reported:

```text
2 failed in 1.74s
```

Both failures exposed `FloatingPointError: overflow encountered in cast` at the
reviewed conversion boundaries. The same two tests passed after the bounded
correction:

```text
2 passed in 1.58s
```

After formatting, the focused rerun reported:

```text
2 passed in 1.72s
```

The complete v27 and v28 owner modules reported:

```text
116 passed, 2 skipped in 3.20s
```

Focused Ruff lint and formatting checks and `git diff --check` also passed.
The correction does not change any accepted scientific dependency,
configuration, population, estimand, selection rule, seed policy, or claim
permission. No scientific workload ran.
