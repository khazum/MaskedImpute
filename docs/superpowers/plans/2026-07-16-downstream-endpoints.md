# Evaluator-Only Downstream Endpoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, complete, reason-coded evaluator-only layer for eight downstream and molecular endpoints.

**Architecture:** A new `downstream_evaluation.py` module owns immutable method-output and evaluator-target types, simulator truth extraction, endpoint algorithms, and the complete record assembler. It does not modify or call either benchmark runner, preserving the truth-isolation boundary.

**Tech Stack:** Python 3.10+, NumPy, SciPy, AnnData-compatible dataset objects, pytest.

## Global Constraints

- Return exactly the eight endpoint rows defined in the approved design.
- Use seed `20260716` for deterministic clustering.
- Correct one global BH family across all group-by-gene hypotheses.
- Record exactly one independent biological unit; descriptive cell/gene counts are never replicates.
- Never derive pseudotime from group labels.
- Do not modify `runner.py` or `final_runner.py`.

---

### Task 1: Typed evaluator boundary and simulator truth adapter

**Files:**
- Create: `maskimpute_benchmark/downstream_evaluation.py`
- Create: `tests/test_downstream_evaluation.py`

**Interfaces:**
- Produces: `MethodOutput`, `EvaluatorTargets`, `TrajectoryTruth`, `EndpointRecord`, and `evaluator_targets_from_dataset(dataset)`.

- [ ] Write failing tests for the truth-free `MethodOutput` shape, exact simulator marker mappings, held-out extraction, absent pseudotime reason, and trajectory-root validation.
- [ ] Run the focused tests and verify failure because the module is absent.
- [ ] Implement only the immutable types, validation, alignment, and adapter behavior required by those tests.
- [ ] Run the focused tests and verify they pass.

### Task 2: Marker rank and positive-control DE endpoints

**Files:**
- Modify: `maskimpute_benchmark/downstream_evaluation.py`
- Modify: `tests/test_downstream_evaluation.py`

**Interfaces:**
- Produces: group-score ranking, global-family Benjamini-Hochberg adjustment, marker-rank loss, marker recall, and false-discovery rate records.

- [ ] Add hand-calculated and permutation-invariance tests, including an assertion that the BH family size is exactly groups times genes.
- [ ] Run each new test and verify its expected feature-missing failure.
- [x] Consume the persisted `log2(CP10k + 1)` evaluator scale without a second normalization; implement group contrasts, tie-aware ranks, one-sided Welch p-values, and global BH adjustment.
- [ ] Run the focused tests and verify green.

### Task 3: Deterministic clustering endpoints

**Files:**
- Modify: `maskimpute_benchmark/downstream_evaluation.py`
- Modify: `tests/test_downstream_evaluation.py`

**Interfaces:**
- Produces: deterministic PCA/k-means labels and exact ARI/NMI losses without an additional dependency.

- [ ] Add deterministic, label-permutation, row/column permutation, and degenerate-group tests.
- [ ] Verify the tests fail because clustering endpoints are not implemented.
- [ ] Implement stable-ID ordering, full-SVD PCA, seeded k-means, ARI, and NMI.
- [ ] Run the focused tests and verify green.

### Task 4: Held-out profile endpoints

**Files:**
- Modify: `maskimpute_benchmark/downstream_evaluation.py`
- Modify: `tests/test_downstream_evaluation.py`

**Interfaces:**
- Produces: independent held-out gene-profile and cell-profile rank-concordance losses.

- [ ] Add hand-calculated, permutation, method-collapse, missing-heldout, and constant-heldout tests.
- [ ] Verify the new tests fail for the intended missing behavior.
- [ ] Implement tie-aware Spearman aggregation on heldout-variable profiles, assigning rho=0 to method-collapsed profiles.
- [ ] Run the focused tests and verify green.

### Task 5: Root-oriented trajectory and complete assembler

**Files:**
- Modify: `maskimpute_benchmark/downstream_evaluation.py`
- Modify: `tests/test_downstream_evaluation.py`

**Interfaces:**
- Produces: multiscale diffusion pseudotime loss and `evaluate_downstream_endpoints(method_output, evaluator_targets)`.

- [ ] Add genuine linear-trajectory, deterministic, permutation, missing-truth, invalid-root, graph-degenerate, schema-completeness, reason-code, and independent-unit tests.
- [ ] Verify each new behavior fails before implementation.
- [ ] Implement deterministic diffusion distance and the fixed-order eight-row assembler.
- [ ] Run focused tests, relevant existing suites, lint, byte compilation, and `git diff --check`.
- [ ] Commit the verified implementation.

### Independent-review remediation

- [x] Replace label-derived cluster count with a truth-free fixed 2--10 grid
  and deterministic minimum Davies--Bouldin selection.
- [x] Cache one full-SVD representation per denominator for clustering and
  trajectory.
- [x] Replace the dense trajectory distance matrix with blockwise exact sparse
  kNN construction and test the 2,700-cell bound.
- [x] Register and hash-bind a separate exact-latent synthetic trajectory panel.
- [x] Convert expected numerical failures to the complete eight-row terminal
  schema.
- [x] Add a hash-bound, resumable development/final evidence stage over the
  persisted evaluator-output contracts from `f6d19e8`.
- [x] Add schema-4 selection completeness so a bound eight-row record is
  required for every reconstruction-selection denominator.
