# Frozen Trajectory and Final Null-DE Implementation Plan

> **For agentic workers:** Use test-driven development and systematic debugging.
> Do not run an actual frozen round while implementing this plan.

**Goal:** Produce receipt-bound supplementary trajectory execution and
deterministically replayable frozen-final null-DE evidence without changing the
primary development or final reconstruction denominators.

**Design:**
[`2026-07-17-final-safety-trajectory-design.md`](../specs/2026-07-17-final-safety-trajectory-design.md)

## Task 1: Frozen supplementary trajectory plan

**Files:** `maskimpute_benchmark/final_runner.py`,
`maskimpute_benchmark/trajectory_dataset.py`, `tests/test_final_runner.py`,
`tests/test_trajectory_dataset.py`

- [ ] Add RED tests proving the plan is derived only from the registered
  trajectory authority and frozen method/configuration/applicability receipt.
- [ ] Add exact truth-free `PreparedDataset` construction and bind its semantic
  dataset, retained IDs, method-input hash, and authority hashes.
- [ ] Build the complete frozen method-by-seed trajectory denominator with an
  explicit supplementary scope and no tuning/CLI overrides.
- [ ] Extend retained-all-development score-policy validation only for the exact
  registered trajectory identity and test rejection of nearby identities.
- [ ] Run focused tests with warnings as errors.

## Task 2: Receipt-bound trajectory execution

**Files:** `maskimpute_benchmark/final_runner.py`,
`maskimpute_benchmark/runner.py`, `tests/test_final_runner.py`

- [ ] Add RED tests for separate primary/trajectory stores, complete terminal
  rows, immutable resume, and final-receipt coverage of trajectory bytes.
- [ ] Reuse the frozen executor and final result contracts under
  `results/trajectory/execution/`; do not duplicate adapter algorithms.
- [ ] Include combined primary/supplementary storage preflight before executing
  the first primary method.
- [ ] Validate the trajectory denominator and manifest before issuing the sole
  final evaluation receipt.
- [ ] Test crash recovery before/after record and manifest publication.
- [ ] Run focused final-runner, score-evidence, and trajectory tests with
  warnings as errors.

## Task 3: Evaluator-only trajectory evidence

**Files:** `maskimpute_benchmark/downstream_evidence.py`,
`scripts/run_final_downstream_evidence.py`,
`tests/test_downstream_evidence.py`

- [ ] Add RED tests that accept only the receipt-bound supplementary trajectory
  manifest and exact registered evaluator dataset.
- [ ] Add a production trajectory source scope without weakening the primary
  final or explicit test-only source contracts.
- [ ] Decode and revalidate outputs, replay the trajectory endpoint, preserve all
  terminal methods, and publish under the receipt-hash external namespace.
- [ ] On load, replay every endpoint and reject coordinated row/hash changes.
- [ ] Run focused downstream tests with warnings as errors.

## Task 4: Frozen-final null-DE evidence

**Files:** create `maskimpute_benchmark/final_null_de.py`, create
`scripts/run_final_null_de.py`, create `tests/test_final_null_de.py`

- [ ] Add RED hand-calculated tests for receipt-derived balanced splits, fixed
  observed-count masks, exact per-run rows, terminal reasons, and permutation
  invariance.
- [ ] Bind the exact evaluated primary final source through the validated final
  downstream/source-plan authority; do not accept free-form records.
- [ ] Reuse `balanced_null_split`, `fixed_null_de_gene_mask`, and
  `evaluate_null_de_fpr` rather than introducing a second statistical test.
- [ ] Implement an immutable resumable plan/record/manifest store outside the
  repository, namespaced by round ID and receipt payload hash.
- [ ] Recompute source bindings, splits, masks, and FPRs on resume and completed
  load.
- [ ] Add exact denominator and symlink/alias rejection tests.
- [ ] Run focused tests with warnings as errors.

## Task 5: Publication synthesis gate

**Files:** publication-synthesis module and tests selected by the asset stage

- [ ] Add RED tests for null-DE pass, fail, and incomplete gates and for the
  trajectory result always remaining descriptive.
- [ ] Require exact primary final, downstream, null-DE, trajectory, and scaling
  evidence bindings before rendering claims.
- [ ] Prohibit `competitive` when any prespecified safety gate is failed or
  unavailable.

## Task 6: Verification and review

- [ ] Run all focused suites with `-W error`, Ruff, byte compilation, and
  `git diff --check`.
- [ ] Independently review output authority, denominator completeness, truth
  isolation, statistics, crash recovery, and claim language.
- [ ] Resolve every Critical or Important finding and rerun affected tests.
- [ ] Commit all Python changes before regenerating development evidence; the
  implementation-source hash must remain fixed thereafter.
