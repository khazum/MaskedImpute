# Receipt-Bound Trajectory Downstream Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-only supplementary trajectory downstream plan that independently replays the receipt-bound registered trajectory denominator and emits exactly one trajectory endpoint per terminal source run.

**Architecture:** Extend the immutable evaluated-round binding with the exact trajectory evidence graph already sealed in the final receipt. Rebuild the trajectory dataset, frozen configurations, execution plan, terminal records, and result inventory through read-only final-runner validation APIs; then route the validated source through an explicit `supplementary_trajectory` scope with a one-endpoint contract and a receipt-derived external namespace. The primary eight-endpoint plan and final null-DE source remain unchanged.

**Tech Stack:** Python 3.11, dataclasses, canonical JSON/SHA-256 receipts, AnnData, NumPy, pytest, Ruff.

## Global Constraints

- Work only in `.worktrees/downstream-trajectory-replay` on `codex/downstream-trajectory-replay`.
- Do not modify `maskimpute_benchmark/final_runner.py` or `maskimpute_benchmark/publication_synthesis.py`.
- The supplementary plan is available only through `build_final_trajectory_downstream_evidence_plan(repository, round_directory)` and accepts no method, endpoint, dataset, or scope override.
- Downstream code may call only read-only final-runner trajectory loaders/rederivation; it must never call a trajectory materializer or publisher.
- Development and primary final plans retain their existing eight-endpoint denominator.
- The trajectory plan emits exactly `trajectory_pseudotime_rank_loss`; every upstream terminal state remains as one reason-coded row.
- Final downstream outputs must be outside the repository and exactly namespaced by evaluated round plus receipt payload hash; trajectory is the primary namespace's `trajectory/` child.

---

### Task 1: Exact Evaluated-Round Trajectory Binding

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: the evaluated receipt's `trajectory_evidence`, result inventory, primary plan hash, and read-only final-runner replay API.
- Produces: new immutable `EvaluatedRoundBinding` trajectory evidence/dataset/authority/plan/manifest/validation/inventory/count fields.

- [x] Add production-shaped receipt tests that require both scaling and trajectory evidence and reject a coordinated rehash/replacement of a trajectory plan, dataset, authority, record, manifest, or result inventory.
- [x] Run the focused tests and confirm they fail because the result-manifest schema and binding omit trajectory.
- [x] Extend the exact result-manifest schema and binding parser; invoke read-only trajectory replay and compare its complete return value with the embedded evidence before deriving binding hashes/counts.
- [x] Re-read the receipt after replay and reject any byte change during validation.
- [x] Run the focused receipt tests to green.

### Task 2: Fixed Supplementary Plan and Source Replay

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: `EvaluatedRoundBinding`, `load_prepared_trajectory_dataset`, frozen method/registry, the receipt-bound claim/environment/trajectory authority, and `TrajectoryExecutionPlan`.
- Produces: `build_final_trajectory_downstream_evidence_plan(repository, round_directory) -> DownstreamEvidencePlan` with one registered dataset and a strict trajectory source-manifest reader.

- [x] Add failing tests for the fixed builder, exact trajectory manifest schema, authority/dataset mismatch, incomplete terminal denominator, and a persisted plan whose `source_plan_authority="independent"` is locally forged.
- [x] Add `supplementary_trajectory` as the only non-primary final scope and require its source root to be `results/trajectory/execution`.
- [x] Rebuild the trajectory dataset and execution plan from receipt-bound frozen inputs and pass the typed plan as source authority; validate each terminal source record against that complete plan.
- [x] Route persisted final plans through the fixed primary or fixed trajectory production builder according to their exact scope; never reconstruct a final plan from caller-provided datasets or an authority label.
- [x] Run plan, source, persistence, and primary-regression tests to green.

### Task 3: Closed One-Endpoint Evaluation Contract

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: `evaluate_downstream_endpoints` for primary/development and the trajectory evaluator result for supplementary trajectory.
- Produces: scope-aware endpoint names, record validation, running counts, completed-manifest counts, resume validation, and reload re-evaluation.

- [x] Add failing tests covering completed, unavailable, failed, timeout, resource-exceeded, infrastructure-error, blocked-authority, and budget-exhausted trajectory runs.
- [x] Define a closed endpoint-name helper: eight existing endpoints for primary/development and exactly `trajectory_pseudotime_rank_loss` for supplementary trajectory.
- [x] Evaluate only the trajectory endpoint for completed runs; use the existing terminal endpoint constructor filtered to the same one endpoint for every noncompleted run.
- [x] Replace every fixed eight-row count/zip assumption in run, manifest, record loading, resume, and completeness validation with the scope-aware closed endpoint tuple.
- [x] Run endpoint/reload/resume tests and the full downstream-evaluation regression suite to green.

### Task 4: Receipt-Bound Output Namespace and Dual Final Script

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `scripts/run_final_downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`
- Create or modify: `tests/test_run_final_downstream_evidence.py`

**Interfaces:**
- Produces: `expected_final_downstream_output_directory(plan) -> Path`; primary maps to the receipt namespace, supplementary trajectory maps to its exact `trajectory/` child.

- [x] Add failing tests for primary/trajectory namespace separation, wrong-path rejection, in-repository rejection, and a script failure proving neither output is written if either fixed plan fails preflight.
- [x] Implement the exported receipt-bound output helper and require exact equality at run and load boundaries for production final plans.
- [x] Update the script to construct both fixed plans before writes, then run primary and trajectory plans in their exact namespaces; expose no endpoint/scope option.
- [x] Run script and namespace tests to green.

### Task 5: Final Null-DE Primary Regression

**Files:**
- Modify only if required: `maskimpute_benchmark/final_null_de.py`
- Modify: `tests/test_final_null_de.py`

**Interfaces:**
- Consumes: the expanded `EvaluatedRoundBinding` through the unchanged primary `build_final_downstream_evidence_plan`.
- Produces: an unchanged primary null-DE source path and denominator, independent of the trajectory child output.

- [x] Add a failing-or-regression test that supplies an expanded binding and confirms the expected null-DE output/source stays at the primary receipt namespace.
- [x] Make only compatibility changes required by the expanded binding; do not let final null-DE build or load trajectory downstream evidence.
- [x] Run all final-null-DE tests to green.

### Task 6: Verification and Atomic Commit

**Files:**
- Review all changed files.

**Interfaces:**
- Produces: one clean, integration-ready commit.

- [x] Run focused downstream, final-null, script, trajectory-dataset, final-analysis, and downstream-evaluation tests with warnings as errors.
- [x] Run Ruff, `compileall`, and `git diff --check`.
- [x] Inspect the exact diff for forbidden final-runner/publication-synthesis changes, materializer calls, path weakening, primary endpoint drift, and unrelated edits.
- [x] Commit the plan, implementation, and tests atomically; report exact range, test counts, and clean status.
