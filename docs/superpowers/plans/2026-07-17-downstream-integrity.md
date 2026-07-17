# Downstream Evidence Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the six publication-blocking downstream evidence integration and authority gaps while leaving trajectory execution as a separate feature.

**Architecture:** Treat the runner/final-runner stores and their source plans as the upstream authority. Bind prepared-dataset facts and exact source-plan/checkpoint identities into downstream plans and manifests, filter the selection-primary development scope through the same selection applicability rule, and require final evidence to retain its evaluated lifecycle receipt at every API boundary.

**Tech Stack:** Python 3.12, dataclasses, canonical JSON/SHA-256 evidence, AnnData, pytest, Ruff.

## Global Constraints

- Work only in `.worktrees/downstream-endpoints`; do not edit the root worktree.
- Add and run one production-shaped failing regression for each of C1, C2, C3, I1, I2, and I3 before production changes.
- Preserve the nonselection MaskImpute ablations through an explicit supplementary plan seam.
- Do not implement trajectory production execution.
- Finish with warning-clean related tests, Ruff, compileall, diff-check, and a clean commit.

---

### Task 1: Current Store Schema Contract (C1)

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: development `CheckpointStore` and final `FinalResultStore` records.
- Produces: exact source-record/storage validation accepting current pre-zero evidence and retained-matrix fields.

- [ ] Write a fixture through the current development store and a final-store-shaped schema test; assert downstream planning accepts both without aliases.
- [ ] Run those tests and record the expected schema rejection.
- [ ] Replace drifting field copies with runner-derived/shared schema helpers; include `retained_gene_count`, `observed_zero_count`, `p_pre_zero_evidence`, `p_pre_zero_encoding`, and `p_pre_zero_compression_level`.
- [ ] Run the focused schema tests to green.

### Task 2: Prepared Dataset and Source Plan Authority (I1)

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: bound `PreparedDataset` values and exact `CompetitionPlan`/`FinalExecutionPlan` entries.
- Produces: dataset bindings and plan entries that rederive and compare biological, QC, input, plan, and seed authority.

- [ ] Add a regression that rehashes forged mechanism/view/draw/method-input/QC/count/source-plan/seed fields and currently reaches plan construction.
- [ ] Run it and record the unexpected acceptance.
- [ ] Bind prepared-derived values and exact source-plan/input/entry authority, then reject every mismatch.
- [ ] Run the authority regression and existing source-drift tests to green.

### Task 3: Final Lifecycle Closure (C2 and I3)

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: `EvaluatedRoundBinding`.
- Produces: receipt-bound build/run/resume/seal/load behavior for every final-source path.

- [ ] Add one real bound run/resume/seal regression and one regression rejecting an unbound final build/load.
- [ ] Run both and record, respectively, the rebuild mismatch and unexpected acceptance.
- [ ] Forward the binding through every rebuild and require it for every production-final API boundary.
- [ ] Exercise final storage policy through the receipt-bound final fixture and run the lifecycle tests to green.

### Task 4: Selection-Primary Exact Denominator (C3)

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `maskimpute_benchmark/development_evaluation.py`
- Modify: `tests/test_downstream_evidence.py`

**Interfaces:**
- Consumes: the same selection-applicability rule used by `build_reconstruction_selection_records`.
- Produces: `selection_primary` and `supplementary_nonselection` development plan scopes.

- [ ] Add a production-configuration count/set test showing the five MaskImpute ablations create 240 extra primary denominators.
- [ ] Run it and record the exact-set failure.
- [ ] Centralize selection applicability and filter primary versus supplementary plans without dropping supplementary evidence support.
- [ ] Run count/set and schema-4 completeness tests to green.

### Task 5: Checkpoint-to-Selection Binding (I2)

**Files:**
- Modify: `maskimpute_benchmark/downstream_evidence.py`
- Modify: `maskimpute_benchmark/selection.py`
- Modify: `maskimpute_benchmark/evaluation_manifest.py`
- Modify: `tests/test_downstream_evidence.py`
- Modify: `tests/test_selection_authority.py`

**Interfaces:**
- Consumes: independently validated reconstruction checkpoint/evaluation manifest evidence.
- Produces: schema-4 binding of source checkpoint path, file/payload/plan/input hashes, and exact run statuses/reasons.

- [ ] Add an attachment regression pairing downstream evidence with a different rehashed checkpoint/status set.
- [ ] Run it and record the unexpected attachment.
- [ ] Carry the source checkpoint identity/status digest into the downstream manifest and schema-4 binding; cross-compare it with evaluation-manifest validation output.
- [ ] Run attachment and evaluation-manifest regressions to green.

### Task 6: Verification and Commit

**Files:**
- Review all changed files.

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: a clean, publication-integrable commit.

- [ ] Run the downstream/trajectory/selection/evaluation related suites with `-W error`.
- [ ] Run Ruff, `compileall`, and `git diff --check`.
- [ ] Review the exact diff for trajectory scope creep and root-worktree changes.
- [ ] Commit the verified changes and report the exact range and residual integration seam.
