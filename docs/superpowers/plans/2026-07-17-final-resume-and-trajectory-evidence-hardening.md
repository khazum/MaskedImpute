# Final Resume and Trajectory Evidence Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every interrupted canonical final-round publication resumable without blessing unvalidated bytes, and make downstream trajectory evidence replay the exact frozen plan, manifest, records, and terminal denominator.

**Architecture:** Resume performs transaction cleanup, validates every unjournaled runner-owned artifact against independently reconstructed frozen authorities, appends the exact cumulative inventory to the result journal, and only then reloads the final claim. Final analysis reconstructs the trajectory plan from its closed payload, parses the bound execution manifest and records, replays terminal-denominator validation, and invokes the existing frozen-authority rederivation before accepting the evidence.

**Tech Stack:** Python 3.11, pytest, immutable canonical JSON/HDF5 artifacts, append-only study result journal.

## Global Constraints

- Keep the public primary final plan strict at exactly 40 datasets.
- Preserve the separate trajectory dataset, authority, plan, execution store, and inventory scopes.
- Never journal arbitrary bytes merely because they occupy a known path.
- Recover an owned unique trajectory H5AD without a receipt by removing and regenerating it; reject a receipt without its dataset.
- Validate and journal recovered publications before `load_final_manifest_claim` performs frozen-repository verification.
- Preserve the sole final evaluation receipt and the requirement that scaling and trajectory evidence are complete first.

---

### Task 1: Interrupted publication reconciliation

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Test: `tests/test_final_runner.py`

**Interfaces:**
- Consumes: frozen method receipt, execution claim bytes, final dataset bindings, registered trajectory receipt, execution authorities, and scaling authority.
- Produces: `_reconcile_interrupted_final_publications(repository, round_dir, frozen_method) -> object | None` and safe incomplete-trajectory cleanup used before claim reload.

- [ ] **Step 1: Write failing crash-seam tests**

Add parameterized regressions for interruption after the registered trajectory dataset, primary execution authority, trajectory execution authority, final/trajectory execution manifest, and a scaling checkpoint. Each regression must prove that only independently validated owned files reach the cumulative journal and that claim reload occurs afterward.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest -q tests/test_final_runner.py -k 'reconcile or unreceipted or publication_order'`

Expected: failures show the current claim reload precedes reconciliation or the incomplete trajectory pair blocks inventory construction.

- [ ] **Step 3: Implement exact recovery and publication ordering**

Factor direct preparation from already validated final bindings, reconstruct the primary and trajectory plans and stores without accepting an unverified claim snapshot, force-validate any scaling checkpoint against its frozen authority, derive the closed owned inventory, and append it through `record_incremental_results`. Add immediate journal callbacks after trajectory dataset and both authority materializations. Keep manifest/checkpoint callbacks after their immutable publication.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2 and require every selected regression to pass without warnings.

### Task 2: Strict downstream trajectory replay

**Files:**
- Modify: `maskimpute_benchmark/final_analysis.py`
- Modify: `tests/test_final_analysis.py`

**Interfaces:**
- Consumes: embedded trajectory evidence, evaluated result bindings, canonical round path, and the primary final-plan digest.
- Produces: strict validation that returns the unchanged evidence only after exact plan/manifest/record/terminal replay.

- [ ] **Step 1: Write malformed-evidence regressions**

Create negative cases for a non-list configuration payload, arbitrary or malformed entry, nonterminal validation status, inconsistent counts/status map, empty or changed record payload hashes, and a trajectory execution manifest whose payload is not the bound plan.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `env -u LD_LIBRARY_PATH PYTHONNOUSERSITE=1 /tmp/maskimpute-supported/bin/python -m pytest -q tests/test_final_analysis.py -k trajectory`

Expected: at least one locally rehashed malformed evidence case is accepted by the current receipt-local validator.

- [ ] **Step 3: Implement semantic replay**

Reconstruct closed `AuthorizedConfiguration`, `RunPlanEntry`, `FinalPlanEntry`, and `TrajectoryExecutionPlan` values from the embedded payload; compare canonical reserialization; read and hash the trajectory execution manifest and every referenced record; replay `validate_trajectory_execution_for_evaluation`; compare the entire validation object; and finally invoke frozen-authority rederivation against the evaluated inventory.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the command from Step 2 and require every selected regression to pass without warnings.

### Task 3: Verification and commit

**Files:**
- Verify all files modified above.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: one clean follow-up commit on `codex/trajectory-execution-integrated`.

- [ ] **Step 1: Run focused and adjacent strict suites**

Run `tests/test_final_runner.py`, `tests/test_final_analysis.py`, `tests/test_trajectory_dataset.py`, `tests/test_scaling_panel.py`, and the authoritative serialized benchmark-runner nodes with the supported interpreter and appropriate spawn flags.

- [ ] **Step 2: Run static verification**

Run Ruff format/check, `git diff --check`, and supported-interpreter `compileall` over changed production and test files.

- [ ] **Step 3: Commit and verify clean state**

Commit the recovery and semantic-validation changes, record the exact parent-to-head range, and require `git status --short` to be empty.
