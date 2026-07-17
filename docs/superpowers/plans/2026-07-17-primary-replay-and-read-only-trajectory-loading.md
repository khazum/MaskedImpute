# Primary Replay and Read-Only Trajectory Loading Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make trajectory evidence acceptance depend on a freshly reconstructed, complete nonzero primary execution and make trajectory evidence replay incapable of publishing or recovering dataset files.

**Architecture:** Split primary execution-authority construction into a deterministic read-only derivation and a thin immutable publisher. During trajectory-chain validation, revalidate the frozen method, method registry, execution environment, exact 40-dataset panel, rebuilt primary plan, primary result store, manifest, records, and terminal denominator before validating the trajectory descendants. Split existing trajectory materialization from a strict loader that only authenticates an already complete dataset/receipt pair; evidence replay uses only the loader.

**Tech Stack:** Python 3.11, pytest, immutable canonical JSON/HDF5 artifacts, append-only final result stores.

## Global Constraints

- The primary plan is rebuilt from exactly 40 final dataset authorities and may not be empty.
- The caller-supplied primary plan digest must equal the rebuilt plan digest; it is never an opaque root.
- Primary authority, score authority, calibration, execution manifest, every record, and terminal denominator are validated against reconstructed frozen inputs.
- Evaluated panel replay revalidates the inode-rooted result journal, exact lifecycle claim, frozen protocol, 60-seed design, path-free simulator runtime/source/R-lock authority, and all 40 status rows.
- `load_prepared_trajectory_dataset(repository, round_dir)` is strictly read-only and requires both existing dataset and receipt files.
- Missing or raced trajectory files fail closed without regeneration, removal, directory creation, or immutable publication.
- Simulator runtime snapshots have an explicit close/context lifecycle and are released deterministically by dataset generation and validation.
- Preserve running-round replay and evaluated-round final-analysis replay.
- Commit the production, tests, and plan atomically; do not merge.

---

### Task 1: Strict read-only trajectory dataset loading

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Test: `tests/test_final_runner.py`

**Interfaces:**
- Consumes: canonical `study/trajectory_panel.json`, existing `results/trajectory/dataset/evaluator.h5ad`, and existing `dataset_receipt.json`.
- Produces: `load_prepared_trajectory_dataset(repository: Path, round_dir: Path) -> TrajectoryPreparedDataset` with no write path.

- [ ] **Step 1: Write missing, race, and tree-stability regressions**

Add tests that materialize a valid pair once, then prove the loader rejects either missing half without invoking generation or publication; prove a receipt replacement between reads is rejected and remains untouched; and snapshot every path, regular-file digest, mode, size, and modification time before and after a successful load.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `env -u LD_LIBRARY_PATH -u PYTHONNOUSERSITE -u PYTHONDONTWRITEBYTECODE /tmp/maskimpute-supported/bin/python -m pytest -q tests/test_final_runner.py -k 'load_prepared_trajectory_dataset or trajectory_loader' --maxfail=1`

Expected: collection or assertion failure because no read-only loader exists and replay still calls the materializer.

- [ ] **Step 3: Implement the minimal loader and refactor materialization**

Read and canonicalize the trajectory authority and receipt, construct and compare `RegisteredTrajectoryBinding`, read the bound H5AD through `_read_bound_h5ad`, prepare it, recompute the receipt, then reread authority and receipt bytes to detect cross-file races. Make materialization generate/publish only when the pair is absent or an explicitly recoverable unreceipted dataset exists, and return through the loader.

- [ ] **Step 4: Run focused GREEN**

Run the Step 2 command and require all selected tests to pass without warnings.

### Task 2: Exact primary execution reconstruction

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Test: `tests/test_final_runner.py`

**Interfaces:**
- Consumes: `validate_frozen_method(repository)`, `load_method_registry(study/methods.json)`, `load_prepared_final_panel(repository, round_dir)` with evaluated-round read-only validation, canonical execution claim, reconstructed `ExecutionEnvironmentRegistry`, and persisted primary authority/result files.
- Produces: `_validate_trajectory_primary_authority_chain(...)` only after exact primary authority bytes, `FinalExecutionPlan`, `FinalResultStore.load_records()`, `load_manifest()`, and `validate_final_execution_for_evaluation()` all agree.

- [ ] **Step 1: Write nonzero-valid and coordinated-rehash regressions**

Create an exact 40-binding primary plan, at least one complete terminal record per rebuilt plan entry, and its real immutable manifest. First require the unmodified chain to pass. Then coherently change and rehash primary authority identity, primary manifest plan/input identity, trajectory authority, and trajectory score authority; require validation to fail because frozen reconstruction still yields the original authority and plan. Add an explicit 0/0 manifest rejection.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `env -u LD_LIBRARY_PATH -u PYTHONNOUSERSITE -u PYTHONDONTWRITEBYTECODE /tmp/maskimpute-supported/bin/python -m pytest -q tests/test_final_runner.py -k 'trajectory_authority_chain' --maxfail=1`

Expected: the coordinated rehash is accepted by the current self-consistency-only validator or the zero-record fake passes.

- [ ] **Step 3: Extract pure authority derivation and rebuild the primary chain**

Factor deterministic calibration, count-score, authority payload, and `ExecutionAuthorityContext` construction out of `materialize_final_execution_authority`; keep publication in the materializer. Reconstruct the environment and runtime lock, load the exact panel, compare every derived primary authority file byte-for-byte, build the plan, require its nonzero digest to equal both the manifest and caller digest, then validate every primary record, the complete immutable manifest, and the terminal denominator through `FinalResultStore`.

- [ ] **Step 4: Preserve evaluated final-analysis loading**

Extend the public final-panel loader with a read-only evaluated-round validation path: validate the evaluated state chain, result manifest, and frozen repository before and after direct status/H5AD preparation. Running rounds continue using `validate_dataset_status`.

- [ ] **Step 5: Run focused GREEN**

Run the Step 2 command and the trajectory evidence tests; require all to pass without warnings.

### Task 3: Read-only evidence replay and complete verification

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Test: `tests/test_final_runner.py`
- Verify: `tests/test_final_analysis.py`
- Verify: `tests/test_scaling_panel.py`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: evidence rederivation whose filesystem snapshot is unchanged and one clean atomic follow-up commit.

- [ ] **Step 1: Refactor replay to call only the loader**

Replace `materialize_prepared_trajectory_dataset` in `_rederive_trajectory_evidence_before_receipt` with `load_prepared_trajectory_dataset`; a test monkeypatches the materializer to raise if replay reaches it and compares the complete tree snapshot before and after replay.

- [ ] **Step 2: Run focused and adjacent GREEN suites**

Run trajectory-focused final-runner and final-analysis tests, then all of `tests/test_final_runner.py`, `tests/test_final_analysis.py`, and `tests/test_scaling_panel.py` with the supported interpreter and warnings promoted to errors where the spawned-runtime contract permits.

- [ ] **Step 3: Run static gates**

Run Ruff format check, Ruff lint, `git diff --check`, and supported-interpreter `compileall` over changed production and test files.

- [ ] **Step 4: Commit atomically and verify clean status**

Commit the plan, tests, and implementation in one follow-up commit, record both `a76b698..HEAD` and `3485d03..HEAD`, and require `git status --short` to be empty. Do not merge.

### Task 4: Review hardening for evaluated panel and late-input races

**Files:**
- Modify: `maskimpute_benchmark/datasets.py`
- Modify: `maskimpute_benchmark/final_runner.py`
- Modify: `maskimpute_benchmark/simulators/runtime_assets.py`
- Test: `tests/test_dataset_registry.py`
- Test: `tests/test_final_runner.py`
- Test: `tests/test_simulator_runtime_assets.py`

- [ ] **Step 1: Reject a real evaluated lifecycle carrying another claim**

Journal and evaluate a canonical 40-row status whose claim ID is coherently replaced. Require evaluated panel loading to reject it using the claim returned by the validated lifecycle chain.

- [ ] **Step 2: Validate a production-shaped evaluated panel**

Generate all 40 tiny final datasets from the exact 60-seed fixture, record the genuine evaluated receipt, and require the evaluated status validator to reconstruct protocol, design, row, runtime, source-ledger, and R-lock authority. Coherently rehash one source authority field and require rejection.

- [ ] **Step 3: Rebuild all primary frozen inputs after store replay**

After record/manifest/terminal validation, fully revalidate the original runtime snapshot and independently reload frozen method, method registry, execution environment, final panel, derived authority, and final plan. A late panel replacement must fail.

- [ ] **Step 4: Close runtime snapshots explicitly**

Expose idempotent `SimulatorRuntimeAssets.close()` plus context-manager use, close dataset-generation and status-validation snapshots in `finally` blocks, and prove warning-strict cleanup and use-after-close failure.
