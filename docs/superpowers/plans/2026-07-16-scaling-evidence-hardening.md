# Scaling Evidence Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make scaling evidence independently replayable, resource-authorized,
efficient to resume, and inseparable from the claimed final-round receipt.

**Architecture:** A fresh `ScalingResultStore` performs one full checkpoint
validation. Existing retained H5ADs are compared with a fresh deterministic
SymSim regeneration. Completed native and evaluator matrices are decoded from
bounded zlib artifacts, native-to-evaluator conversion and all metric rows are
replayed, and executor receipts are checked against plan-bound resource
ceilings and telemetry labels. Appends validate an isolated per-run directory,
compare the complete on-disk history with cached authority, and publish a new
immutable numbered snapshot under a stable directory-descriptor lock. The
claimed final runner journals every snapshot, requires the complete denominator,
and binds all scaling bytes into its sole final-evaluation receipt.

**Tech Stack:** Python 3.11, NumPy, AnnData, zlib, canonical JSON/SHA-256, pytest, Ruff.

## Global Constraints

- Keep the fixed 20-run scaling denominator and scientific metrics unchanged.
- Retain moderate H5ADs plus bounded compressed native and evaluator outputs; discard severe/native simulator data after receipts.
- Derive authority only from tracked protocol, scaling contract, method registry, runtime, implementation, and deterministic SymSim execution.
- Keep dense work cell-linear and gene-correlation work at 500-by-500.

---

### Task 1: RED integrity and performance tests

**Files:**
- Modify: `tests/test_scaling_panel.py`

**Interfaces:**
- Consumes: `ScalingResultStore`, `scaling_attempt_record`, deterministic fixture simulator.
- Produces: regressions for H5AD coordinated rehash, finite result/resource rehash, compressed-output replay, and one-pass resume hashing.

- [x] Add a deterministic fixture simulator that returns request-bound `SimulationArtifact` objects and use it to build accepted dataset receipts.
- [x] Add a fresh-store test that mutates an H5AD and all self-hashes but expects regenerated SymSim authority rejection.
- [x] Add a fresh-store test that rewrites finite metric/resource fields and all self-hashes but expects plan/replay rejection.
- [x] Add a test that counts H5AD file hash calls across appends and requires no recursive historical rehashing.
- [x] Run each new test and confirm it fails because the required hardening is absent.

### Task 2: Bound resources and retained evaluator evidence

**Files:**
- Modify: `maskimpute_benchmark/scaling.py`
- Modify: `study/scaling_panel.json`

**Interfaces:**
- `ScalingPlanEntry` produces exact `timeout_seconds`, `max_rss_bytes`, `max_gpu_bytes`, `rss_measurement`, and `gpu_measurement` authority.
- Transient attempt records carry native and evaluator ndarrays plus executor-receipt bytes; stored records carry exact zlib and sidecar receipts.
- Fresh validation replays the native converter before comparing `_bounded_scaling_metric_values` rows exactly with checkpoint metrics.

- [x] Extend plan entries from the method registry and bind exact resource ceilings, native output scales, and Linux telemetry labels into the plan hash.
- [x] Retain canonical native and evaluator matrices with zlib level 6 and an executor receipt captured before conversion.
- [x] Decode with exact compression bounds; reject trailing, short, nonfinite, negative, or wrong-shape bytes; replay native conversion and every metric row exactly.
- [x] Reject forged runtime/RSS/GPU values and measurement labels while preserving valid simultaneous RSS/GPU exceedance evidence.
- [x] Run focused RED/GREEN tests, including warnings-as-errors tamper cases.

### Task 3: Deterministic dataset authority and transactional snapshots

**Files:**
- Modify: `maskimpute_benchmark/scaling.py`
- Modify: `tests/test_scaling_panel.py`

**Interfaces:**
- A store receives the simulator used by the publication execution; a fresh validation regenerates each existing size in a temporary root and compares semantic, truth, provenance/config, severe-view, and native-manifest identities.
- `load(force_validate=False)` returns the already validated in-memory snapshot after its first call; `force_validate=True` reopens all evidence.

- [x] Regenerate each retained dataset once per fresh validation and compare the independently produced generator receipt fields.
- [x] Cache an isolated checkpoint snapshot, return detached copies, and require unchanged on-disk checkpoint history before an append.
- [x] Make `_write` construct and cache the immutable snapshot directly instead of recursively loading it.
- [x] Publish each validated run as one atomic directory transaction and make closed crash orphans retryable.
- [x] Reject symbolic links in every run/checkpoint path component on append and fresh validation.
- [x] Remove repeated `load()` calls from `execute_scaling_plan`; force one terminal validation before return.
- [x] Run focused tests and confirm GREEN.

### Task 4: Binding documentation and verification

**Files:**
- Modify: `docs/superpowers/specs/2026-07-16-scaling-panel-design.md`
- Modify: `study/scaling_panel.json`

**Interfaces:**
- The tracked contract fixes native/evaluator zlib retention, the 7.4-GB raw-matrix bound, executor receipts, conversion/metric replay, and regeneration policy.

- [x] Amend the design and artifact policy to describe bounded native/evaluator evidence, exact conversion and metric replay, deterministic SymSim regeneration, telemetry, cache isolation, atomic run publication, and symlink rejection.
- [x] Run the full scaling test file, diagnose its four stale-regex failures, and rerun the affected parameterized cases under `-W error`.
- [x] Run Ruff and bytecode compilation over all changed Python files.
- [x] Review `git diff --check`, the complete diff, and worktree status; prepare one focused hardening commit.

### Task 5: Bind scaling to the final-round publication lifecycle

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Modify: `maskimpute_benchmark/final_analysis.py`
- Modify: `maskimpute_benchmark/scaling.py`
- Modify: `scripts/run_scaling_panel.py`
- Modify: `tests/test_final_runner.py`
- Modify: `tests/test_final_analysis.py`
- Modify: `tests/test_scaling_panel.py`

**Interfaces:**
- The scaling CLI consumes only a claimed canonical round and writes beneath
  `<round>/results/scaling`.
- The sole evaluation manifest carries canonical `scaling_evidence` and an
  exact cumulative inventory; the publication loader requires that evaluated
  receipt.
- `scaling_storage_preflight(authority)` returns a pure, receipted storage bound
  for later composition with primary and trajectory preflights.

- [x] Move scaling into the supplementary pre-receipt final-runner phase while
  preserving the planned trajectory insertion seam.
- [x] Require a complete, replay-valid scaling denominator before recording the
  sole evaluation receipt.
- [x] Replace the mutable checkpoint with contiguous immutable numbered
  snapshots and journal every cleaned dataset and attempt snapshot.
- [x] Bind the full plan, latest checkpoint payload, all checkpoint history, and
  every dataset/run/receipt byte into the final cumulative inventory.
- [x] Require canonical evaluated lifecycle and exact inventory in the
  publication scaling loader.
- [x] Serialize writers with a stable directory-descriptor lock and recover only
  validated, unreferenced closed transactions.
- [x] Enforce plan-derived H5AD size ceilings, bounded metadata validation before
  AnnData deserialization, and file-identity revalidation after it.
- [x] Add the pure scaling storage preflight and its authority-derived receipt.
- [x] Cover receipt replacement, wrong-round, incomplete-denominator,
  two-writer, immutable-history, H5AD, loader, and journal regressions.
- [x] Run warning-strict integration tests, Ruff, bytecode compilation, and
  `git diff --check`; then create the focused commit.
