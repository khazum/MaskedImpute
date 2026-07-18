# Frozen Final Runtime Preclaim Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans. Follow
> strict RED-GREEN TDD and commit only after the complete gate is green.

**Goal:** Require and revalidate exact external simulator runtime authorities
before any frozen-final claim/resume mutation while preserving path-free
evaluated replay.

**Architecture:** Keep `load_simulator_runtime_assets` as the sole live path
authority boundary. Capture its path-independent receipt before lifecycle
mutation, pass exact operational paths through generation/running validation,
and compare status semantics before execution. Establish one centralized stable
process environment at both supported final CLI boundaries before runtime
verification. Evaluated consumers continue to reconstruct authority solely from
frozen receipts.

**Tech Stack:** Python 3.11+, pytest, pathlib, existing MaskedImpute lifecycle,
dataset-registry, and simulator-runtime contracts.

## Global Constraints

- Never execute a real frozen final round in tests or during implementation.
- Never open final scientific holdout truth or change methods, configurations,
  seeds, dataset design, denominators, or claim gates.
- External paths are operational locators and must never enter path-independent
  scientific receipts.
- Invalid runtime paths must precede every claim or resume mutation.
- Running validation requires both paths; evaluated replay requires neither.
- All private runtime snapshots close deterministically.
- Supported final CLI execution uses exactly
  `PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin` with
  `LD_LIBRARY_PATH` absent before runtime verification.
- Development dataset generation does not mutate its caller environment.
- Do not edit the simulator R lock or runtime authority here; regenerate them
  later under the sanctioned environment and integrate that authority change
  separately after review.

---

### Task 1: Fail-closed runtime preclaim and path propagation

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Modify: `scripts/run_frozen_final.py`
- Modify: `scripts/generate_study_datasets.py`
- Add: `maskimpute_benchmark/operational_environment.py`
- Modify: `tests/test_final_runner.py`
- Modify: `tests/test_dataset_registry.py`
- Add/commit: `docs/superpowers/specs/2026-07-17-final-runtime-preclaim-design.md`
- Add/commit: `docs/superpowers/plans/2026-07-17-final-runtime-preclaim.md`

**Interfaces:**
- Produces:
  `run_frozen_final_round(repository: Path, round_dir: Path, *, simulator_assets_root: Path, simulator_r_environment: Path) -> dict[str, object]`
- Extends:
  `load_prepared_final_panel(repository: Path, round_dir: Path, *, allow_evaluated: bool = False, simulator_assets_root: Path | None = None, simulator_r_environment: Path | None = None)`
- Preserves: `load_prepared_final_panel(..., allow_evaluated=True)` with neither
  live path for evaluated downstream replay.
- Produces:
  `establish_supported_final_runtime_environment() -> None`, backed by one
  importable exact `SUPPORTED_FINAL_RUNTIME_PATH` constant.

- [x] **Step 1: Add RED public-contract and ordering tests**

Add tests that inspect the exact keyword-only signature; make the authoritative
runtime loader raise for both unclaimed and resumable fixtures; instrument
`assert_final_runnable`, stale-temporary cleanup, transaction recovery, scaling
recovery, and reconciliation; assert none are called and round bytes do not
change. Run the exact nodes and confirm failure because the runner lacks the
required preclaim API/ordering.

- [x] **Step 2: Add RED propagation, semantic-match, and cleanup tests**

Use a context-capable fake runtime snapshot exposing defensive
`semantic_sha256`/`semantic_receipt`. Record kwargs received by
`generate_dataset_panel` and `load_prepared_final_panel`. Require the exact path
objects at both boundaries, one deterministic preclaim close, exact generated
status comparison, and no method-registry/executor call after a mismatched
status receipt. Confirm RED for missing behavior rather than fixture errors.

- [x] **Step 3: Add RED prepared-panel state tests**

Require a complete path pair for running validation and assert both values are
forwarded on both status validations. Reject every partial pair. Retain the
existing evaluated fixture with `allow_evaluated=True` and neither path; assert
two stable evaluated snapshots and no live runtime lookup.

- [x] **Step 4: Implement the minimal preclaim boundary**

Import `load_simulator_runtime_assets` locally at the runner boundary. Validate
and close one preclaim snapshot, retaining only:

```python
preclaim_sha256 = runtime_assets.semantic_sha256
preclaim_receipt = runtime_assets.semantic_receipt
```

Perform this before the `resuming` branch. Pass the exact paths into generation
and prepared-panel validation. Require both generated semantic fields to equal
the preclaim fields before creating method registries, environments, stores, or
executors. Extend running status calls to forward the path pair; keep evaluated
fallback path-free.

- [x] **Step 5: Add RED operational-environment boundary tests**

Require the centralized helper to install the exact sanctioned `PATH` and
remove `LD_LIBRARY_PATH`. Instrument both supported CLI downstream calls and
require the sanitized environment to be visible before the frozen runner or
final dataset generation. Require development generation to preserve the
caller's environment. Confirm RED for the absent centralized boundary.

- [x] **Step 6: Implement the centralized operational environment**

Add one importable helper/constant and call it from
`scripts/run_frozen_final.py` on every invocation and from
`scripts/generate_study_datasets.py` only for the final namespace. Do not
duplicate the environment literal and do not change direct library semantics.

- [x] **Step 7: Add and satisfy the CLI runtime-locator RED test**

Require:

```text
--simulator-assets-root PATH
--simulator-r-environment PATH
```

Pass them as keyword-only arguments. Assert help still omits repository,
environment-selection, seed, mechanism, configuration, and method overrides.

- [x] **Step 8: Run focused and complete verification**

Run the new nodes under `-W error`, then the full final-runner, dataset-registry,
simulator-runtime, downstream-evidence, and final-null-DE suites under
`-W error`, including both script boundary tests. Run Ruff format/check,
`compileall`, `git diff --check`, inspect the exact range, and commit atomically.
Do not run `scripts/run_frozen_final.py` against a round. Leave the simulator R
lock and runtime authority unchanged; their stable-environment regeneration is
a later separately reviewed change.
