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
and compare status semantics before execution. Evaluated consumers continue to
reconstruct authority solely from frozen receipts.

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

---

### Task 1: Fail-closed runtime preclaim and path propagation

**Files:**
- Modify: `maskimpute_benchmark/final_runner.py`
- Modify: `scripts/run_frozen_final.py`
- Modify: `tests/test_final_runner.py`
- Modify: `tests/test_dataset_registry.py` only if a production boundary test belongs there
- Add/commit: `docs/superpowers/specs/2026-07-17-final-runtime-preclaim-design.md`
- Add/commit: `docs/superpowers/plans/2026-07-17-final-runtime-preclaim.md`

**Interfaces:**
- Produces:
  `run_frozen_final_round(repository: Path, round_dir: Path, *, simulator_assets_root: Path, simulator_r_environment: Path) -> dict[str, object]`
- Extends:
  `load_prepared_final_panel(repository: Path, round_dir: Path, *, allow_evaluated: bool = False, simulator_assets_root: Path | None = None, simulator_r_environment: Path | None = None)`
- Preserves: `load_prepared_final_panel(..., allow_evaluated=True)` with neither
  live path for evaluated downstream replay.

- [ ] **Step 1: Add RED public-contract and ordering tests**

Add tests that inspect the exact keyword-only signature; make the authoritative
runtime loader raise for both unclaimed and resumable fixtures; instrument
`assert_final_runnable`, stale-temporary cleanup, transaction recovery, scaling
recovery, and reconciliation; assert none are called and round bytes do not
change. Run the exact nodes and confirm failure because the runner lacks the
required preclaim API/ordering.

- [ ] **Step 2: Add RED propagation, semantic-match, and cleanup tests**

Use a context-capable fake runtime snapshot exposing defensive
`semantic_sha256`/`semantic_receipt`. Record kwargs received by
`generate_dataset_panel` and `load_prepared_final_panel`. Require the exact path
objects at both boundaries, one deterministic preclaim close, exact generated
status comparison, and no method-registry/executor call after a mismatched
status receipt. Confirm RED for missing behavior rather than fixture errors.

- [ ] **Step 3: Add RED prepared-panel state tests**

Require a complete path pair for running validation and assert both values are
forwarded on both status validations. Reject every partial pair. Retain the
existing evaluated fixture with `allow_evaluated=True` and neither path; assert
two stable evaluated snapshots and no live runtime lookup.

- [ ] **Step 4: Implement the minimal preclaim boundary**

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

- [ ] **Step 5: Add and satisfy the CLI RED test**

Require:

```text
--simulator-assets-root PATH
--simulator-r-environment PATH
```

Pass them as keyword-only arguments. Assert help still omits repository,
environment-selection, seed, mechanism, configuration, and method overrides.

- [ ] **Step 6: Run focused and complete verification**

Run the new nodes under `-W error`, then the full final-runner, dataset-registry,
simulator-runtime, downstream-evidence, and final-null-DE suites under
`-W error`. Run Ruff format/check, `compileall`, `git diff --check`, inspect the
exact range, and commit atomically. Do not run `scripts/run_frozen_final.py`
against a round.
