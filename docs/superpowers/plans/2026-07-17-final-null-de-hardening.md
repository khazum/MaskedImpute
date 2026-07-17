# Final Null-DE Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate downstream `7f6dd07`, restore final-null-DE compatibility, and reject completed manifests whose record prefix is shorter than the frozen denominator.

**Architecture:** Merge the approved downstream branch before changing final-null-DE behavior, preserving the row-order-independent v3 split receipt. Keep the existing immutable-prefix design, but make completeness an invariant of manifest construction and loading so a partial prefix can never be represented or accepted as completed.

**Tech Stack:** Python 3.10+, dataclasses, canonical JSON/SHA-256, pytest 8, Ruff, Git.

## Global Constraints

- Work only in `/home/marcinmaleclocal/Coding/MaskedImpute/.worktrees/final-null-de`.
- Do not run actual final evidence.
- Use `PYTHONWARNINGS=error` for focused tests.
- Preserve both downstream revision-aware plan fields and the v3 stable-ID permutation digest.

---

### Task 1: Integrate approved downstream

**Files:**
- Merge: `7f6dd07` into `codex/final-null-de`
- Verify: `maskimpute_benchmark/development_evaluation.py`

**Interfaces:**
- Consumes: downstream schema-v3 `DownstreamEvidencePlan`
- Produces: one branch containing revision-aware downstream fields and `maskimpute-null-de-balanced-split-v3`

- [x] Merge `7f6dd07` without dropping either branch's changes.
- [x] Inspect the merged diff and verify the v3 split implementation remains intact.
- [x] Run the focused final-null-DE test to reproduce the known fixture incompatibility.

### Task 2: Restore downstream-plan compatibility

**Files:**
- Modify: `tests/test_final_null_de.py`

**Interfaces:**
- Consumes: `DownstreamEvidencePlan(..., development_revision_versions, development_sources, ...)`
- Produces: an exact final-source fixture with both revision-only fields empty

- [x] Treat the post-merge constructor failures as RED evidence.
- [x] Add `development_revision_versions=()` and `development_sources=()` to the final fixture.
- [x] Run `tests/test_final_null_de.py` under warning-strict pytest and verify GREEN.

### Task 3: Enforce complete final-null-DE manifests

**Files:**
- Modify: `tests/test_final_null_de.py`
- Modify: `maskimpute_benchmark/final_null_de.py`

**Interfaces:**
- Consumes: `_manifest_payload(output_root, plan, records)` and `load_final_null_de_manifest(output_root)`
- Produces: `FinalNullDEError` whenever a completed manifest is built from or resolves to fewer records than `len(plan.source_plan.entries)`

- [x] Add a regression that runs one denominator, forges a canonical completed manifest for that prefix, and requires both construction/loading and rerun to reject it.
- [x] Run only that regression and verify it fails for the missing completeness invariant.
- [x] Add the minimal denominator equality checks in manifest production and loading.
- [x] Re-run the regression and focused final/downstream/development suites under `PYTHONWARNINGS=error`.

### Task 4: Verify and commit

**Files:**
- Check all touched Python files and this plan

**Interfaces:**
- Consumes: the merged and hardened branch
- Produces: clean focused tests, Ruff, compilation, diff check, and committed history

- [x] Run Ruff check and format verification for touched Python files.
- [x] Compile touched Python files.
- [x] Run `git diff --check` and inspect status/diff.
- [ ] Commit the implementation and report the exact new commit range.
