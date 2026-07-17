# Frozen Final Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict, canonical generator for publication analysis from only a validated evaluated frozen-final round.

**Architecture:** A new module separates immutable-evidence loading from pure report construction.  It adapts validated long-form rows to the existing hierarchical statistics implementation, keeping biological draws as inference units, and exposes a one-positional-argument CLI that writes only to stdout.

**Tech Stack:** Python 3.11+, standard library, NumPy, existing `maskimpute_benchmark.statistics`, pytest, Ruff.

## Global Constraints

- Do not modify `runner.py`, `final_runner.py`, or `development_evaluation.py`.
- Consume only canonical, hash-bound records declared by an evaluated receipt.
- Use no scientific CLI overrides and fabricate no scientific values.
- Use biological draws, never cells or genes, as inference units.
- Bootstrap policy is fixed at 10,000 replicates and seed 20,260,712.
- Pareto directions must come from frozen protocol/contract authority.
- Never import reconstruction direction constants from live code; bind frozen
  gate declarations to the immutable method-commit selection source.
- Require the exact score-aware final record schema and keep realized
  `p_pre_zero` metrics in a separate descriptive family.

---

### Task 1: Pure analysis contract

**Files:**
- Create: `tests/test_final_analysis.py`
- Create: `maskimpute_benchmark/final_analysis.py`

**Interfaces:**
- Consumes: validated execution records, protocol mapping, selection-contract mapping.
- Produces: `build_final_analysis(records, *, protocol, selection_contract, input_bindings) -> dict[str, object]`.

- [ ] **Step 1: Write failing tests for normalization and denominators**

  Create records covering `completed`, `failed`, `timeout`,
  `resource_exceeded`, and `unavailable`, then assert exact run counts, metric
  analytic-status counts, reason counts, and `completed -> ok` normalization.

- [ ] **Step 2: Run the focused test and verify RED**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py -k denominator` and expect import/function failure.

- [ ] **Step 3: Implement strict input normalization**

  Validate exact run/metric identity fields, status/value/reason consistency,
  complete per-record metric names, deterministic singleton seed encoding, and
  canonical ordering.  Return denominator evidence without dropping terminal
  failures.

- [ ] **Step 4: Run the focused test and verify GREEN**

  Re-run the command from Step 2 and expect all selected tests to pass.

- [ ] **Step 5: Write failing tests for biological-draw summaries and inference**

  Assert that three stochastic seeds and multiple technical views collapse to
  one independent draw; assert biological-draw median/Q1/Q3, paired W/T/L,
  probability of improvement, fixed bootstrap metadata/checksum, Holm grouping,
  and explicit unavailability for an unpaired comparator.

- [ ] **Step 6: Run inference tests and verify RED**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py -k 'summary or paired or holm'` and expect missing-section failures.

- [ ] **Step 7: Implement fixed-policy summaries and comparisons**

  Collapse seed means within dataset/view and view means within biological draw;
  call `hierarchical_paired_bootstrap`, retain compact distribution bindings,
  and apply `holm_adjust` within each comparator's declared primary family.

- [ ] **Step 8: Run inference tests and verify GREEN**

  Re-run Step 6 and expect all selected tests to pass.

- [ ] **Step 9: Write failing variance and Pareto tests**

  Assert all identifiable variance components/counts, reason-coded unavailable
  components, Pareto non-domination when explicit lower directions exist, and
  Pareto unavailability when directions are absent or non-reconstruction.

- [ ] **Step 10: Run variance/Pareto tests and verify RED**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py -k 'variance or pareto'` and expect missing-section failures.

- [ ] **Step 11: Implement variance and authority-gated Pareto evidence**

  Adapt `summarize_seed_variance` output with component availability reasons;
  compute domination only over complete biological-draw medians for explicitly
  lower-is-better reconstruction metrics.

- [ ] **Step 12: Add the separate realized-score family**

  Validate the exact `p_pre_zero_evidence` schema and summarize overall,
  library-size-quartile, and truth-expression-bin metrics at biological-draw
  level.  Declare AUROC/AUPRC higher, Brier/log-loss/ECE lower, and calibration
  intercept/slope descriptive; do not add score endpoints to reconstruction
  Holm or Pareto families.

- [ ] **Step 13: Run the complete pure-analysis tests**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py` and expect all tests to pass.

### Task 2: Evaluated evidence loader

**Files:**
- Modify: `tests/test_final_analysis.py`
- Modify: `maskimpute_benchmark/final_analysis.py`

**Interfaces:**
- Consumes: repository `Path`, evaluated round `Path`, lifecycle records, evaluation receipt, execution manifest, record files.
- Produces: `generate_final_analysis(repository: Path, round_dir: Path) -> dict[str, object]` and `FinalAnalysisContractError`.

- [ ] **Step 1: Write failing evidence-integrity tests**

  Build a minimal canonical evaluated-evidence fixture around a monkeypatched
  lifecycle boundary, then cover exact receipt schema, declared result paths,
  raw/payload/plan/ordered-record hash bindings, path traversal, symlink or
  noncanonical files, and post-read changes.

- [ ] **Step 2: Run loader tests and verify RED**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py -k 'evidence or receipt or manifest or tamper'` and expect loader failures.

- [ ] **Step 3: Implement read-only evaluated evidence validation**

  Reuse study lifecycle validators for evaluated state and frozen repository;
  securely read canonical JSON with `O_NOFOLLOW`; validate the exact evaluation
  and execution schemas, containment, raw hashes, canonical payload hashes,
  ordered record identities, and receipt validation counts; recheck evidence
  hashes after analysis to detect concurrent mutation.  Bind every compressed
  realized-score artifact to its record receipt and evaluated result allowlist,
  and validate the exact storage-preflight receipt.

- [ ] **Step 4: Run loader tests and verify GREEN**

  Re-run Step 2 and expect all selected tests to pass.

### Task 3: Positional-only CLI and canonical output

**Files:**
- Modify: `tests/test_final_analysis.py`
- Create: `scripts/generate_final_analysis.py`
- Modify: `maskimpute_benchmark/final_analysis.py`

**Interfaces:**
- Consumes: one repository-relative or absolute round path.
- Produces: one newline-terminated canonical JSON report on stdout.

- [ ] **Step 1: Write failing CLI/signature/self-hash tests**

  Assert only `repository` and `round_dir` in the public generator signature,
  only one positional round locator in CLI help, no scientific flags, canonical
  sorted JSON, and `analysis_sha256 == canonical_sha256(report body)`.

- [ ] **Step 2: Run CLI tests and verify RED**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py -k 'cli or canonical or signature'` and expect failure.

- [ ] **Step 3: Implement the CLI and self-hashed report**

  Resolve the repository from the script location, pass the round to the public
  generator, and serialize with `allow_nan=False`, sorted keys, compact
  separators, and one trailing newline.

- [ ] **Step 4: Run CLI tests and verify GREEN**

  Re-run Step 2 and expect all selected tests to pass.

### Task 4: Verification and review

**Files:**
- Review: `maskimpute_benchmark/final_analysis.py`
- Review: `scripts/generate_final_analysis.py`
- Review: `tests/test_final_analysis.py`
- Review: both final-analysis documentation files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: reviewed commit on `codex/final-analysis`.

- [ ] **Step 1: Run focused tests**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_analysis.py tests/test_statistics.py` and require zero failures.

- [ ] **Step 2: Run regression tests around frozen-final contracts**

  Run `python -m pytest -q -p no:cacheprovider tests/test_final_runner.py -k 'final_manifest_requires or final_plan_uses or final_result_store or final_evaluation or frozen_final_cli'` and require zero failures.

- [ ] **Step 3: Run Ruff**

  Run `python -m ruff check maskimpute_benchmark/final_analysis.py scripts/generate_final_analysis.py tests/test_final_analysis.py` and require zero findings.

- [ ] **Step 4: Inspect scope and prohibited files**

  Run `git status --short`, `git diff --check`, and `git diff --stat 0a7ba61...HEAD`; verify no prohibited file is changed and no placeholder text remains.

- [ ] **Step 5: Commit**

  Stage only the five planned files and commit with `feat: generate frozen final publication analysis`.
