# Publication Integration Full Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the complete `codex/publication-integration` branch, correct
every reproducible in-scope defect test-first, and finish with evidence that the
software and Genome Biology publication infrastructure are internally
consistent.

**Architecture:** Treat the final branch as a sequence of explicit authority
boundaries from study configuration through model execution, selection,
freeze, final analysis, and manuscript synthesis. Review each boundary with
static inspection, focused negative tests, and subsystem suites, then run
whole-branch verification at the final head. Record findings and their
disposition in one tracked review report.

**Tech Stack:** Python 3.11, pytest, Ruff, NumPy, SciPy, PyTorch, JSON study
contracts, Bash/R adapter scripts, pdfLaTeX/BibTeX, Git.

## Global Constraints

- Work only in
  `/home/marcinmaleclocal/Coding/MaskedImpute/.worktrees/publication-integration`.
- Do not touch unrelated changes in the primary checkout.
- Do not perform cyber-related work.
- Do not introduce or extend hashes, checksums, fingerprints, or content
  summaries in the fair-comparator workflow or this review.
- Do not add parent-directory race hardening.
- Preserve legacy outer provenance outside the direct comparator segment.
- Do not execute the real scientific workload; use only bounded tests,
  collection, dry-run, planning, and compilation commands.
- Do not claim empirical competitiveness without completed frozen scientific
  results.
- Do not invent human declarations, licenses, repository identifiers,
  accessions, numerical results, figures, or submission approval.
- Fix critical and important findings. Fix in-scope, low-risk minor findings and
  record every intentional exclusion.
- Diagnose every failure before changing code. Add a focused failing regression
  test before every production bug fix.

---

## File responsibility map

- `maskimpute/`: current model, training, numerical, sparse-input, calibration,
  and public-result implementation.
- `masked_imputation26.py`: compatibility-facing legacy entry point.
- `maskimpute_benchmark/schema.py`, `protocol.py`, `study.py`, `config.py`,
  `datasets.py`, `sources.py`: study configuration and dataset authority.
- `maskimpute_benchmark/simulators/`: bounded simulator adapters and runtime
  asset declarations.
- `maskimpute_benchmark/methods/`: method identities, registry, and adapters.
- `maskimpute_benchmark/comparator_tuning.py`,
  `fair_comparator_execution.py`, `fair_comparator_plan.py`,
  `fair_comparator_checkpoint.py`: fair-comparator scheduling and execution.
- `maskimpute_benchmark/metrics.py`, `statistics.py`, `runner.py`,
  `evaluation_manifest.py`, `external_reference_development.py`,
  `prezero_evidence.py`: metric, inferential, and evaluation-evidence
  correctness.
- `maskimpute_benchmark/development_evaluation.py`,
  `development_scores.py`, `selection.py`, `selection_promotion.py`,
  `direct_values.py`: development evidence and direct selection authority.
- `maskimpute_benchmark/revisions.py`, `revision_commands.py`,
  `revision_evaluation.py`: revision activation and competition.
- `maskimpute_benchmark/publication_freeze.py`, `final_runner.py`,
  `final_analysis.py`, `final_null_de.py`, `trajectory_dataset.py`,
  `downstream_evaluation.py`, `downstream_evidence.py`, `scaling.py`,
  `publication_synthesis.py`: post-freeze execution and publication evidence.
- `scripts/`: executable study-control and adapter entry points.
- `study/`: machine-readable protocol and search-space contracts.
- `tests/`: unit, contract, integration, negative, hygiene, and CLI tests.
- `paper/` and `docs/genome-biology-submission-checklist.md`: Genome Biology
  manuscript sources and fail-closed submission requirements.
- `docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md`:
  tracked final audit evidence and finding dispositions.

### Task 1: Establish the branch baseline and review ledger

**Files:**
- Create:
  `docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md`
- Inspect: `.gitignore`
- Inspect: `.gitattributes`
- Inspect: `.gitmodules`
- Inspect: `pyproject.toml`

**Interfaces:**
- Consumes: approved design in
  `docs/superpowers/specs/2026-07-23-publication-integration-full-review-design.md`
- Produces: exact baseline identity, change inventory, command ledger, and
  severity/disposition table used by Tasks 2–10

- [ ] **Step 1: Confirm isolation and record exact Git identities**

Run:

```bash
pwd
git status --short --branch
git branch --show-current
git merge-base main HEAD
git rev-parse HEAD
git rev-list --count main..HEAD
```

Expected: the path ends in `.worktrees/publication-integration`, the branch is
`codex/publication-integration`, and the worktree is clean before the report is
created.

- [ ] **Step 2: Inventory every changed path without expensive rename guessing**

Run:

```bash
git -c diff.renameLimit=20000 diff --name-status main...HEAD \
  > /tmp/publication-integration-name-status.txt
git -c diff.renameLimit=20000 diff --numstat main...HEAD \
  > /tmp/publication-integration-numstat.txt
git diff --summary main...HEAD
git submodule status
git ls-files -d
git ls-files -o --exclude-standard
```

Expected: inventories complete successfully; untracked or deleted paths are
classified before any cleanup.

- [ ] **Step 3: Create the review report with fixed sections**

Use `apply_patch` to create the report with these headings:

```markdown
# Publication Integration Full Review

## Scope and constraints
## Baseline
## Changed-path coverage
## Findings
### Critical
### Important
### Minor
### Human and scientific blockers
## Corrections
## Verification ledger
## Final disposition
```

Populate `Baseline` with the exact outputs from Steps 1–2. Do not paste
environment secrets, generated scientific data, or the contents of unrelated
files.

- [ ] **Step 4: Run collection and fast static baseline checks**

Run:

```bash
python -m pytest --collect-only -q
python -m ruff check .
python -m ruff format --check .
python -m compileall -q maskimpute maskimpute_benchmark scripts tests
git diff --check main...HEAD
```

Expected: test collection completes, Ruff and compilation exit zero, and Git
reports no whitespace errors. A failure becomes the first finding and is
diagnosed under Task 8 before proceeding.

- [ ] **Step 5: Commit the baseline ledger**

Run:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: start publication integration full review"
```

Expected: exactly the review report is committed.

### Task 2: Audit model and numerical correctness

**Files:**
- Inspect/modify: `maskimpute/__init__.py`
- Inspect/modify: `maskimpute/ablations.py`
- Inspect/modify: `maskimpute/calibration.py`
- Inspect/modify: `maskimpute/config.py`
- Inspect/modify: `maskimpute/count_model.py`
- Inspect/modify: `maskimpute/impute.py`
- Inspect/modify: `maskimpute/model.py`
- Inspect/modify: `maskimpute/nb_model.py`
- Inspect/modify: `maskimpute/prezero.py`
- Inspect/modify: `maskimpute/result.py`
- Inspect/modify: `maskimpute/sparse_input.py`
- Inspect/modify: `maskimpute/structure.py`
- Inspect/modify: `maskimpute/train.py`
- Inspect/modify: `masked_imputation26.py`
- Test/modify: `tests/test_count_model.py`
- Test/modify: `tests/test_maskimpute_ablations.py`
- Test/modify: `tests/test_maskimpute_api.py`
- Test/modify: `tests/test_maskimpute_v27.py`
- Test/modify: `tests/test_maskimpute_v28.py`
- Test/modify: `tests/test_maskimpute_v29.py`
- Test/modify: `tests/test_prezero_calibration.py`
- Test/modify: `tests/test_prezero_evidence.py`

**Interfaces:**
- Consumes: public configuration and array-like inputs
- Produces: validated imputation results and model metadata used by the
  MaskImpute benchmark adapter

- [ ] **Step 1: Trace input-to-result invariants**

Inspect every listed production file and record coverage for shape validation,
sparse/dense parity, dtype/device changes, finite-value checks, library-size
zeros, mask meaning, seed propagation, train/eval transitions, and result
immutability.

Run:

```bash
rg -n "nan|inf|isfinite|sparse|seed|manual_seed|train\\(|eval\\(|no_grad|dtype|device|shape|mask|library" \
  maskimpute masked_imputation26.py
```

Expected: every match is classified as valid, a test gap, or a finding in the
review report.

- [ ] **Step 2: Run the complete focused model suite**

Run:

```bash
python -m pytest -q \
  tests/test_count_model.py \
  tests/test_maskimpute_ablations.py \
  tests/test_maskimpute_api.py \
  tests/test_maskimpute_v27.py \
  tests/test_maskimpute_v28.py \
  tests/test_maskimpute_v29.py \
  tests/test_prezero_calibration.py \
  tests/test_prezero_evidence.py
```

Expected: all tests pass. Diagnose any failure with
`superpowers:systematic-debugging`.

- [ ] **Step 3: Add negative or metamorphic tests only for demonstrated gaps**

For each material gap, first add one focused test to the owning test file listed
above. Record the complete pytest node identifier in the review report, then
run that identifier alone. The new test must fail because the identified
invariant is violated, not because of fixture setup.

- [ ] **Step 4: Correct each demonstrated model defect minimally**

Use `apply_patch` on the owning production file. Do not retune hyperparameters
or change the scientific estimand. Rerun the complete pytest node identifier
recorded in Step 3, then run:

```bash
python -m pytest -q \
  tests/test_count_model.py \
  tests/test_maskimpute_ablations.py \
  tests/test_maskimpute_api.py \
  tests/test_maskimpute_v27.py \
  tests/test_maskimpute_v28.py \
  tests/test_maskimpute_v29.py \
  tests/test_prezero_calibration.py \
  tests/test_prezero_evidence.py
```

Expected: each regression and the complete focused model suite pass.

- [ ] **Step 5: Record and commit the model audit**

Update the report with reviewed files, findings, regression nodes, and
dispositions. If code or tests changed, run:

```bash
git add maskimpute masked_imputation26.py tests \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "fix: correct reviewed model invariants"
```

If no code changed, commit only the report with:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: record model correctness audit"
```

### Task 3: Audit study schemas, datasets, and simulator contracts

**Files:**
- Inspect/modify: `maskimpute_benchmark/config.py`
- Inspect/modify: `maskimpute_benchmark/schema.py`
- Inspect/modify: `maskimpute_benchmark/protocol.py`
- Inspect/modify: `maskimpute_benchmark/study.py`
- Inspect/modify: `maskimpute_benchmark/datasets.py`
- Inspect/modify: `maskimpute_benchmark/sources.py`
- Inspect/modify: `maskimpute_benchmark/trajectory_dataset.py`
- Inspect/modify: `maskimpute_benchmark/simulators/__init__.py`
- Inspect/modify: `maskimpute_benchmark/simulators/base.py`
- Inspect/modify: `maskimpute_benchmark/simulators/native.py`
- Inspect/modify: `maskimpute_benchmark/simulators/runtime_assets.py`
- Inspect/modify: `maskimpute_benchmark/simulators/semisynthetic.py`
- Inspect/modify: `maskimpute_benchmark/simulators/sergio.py`
- Inspect/modify: `maskimpute_benchmark/simulators/sparsim.py`
- Inspect/modify: `maskimpute_benchmark/simulators/symsim.py`
- Inspect: `study/*.json`
- Test/modify: `tests/test_dataset_registry.py`
- Test/modify: `tests/test_dataset_schema.py`
- Test/modify: `tests/test_protocol.py`
- Test/modify: `tests/test_sources.py`
- Test/modify: `tests/test_study_state.py`
- Test/modify: `tests/test_trajectory_dataset.py`
- Test/modify: `tests/test_semisynthetic_adapter.py`
- Test/modify: `tests/test_sergio_adapter.py`
- Test/modify: `tests/test_simulator_contract.py`
- Test/modify: `tests/test_simulator_runtime_assets.py`
- Test/modify: `tests/test_sparsim_adapter.py`
- Test/modify: `tests/test_symsim_adapter.py`

**Interfaces:**
- Consumes: JSON study contracts and bounded fixture inputs
- Produces: validated dataset, simulator, source, and trajectory authorities
  used by method execution

- [ ] **Step 1: Validate every tracked JSON document**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("study").glob("*.json")):
    with path.open(encoding="utf-8") as handle:
        json.load(handle)
    print(path)
for path in sorted(Path("study/method-attempts").glob("*.json")):
    with path.open(encoding="utf-8") as handle:
        json.load(handle)
    print(path)
PY
```

Expected: every listed JSON file parses exactly once without scientific
execution.

- [ ] **Step 2: Inspect schema strictness and cross-file identity**

Check required keys, rejection of extra keys, bool-versus-int handling,
non-empty identifiers, uniqueness, deterministic ordering, path resolution,
complete mechanism/draw/view/seed coverage, and exact agreement between study
documents and code registries.

Run:

```bash
rg -n "from_dict|json\\.load|set\\(|sorted\\(|required|unexpected|extra|bool|int|dataset_id|seed|mechanism|draw|view" \
  maskimpute_benchmark/{config.py,schema.py,protocol.py,study.py,datasets.py,sources.py,trajectory_dataset.py} \
  maskimpute_benchmark/simulators study tests/test_dataset_registry.py \
  tests/test_dataset_schema.py tests/test_protocol.py tests/test_sources.py
```

Expected: all authority-bearing fields have an explicit parser and validation
path or are reported as findings.

- [ ] **Step 3: Run schema, dataset, and simulator suites**

Run:

```bash
python -m pytest -q \
  tests/test_dataset_registry.py \
  tests/test_dataset_schema.py \
  tests/test_protocol.py \
  tests/test_sources.py \
  tests/test_study_state.py \
  tests/test_trajectory_dataset.py \
  tests/test_semisynthetic_adapter.py \
  tests/test_sergio_adapter.py \
  tests/test_simulator_contract.py \
  tests/test_simulator_runtime_assets.py \
  tests/test_sparsim_adapter.py \
  tests/test_symsim_adapter.py
```

Expected: all tests pass.

- [ ] **Step 4: Fix demonstrated contract defects test-first**

For each finding, add a mutation test that changes one field or population
member and proves fail-closed behavior. Run the exact new node to RED, patch the
owning parser or validator with `apply_patch`, then rerun the node and the suite
from Step 3.

Expected: the mutation fails before the fix and all focused tests pass after
the fix.

- [ ] **Step 5: Record and commit the schema audit**

Stage only the exact reviewed production/test files changed plus the report and
commit:

```bash
git add maskimpute_benchmark/config.py \
  maskimpute_benchmark/schema.py \
  maskimpute_benchmark/protocol.py \
  maskimpute_benchmark/study.py \
  maskimpute_benchmark/datasets.py \
  maskimpute_benchmark/sources.py \
  maskimpute_benchmark/trajectory_dataset.py \
  maskimpute_benchmark/simulators \
  study tests/test_dataset_registry.py tests/test_dataset_schema.py \
  tests/test_protocol.py tests/test_sources.py tests/test_study_state.py \
  tests/test_trajectory_dataset.py tests/test_semisynthetic_adapter.py \
  tests/test_sergio_adapter.py tests/test_simulator_contract.py \
  tests/test_simulator_runtime_assets.py tests/test_sparsim_adapter.py \
  tests/test_symsim_adapter.py \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git diff --check
git commit -m "fix: close reviewed study contract gaps"
```

If no defects were found, commit the report update as:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: record study contract audit"
```

### Task 4: Audit method adapters and fair-comparator execution

**Files:**
- Inspect/modify: `maskimpute_benchmark/methods/*.py`
- Inspect/modify: `maskimpute_benchmark/comparator_tuning.py`
- Inspect/modify: `maskimpute_benchmark/fair_comparator_plan.py`
- Inspect/modify: `maskimpute_benchmark/fair_comparator_execution.py`
- Inspect/modify: `maskimpute_benchmark/fair_comparator_checkpoint.py`
- Inspect/modify: `maskimpute_benchmark/operational_environment.py`
- Inspect/modify: `maskimpute_benchmark/runtime_environments.py`
- Test/modify: `tests/test_core_method_adapters.py`
- Test/modify: `tests/test_maskimpute_method_adapter.py`
- Test/modify: `tests/test_method_registry.py`
- Test/modify: `tests/test_priority_method_adapters.py`
- Test/modify: `tests/test_required_legacy_method_adapters.py`
- Test/modify: `tests/test_sctsi_method_adapter.py`
- Test/modify: `tests/test_comparator_tuning.py`
- Test/modify: `tests/test_fair_comparator_plan.py`
- Test/modify: `tests/test_fair_comparator_execution.py`
- Test/modify: `tests/test_fair_comparator_checkpoint.py`
- Test/modify: `tests/test_runtime_environments.py`

**Interfaces:**
- Consumes: validated dataset authorities, method registry entries, smoke plans,
  and bounded fixture environments
- Produces: complete scheduled/terminal comparator populations and direct
  execution evidence

- [ ] **Step 1: Cross-check all method identities and binding fields**

Run:

```bash
rg -n "method_id|configuration_id|runtime|binding|selected|unavailable|timeout|resource|failed|scheduled|terminal" \
  maskimpute_benchmark/methods \
  maskimpute_benchmark/comparator_tuning.py \
  maskimpute_benchmark/fair_comparator_{plan,execution,checkpoint}.py \
  tests/test_method_registry.py tests/test_comparator_tuning.py \
  tests/test_fair_comparator_plan.py tests/test_fair_comparator_execution.py
```

Expected: every method is canonical, every selected configuration has complete
bindings, and every scheduled item has exactly one reason-coded terminal state.

- [ ] **Step 2: Inspect adapter semantic parity**

For every adapter, compare declared input domain, preprocessing, seed handling,
output orientation, observed-value policy, runtime invocation, and failure
translation against `methods/base.py` and `methods/registry.py`. Do not install
dependencies or run the real comparator workload.

Expected: differences are intentional and tested or recorded as findings.

- [ ] **Step 3: Run all adapter and fair-comparator tests**

Run:

```bash
python -m pytest -q \
  tests/test_core_method_adapters.py \
  tests/test_maskimpute_method_adapter.py \
  tests/test_method_registry.py \
  tests/test_priority_method_adapters.py \
  tests/test_required_legacy_method_adapters.py \
  tests/test_sctsi_method_adapter.py \
  tests/test_comparator_tuning.py \
  tests/test_fair_comparator_plan.py \
  tests/test_fair_comparator_execution.py \
  tests/test_fair_comparator_checkpoint.py \
  tests/test_runtime_environments.py
```

Expected: all tests pass without performing the scientific competition.

- [ ] **Step 4: Fix demonstrated adapter/execution defects test-first**

Add a focused production-boundary test for each finding, verify RED, apply the
minimal correction, and rerun the exact node plus Step 3. Do not add hashes,
checksums, fingerprints, content summaries, or cyber-adjacent hardening.

- [ ] **Step 5: Record and commit the adapter/execution audit**

Update the report. Stage only exact changed files and commit:

```bash
git add maskimpute_benchmark/methods \
  maskimpute_benchmark/comparator_tuning.py \
  maskimpute_benchmark/fair_comparator_plan.py \
  maskimpute_benchmark/fair_comparator_execution.py \
  maskimpute_benchmark/fair_comparator_checkpoint.py \
  maskimpute_benchmark/operational_environment.py \
  maskimpute_benchmark/runtime_environments.py \
  tests/test_core_method_adapters.py \
  tests/test_maskimpute_method_adapter.py tests/test_method_registry.py \
  tests/test_priority_method_adapters.py \
  tests/test_required_legacy_method_adapters.py \
  tests/test_sctsi_method_adapter.py tests/test_comparator_tuning.py \
  tests/test_fair_comparator_plan.py \
  tests/test_fair_comparator_execution.py \
  tests/test_fair_comparator_checkpoint.py \
  tests/test_runtime_environments.py \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git diff --check
git commit -m "fix: correct reviewed comparator execution defects"
```

If no defect exists, commit only the report:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: record comparator execution audit"
```

### Task 5: Audit metrics, statistics, and evaluation orchestration

**Files:**
- Inspect/modify: `maskimpute_benchmark/__init__.py`
- Inspect/modify: `maskimpute_benchmark/metrics.py`
- Inspect/modify: `maskimpute_benchmark/statistics.py`
- Inspect/modify: `maskimpute_benchmark/runner.py`
- Inspect/modify: `maskimpute_benchmark/evaluation_manifest.py`
- Inspect/modify: `maskimpute_benchmark/external_reference_development.py`
- Inspect/modify: `maskimpute_benchmark/prezero_evidence.py`
- Test/modify: `tests/test_metrics.py`
- Test/modify: `tests/test_statistics.py`
- Test/modify: `tests/test_benchmark_runner.py`
- Test/modify: `tests/test_external_reference_development.py`
- Test/modify: `tests/test_prezero_evidence.py`

**Interfaces:**
- Consumes: validated method outputs, truth-role authorities, dataset bindings,
  and reason-coded execution outcomes
- Produces: truth-isolated metrics, biological-draw-level inference, evaluation
  rows, calibration evidence, and validated development evidence

- [ ] **Step 1: Audit metric domains and unavailable states**

Inspect matrix validation, truth-kind permissions, subsets, denominators,
probability bounds, degenerate correlations, constant features, unavailable
reasons, metric direction, and finite output handling.

Run:

```bash
rg -n "MetricValue|truth_kind|denominator|n=|no_entries|unavailable|reason|finite|probability|correlation|marker|direction" \
  maskimpute_benchmark/metrics.py tests/test_metrics.py
```

Expected: every metric either returns a finite value with the exact denominator
or a reason-coded unavailable value; no input is silently sanitized onto a
different estimand.

- [ ] **Step 2: Audit inferential independence and multiplicity**

Inspect pairing at biological-draw, technical-view, dataset, method, seed, and
metric levels; duplicate/conflicting rows; bootstrap resampling units; interval
construction; probability of improvement; win/tie/loss counts; variance
components; and Holm adjustment.

Run:

```bash
rg -n "biological|technical_view|dataset_id|model_seed|duplicate|conflict|bootstrap|interval|probability|wins|ties|losses|variance|holm|adjust" \
  maskimpute_benchmark/statistics.py tests/test_statistics.py
```

Expected: model seeds and technical views remain repeated measures, biological
draws remain the independent unit, and each multiplicity family is explicit.
Legacy outer provenance fields are not redesigned in this task.

- [ ] **Step 3: Audit runner and evaluation evidence boundaries**

Check plan completeness, adapter outcome translation, checkpoint replay,
calibration-fold separation, external-reference isolation, pre-zero evidence
semantics, manifest parser strictness, and deterministic evaluation rows. Do not
extend legacy digest or filesystem-hardening mechanisms.

Run:

```bash
rg -n "CompetitionPlan|RunPlanEntry|AdapterOutcome|status|checkpoint|replay|calibration|external|prezero|evaluation|manifest|from_dict|to_dict" \
  maskimpute_benchmark/runner.py \
  maskimpute_benchmark/evaluation_manifest.py \
  maskimpute_benchmark/external_reference_development.py \
  maskimpute_benchmark/prezero_evidence.py \
  tests/test_benchmark_runner.py \
  tests/test_external_reference_development.py \
  tests/test_prezero_evidence.py
```

Expected: incomplete, duplicate, conflicting, or cross-role evidence is
rejected before it can enter selection.

- [ ] **Step 4: Run the complete evaluation suite**

Run:

```bash
python -m pytest -q \
  tests/test_metrics.py \
  tests/test_statistics.py \
  tests/test_benchmark_runner.py \
  tests/test_external_reference_development.py \
  tests/test_prezero_evidence.py
```

Expected: all tests pass.

- [ ] **Step 5: Fix demonstrated evaluation defects test-first**

For each material finding, add a focused failing test in the owning listed test
file, run its complete node identifier to RED, patch the smallest owning
production function, then rerun the node and Step 4. Do not change the
scientific estimand, bootstrap design, multiplicity family, or legacy outer
provenance unless the approved design explicitly requires it.

- [ ] **Step 6: Record and commit the evaluation audit**

Update the report, stage only the exact evaluation files and tests, and commit:

```bash
git add maskimpute_benchmark/__init__.py \
  maskimpute_benchmark/metrics.py \
  maskimpute_benchmark/statistics.py \
  maskimpute_benchmark/runner.py \
  maskimpute_benchmark/evaluation_manifest.py \
  maskimpute_benchmark/external_reference_development.py \
  maskimpute_benchmark/prezero_evidence.py \
  tests/test_metrics.py tests/test_statistics.py \
  tests/test_benchmark_runner.py \
  tests/test_external_reference_development.py \
  tests/test_prezero_evidence.py \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git diff --check
git commit -m "fix: correct reviewed evaluation defects"
```

If no defect exists, commit only the report:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: record evaluation audit"
```

### Task 6: Audit selection, revision, freeze, and post-freeze authority

**Files:**
- Inspect/modify: `maskimpute_benchmark/development_evaluation.py`
- Inspect/modify: `maskimpute_benchmark/development_scores.py`
- Inspect/modify: `maskimpute_benchmark/direct_values.py`
- Inspect/modify: `maskimpute_benchmark/selection.py`
- Inspect/modify: `maskimpute_benchmark/selection_promotion.py`
- Inspect/modify: `maskimpute_benchmark/revisions.py`
- Inspect/modify: `maskimpute_benchmark/revision_commands.py`
- Inspect/modify: `maskimpute_benchmark/revision_evaluation.py`
- Inspect/modify: `maskimpute_benchmark/publication_freeze.py`
- Inspect/modify: `maskimpute_benchmark/final_runner.py`
- Inspect/modify: `maskimpute_benchmark/final_analysis.py`
- Inspect/modify: `maskimpute_benchmark/final_null_de.py`
- Inspect/modify: `maskimpute_benchmark/downstream_evaluation.py`
- Inspect/modify: `maskimpute_benchmark/downstream_evidence.py`
- Inspect/modify: `maskimpute_benchmark/scaling.py`
- Inspect/modify: `maskimpute_benchmark/publication_synthesis.py`
- Test/modify: `tests/test_development_evaluation.py`
- Test/modify: `tests/test_development_scores.py`
- Test/modify: `tests/test_candidate_selection.py`
- Test/modify: `tests/test_selection_authority.py`
- Test/modify: `tests/test_selection_promotion.py`
- Test/modify: `tests/test_revision_authority.py`
- Test/modify: `tests/test_revision_evaluation.py`
- Test/modify: `tests/test_freeze_publication_round.py`
- Test/modify: `tests/test_final_runner.py`
- Test/modify: `tests/test_final_analysis.py`
- Test/modify: `tests/test_final_null_de.py`
- Test/modify: `tests/test_downstream_evaluation.py`
- Test/modify: `tests/test_downstream_evidence.py`
- Test/modify: `tests/test_scaling_panel.py`
- Test/modify: `tests/test_publication_synthesis.py`

**Interfaces:**
- Consumes: complete development execution evidence and selected direct method
  bindings
- Produces: frozen authorities and exact claim permissions for final,
  trajectory, downstream, scaling, and manuscript synthesis

- [ ] **Step 1: Trace direct authority end to end**

Starting from persisted development scores, follow selected and nonexecution
authorities through promotion, revisions, freeze, final plans, trajectory,
downstream, scaling, and synthesis. Record every serialization and reload
boundary.

Run:

```bash
rg -n "FrozenPlanMethodAuthority|selected_method|selected_methods|nonexecution|scheduled|numerical|claim|permission|freeze|from_dict|to_dict|json" \
  maskimpute_benchmark/{development_evaluation.py,development_scores.py,direct_values.py,selection.py,selection_promotion.py,revisions.py,revision_commands.py,revision_evaluation.py,publication_freeze.py,final_runner.py,final_analysis.py,final_null_de.py,downstream_evaluation.py,downstream_evidence.py,scaling.py,publication_synthesis.py}
```

Expected: complete canonical values cross each boundary without live-registry
substitution or permissive reconstruction.

- [ ] **Step 2: Check leakage and population-accounting invariants**

Confirm development-only selection cannot access final performance, downstream
endpoints, final data, or post-freeze results. Check exact scheduled versus
numerical populations, reason-coded nonexecution denominator, one-use final
authority, revision limits, and claim suppression for unavailable estimands.

Expected: every invariant is enforced in production and challenged by at least
one negative test.

- [ ] **Step 3: Run all authority and post-freeze suites**

Run:

```bash
python -m pytest -q \
  tests/test_development_evaluation.py \
  tests/test_development_scores.py \
  tests/test_candidate_selection.py \
  tests/test_selection_authority.py \
  tests/test_selection_promotion.py \
  tests/test_revision_authority.py \
  tests/test_revision_evaluation.py \
  tests/test_freeze_publication_round.py \
  tests/test_final_runner.py \
  tests/test_final_analysis.py \
  tests/test_final_null_de.py \
  tests/test_downstream_evaluation.py \
  tests/test_downstream_evidence.py \
  tests/test_scaling_panel.py \
  tests/test_publication_synthesis.py
```

Expected: all tests pass.

- [ ] **Step 4: Add mutation tests for any uncovered authority gap**

Mutate exactly one identity, type, order, missing population member, duplicate
member, status, or claim permission per new test. Verify RED, patch the owning
boundary minimally, then rerun the node and Step 3.

- [ ] **Step 5: Record and commit the authority audit**

Update the report and commit exact changed files:

```bash
git add maskimpute_benchmark/development_evaluation.py \
  maskimpute_benchmark/development_scores.py \
  maskimpute_benchmark/direct_values.py \
  maskimpute_benchmark/selection.py \
  maskimpute_benchmark/selection_promotion.py \
  maskimpute_benchmark/revisions.py \
  maskimpute_benchmark/revision_commands.py \
  maskimpute_benchmark/revision_evaluation.py \
  maskimpute_benchmark/publication_freeze.py \
  maskimpute_benchmark/final_runner.py \
  maskimpute_benchmark/final_analysis.py \
  maskimpute_benchmark/final_null_de.py \
  maskimpute_benchmark/downstream_evaluation.py \
  maskimpute_benchmark/downstream_evidence.py \
  maskimpute_benchmark/scaling.py \
  maskimpute_benchmark/publication_synthesis.py \
  tests/test_development_evaluation.py tests/test_development_scores.py \
  tests/test_candidate_selection.py tests/test_selection_authority.py \
  tests/test_selection_promotion.py tests/test_revision_authority.py \
  tests/test_revision_evaluation.py tests/test_freeze_publication_round.py \
  tests/test_final_runner.py tests/test_final_analysis.py \
  tests/test_final_null_de.py tests/test_downstream_evaluation.py \
  tests/test_downstream_evidence.py tests/test_scaling_panel.py \
  tests/test_publication_synthesis.py \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git diff --check
git commit -m "fix: close reviewed publication authority gaps"
```

If no defect exists, commit only the report:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: record publication authority audit"
```

### Task 7: Audit CLIs, study documents, and branch migration integrity

**Files:**
- Inspect/modify: `scripts/*.py`
- Inspect/modify: `scripts/*.sh`
- Inspect/modify: `scripts/simulators/*`
- Inspect: `study/*.json`
- Inspect: `study/method-attempts/*.json`
- Inspect/modify: `docs/development-selection-workflow.md`
- Inspect: `historical/`
- Inspect: `.gitmodules`
- Test/modify: `tests/test_benchmark_runner.py`
- Test/modify: `tests/test_external_reference_development.py`
- Test/modify: `tests/test_obsolete_terms.py`
- Test/modify: `tests/test_publication_repository_hygiene.py`

**Interfaces:**
- Consumes: production library APIs and study JSON
- Produces: operator-facing commands whose output paths and exit statuses match
  the library contracts

- [ ] **Step 1: Map every active script to an importable production API**

Run:

```bash
for path in scripts/*.py scripts/simulators/*.py; do
  python "$path" --help >/dev/null
done
bash -n scripts/*.sh scripts/simulators/*.sh
```

Expected: all Python CLIs expose help without scientific execution and all shell
scripts parse.

- [ ] **Step 2: Inspect CLI state isolation and failure behavior**

Check argument validation, explicit output roots, no import-time execution,
nonzero failures, bounded overwrite behavior, environment restoration in
tests, dry-run/planning boundaries, and agreement with study documents.

Run:

```bash
rg -n "argparse|ArgumentParser|main\\(|sys\\.exit|os\\.environ|subprocess|output|overwrite|dry.run|plan" \
  scripts maskimpute_benchmark tests
```

Expected: global or environment state is owned and restored by the caller or
test; errors cannot silently become successful scientific evidence.

- [ ] **Step 3: Audit deleted and migrated paths for active references**

Generate a deleted-path basename list and search active tracked sources for
stale references. Exclude `historical/` from the consumer side because it is an
archive.

Run:

```bash
git diff --diff-filter=D --name-only main...HEAD \
  > /tmp/publication-integration-deleted-paths.txt
rg -n "supplementary_anonymous|synthetic_datasets|archive/|configs/grid|\\.venv_scvi|AutoClass|MAGIC|SAVER|rds2py|splatter" \
  --glob '!historical/**' --glob '!.git/**' .
```

Expected: any active reference is either an intentional historical pointer or a
finding. No deletion is restored merely because it existed on `main`.

- [ ] **Step 4: Run CLI, external-reference, obsolete-term, and hygiene tests**

Run:

```bash
python -m pytest -q \
  tests/test_benchmark_runner.py \
  tests/test_external_reference_development.py \
  tests/test_obsolete_terms.py \
  tests/test_publication_repository_hygiene.py
```

Expected: all tests pass.

- [ ] **Step 5: Fix demonstrated CLI or migration defects test-first**

For production defects, write a focused failing CLI or hygiene test, verify RED,
patch minimally, and rerun Step 4. Documentation reference corrections may be
patched directly and verified by the exact `rg` query that exposed them.

- [ ] **Step 6: Record and commit the CLI/migration audit**

Update the report and commit exact changed paths:

```bash
git add scripts study docs/development-selection-workflow.md \
  tests/test_benchmark_runner.py \
  tests/test_external_reference_development.py tests/test_obsolete_terms.py \
  tests/test_publication_repository_hygiene.py \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git diff --check
git commit -m "fix: correct reviewed CLI and migration defects"
```

If no defect exists, commit only the report:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: record CLI and migration audit"
```

### Task 8: Audit and correct the Genome Biology package

**Files:**
- Inspect/modify: `paper/manuscript.tex`
- Inspect/modify: `paper/references.bib`
- Inspect/modify: `paper/README.md`
- Inspect/modify: `paper/submission_checklist.md`
- Inspect: `paper/sn-jnl.cls`
- Inspect: `paper/sn-vancouver-num.bst`
- Inspect/modify: `docs/genome-biology-submission-checklist.md`
- Test/modify: `tests/test_publication_repository_hygiene.py`

**Interfaces:**
- Consumes: only claims permitted by frozen executable evidence
- Produces: a compilable sealed-evidence Methodology draft and fail-closed
  submission checklist

- [ ] **Step 1: Remove active digest-verification requirements**

Inspect both checklists for active checksum, hash, fingerprint, or content
summary requirements. Remove those requirements while preserving ordinary
template provenance, licensing caveats, and scientific evidence checks.

Run before and after:

```bash
rg -n -i "checksum|sha-?256|hash|fingerprint|content summary" \
  paper/submission_checklist.md docs/genome-biology-submission-checklist.md
```

Expected before: existing template digest requirements may be found. Expected
after: no active digest-verification requirement remains.

- [ ] **Step 2: Check manuscript evidence and venue structure**

Inspect title page, abstract length, keywords, section order, Methods placement,
all seven declarations, AI-use statement placement, citations, unavailable
results, placeholders, and claim language.

Run:

```bash
rg -n -i "PENDING|TBD|TODO|placeholder|outperform|superior|state.of.the.art|author|affiliation|funding|ethics|license|doi|accession" \
  paper/manuscript.tex paper/submission_checklist.md \
  docs/genome-biology-submission-checklist.md
```

Expected: scientific and human blockers remain clearly marked; no unavailable
result is written as a finding.

- [ ] **Step 3: Build the manuscript from a clean output state**

Run:

```bash
cd paper
rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.log \
  manuscript.out manuscript.pdf
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
cd ..
```

Expected: all commands exit zero and produce `paper/manuscript.pdf`.

- [ ] **Step 4: Inspect the build log and rendered structure**

Run:

```bash
rg -n "Undefined|Citation.*undefined|Reference.*undefined|There were undefined|Emergency stop|Fatal error" \
  paper/manuscript.log paper/manuscript.blg
pdfinfo paper/manuscript.pdf
pdftotext paper/manuscript.pdf /tmp/publication-integration-manuscript.txt
```

Expected: no unresolved citations/references or fatal errors; PDF metadata
reports a nonzero page count.

- [ ] **Step 5: Correct evidence-backed manuscript defects**

Use `apply_patch` for exact factual, structural, citation, or checklist
inconsistencies. Do not fill human or scientific placeholders. Rebuild with
Steps 3–4 after every correction group.

- [ ] **Step 6: Clean products, run hygiene, record, and commit**

Run:

```bash
rm -f paper/manuscript.aux paper/manuscript.bbl paper/manuscript.blg \
  paper/manuscript.log paper/manuscript.out paper/manuscript.pdf
python -m pytest -q tests/test_publication_repository_hygiene.py
git status --short
git diff --check
```

Update the report and commit manuscript/checklist changes plus the report:

```bash
git add paper docs/genome-biology-submission-checklist.md \
  tests/test_publication_repository_hygiene.py \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: correct reviewed Genome Biology package"
```

### Task 9: Resolve cross-cutting failures and test-order defects

**Files:**
- Modify: exact production and test files already enumerated in Tasks 2–7
- Modify:
  `docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md`

**Interfaces:**
- Consumes: failures and findings produced by Tasks 1–8
- Produces: minimal regression-protected corrections with no unresolved
  critical or important finding

- [ ] **Step 1: Run the combined focused suite in deterministic order**

Run all test files named in Tasks 2–8 in one pytest process:

```bash
python -m pytest -q \
  tests/test_count_model.py \
  tests/test_maskimpute_ablations.py \
  tests/test_maskimpute_api.py \
  tests/test_maskimpute_v27.py \
  tests/test_maskimpute_v28.py \
  tests/test_maskimpute_v29.py \
  tests/test_prezero_calibration.py \
  tests/test_prezero_evidence.py \
  tests/test_dataset_registry.py \
  tests/test_dataset_schema.py \
  tests/test_protocol.py \
  tests/test_sources.py \
  tests/test_study_state.py \
  tests/test_trajectory_dataset.py \
  tests/test_semisynthetic_adapter.py \
  tests/test_sergio_adapter.py \
  tests/test_simulator_contract.py \
  tests/test_simulator_runtime_assets.py \
  tests/test_sparsim_adapter.py \
  tests/test_symsim_adapter.py \
  tests/test_core_method_adapters.py \
  tests/test_maskimpute_method_adapter.py \
  tests/test_method_registry.py \
  tests/test_priority_method_adapters.py \
  tests/test_required_legacy_method_adapters.py \
  tests/test_sctsi_method_adapter.py \
  tests/test_comparator_tuning.py \
  tests/test_fair_comparator_plan.py \
  tests/test_fair_comparator_execution.py \
  tests/test_fair_comparator_checkpoint.py \
  tests/test_runtime_environments.py \
  tests/test_metrics.py \
  tests/test_statistics.py \
  tests/test_development_evaluation.py \
  tests/test_development_scores.py \
  tests/test_candidate_selection.py \
  tests/test_selection_authority.py \
  tests/test_selection_promotion.py \
  tests/test_revision_authority.py \
  tests/test_revision_evaluation.py \
  tests/test_freeze_publication_round.py \
  tests/test_final_runner.py \
  tests/test_final_analysis.py \
  tests/test_final_null_de.py \
  tests/test_downstream_evaluation.py \
  tests/test_downstream_evidence.py \
  tests/test_scaling_panel.py \
  tests/test_publication_synthesis.py \
  tests/test_benchmark_runner.py \
  tests/test_external_reference_development.py \
  tests/test_obsolete_terms.py \
  tests/test_publication_repository_hygiene.py
```

Expected: all focused tests pass in a shared process.

- [ ] **Step 2: Diagnose every unexpected failure before patching**

For each failure, use `superpowers:systematic-debugging` to reproduce the
smallest test ordering that fails. Record root cause, not merely the observed
exception.

- [ ] **Step 3: Add the exact failing order or invariant as a regression**

When a failure is due to leaked process state, make the owner set and restore
that state. When it is a production invariant, add a single negative or
metamorphic test. Verify the regression is RED before the production fix.

- [ ] **Step 4: Apply minimal fixes and rerun adjacency**

Patch with `apply_patch`; rerun the exact regression, its file, and the combined
suite from Step 1.

- [ ] **Step 5: Close finding dispositions and commit**

The report must give every finding one of: `fixed`, `not reproducible with
evidence`, `human/scientific blocker`, or `minor excluded by standing scope`.
No critical or important finding may remain open.

Run:

```bash
git add maskimpute maskimpute_benchmark scripts study tests paper \
  docs/development-selection-workflow.md \
  docs/genome-biology-submission-checklist.md \
  docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git diff --check
git status --short
git commit -m "fix: resolve cross-cutting review findings"
```

If only the report changed, use:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: close cross-cutting review findings"
```

### Task 10: Independent review and exact-head verification

**Files:**
- Modify:
  `docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md`
- Inspect: every path changed since the review baseline

**Interfaces:**
- Consumes: completed corrections and closed finding ledger from Tasks 1–9
- Produces: exact-head verification evidence and final review disposition

- [ ] **Step 1: Invoke independent code review**

Use `superpowers:requesting-code-review` against the review baseline through the
current head. The reviewer receives the approved design, this plan, standing
constraints, complete diff, finding ledger, and focused verification evidence.

Expected: findings are severity-ranked with file/line evidence. Diagnose and
fix any critical or important finding under Task 9, then request re-review.

- [ ] **Step 2: Run candidate-head static verification**

Run:

```bash
git rev-parse HEAD
python -m ruff check .
python -m ruff format --check .
python -m compileall -q maskimpute maskimpute_benchmark scripts tests
git diff --check main...HEAD
python -m pytest -q tests/test_publication_repository_hygiene.py
```

Expected: every command exits zero.

- [ ] **Step 3: Run the complete candidate-head test suite**

Run:

```bash
python -m pytest -q
```

Expected: exit zero with no unreported test exclusions. Record the exact passed,
skipped, failed, and elapsed summary in the report.

- [ ] **Step 4: Rebuild the manuscript at the candidate head**

Run:

```bash
cd paper
rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.log \
  manuscript.out manuscript.pdf
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
rg -n "Undefined|Citation.*undefined|Reference.*undefined|There were undefined|Emergency stop|Fatal error" \
  manuscript.log manuscript.blg
pdfinfo manuscript.pdf
cd ..
```

Expected: build succeeds, the unresolved-reference scan is empty, and the PDF
has a nonzero page count.

- [ ] **Step 5: Clean products, finalize the report, and commit**

Run:

```bash
rm -f paper/manuscript.aux paper/manuscript.bbl paper/manuscript.blg \
  paper/manuscript.log paper/manuscript.out paper/manuscript.pdf
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type d -name .pytest_cache -prune -exec rm -rf {} +
git status --short
git ls-files -o --exclude-standard
```

Expected: only the final review-report update is present before its commit; no
generated scientific or manuscript product is tracked or untracked.

State the exact head, review range, all verification outputs, independent review
disposition, remaining minor exclusions, and human/scientific blockers. Explicitly
state that infrastructure correctness does not establish empirical
competitiveness or submission readiness.

Run:

```bash
git add docs/superpowers/reviews/2026-07-23-publication-integration-full-review.md
git commit -m "docs: finalize publication integration full review"
git status --short --branch
```

Expected: the report-only finalization commit succeeds.

- [ ] **Step 6: Rerun all verification at the exact final head**

The report commit changes the tested tree, so run the complete gates again:

```bash
git rev-parse HEAD
python -m ruff check .
python -m ruff format --check .
python -m compileall -q maskimpute maskimpute_benchmark scripts tests
git diff --check main...HEAD
python -m pytest -q
cd paper
rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.log \
  manuscript.out manuscript.pdf
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
bibtex manuscript
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
rg -n "Undefined|Citation.*undefined|Reference.*undefined|There were undefined|Emergency stop|Fatal error" \
  manuscript.log manuscript.blg
pdfinfo manuscript.pdf
cd ..
```

Expected: every command exits zero at the exact final commit. Preserve this
output for the final handoff; do not alter the committed report afterward.

- [ ] **Step 7: Remove verification products and prove final cleanliness**

Run:

```bash
rm -f paper/manuscript.aux paper/manuscript.bbl paper/manuscript.blg \
  paper/manuscript.log paper/manuscript.out paper/manuscript.pdf
find . -type d -name __pycache__ -prune -exec rm -rf {} +
find . -type d -name .pytest_cache -prune -exec rm -rf {} +
git status --short --branch
git ls-files -o --exclude-standard
```

Expected: the final worktree is clean on `codex/publication-integration`.
