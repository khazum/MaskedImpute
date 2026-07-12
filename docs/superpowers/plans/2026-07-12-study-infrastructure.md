# Publication Study Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-resistant protocol, dataset contract, metric implementation, and hierarchical inference used by every later experiment.

**Architecture:** A small `maskimpute_benchmark` package owns canonical protocol parsing, study-round state, truth-bearing dataset validation, truth-stripped inference views, metrics, and blocked inference. Runtime state lives under ignored `artifacts/study/`; tracked source and protocol files remain immutable inputs to a frozen round.

**Tech Stack:** Python 3.10+, NumPy, SciPy, AnnData, pytest, standard-library JSON/CSV/hashlib/subprocess.

## Global Constraints

- Existing Splatter tune/test artifacts are development-only and must never be labeled final.
- Final execution requires a clean repository at the exact frozen Git commit and matching protocol/config SHA-256 hashes.
- Experimental units are independent dataset draws; model seeds are nested within draws.
- Production names are `pre_dropout_zero` and `p_pre_zero`; legacy names may occur only in migration tests and historical documentation.
- Imputation processes receive observed counts and non-evaluative covariates only; truth layers, labels, markers, and pseudotime are stripped before method execution.
- Every metric row has either a finite value or an explicit reason code.

---

### Task 1: Package and canonical protocol

**Files:**
- Create: `pyproject.toml`
- Create: `maskimpute_benchmark/__init__.py`
- Create: `maskimpute_benchmark/protocol.py`
- Create: `study/protocol.json`
- Test: `tests/test_protocol.py`

**Interfaces:**
- Produces: `Protocol`, `load_protocol(path: Path) -> Protocol`, `canonical_sha256(value: object) -> str`, and `file_sha256(path: Path) -> str`.

- [ ] **Step 1: Write failing protocol tests**

```python
from pathlib import Path
import json
import pytest
from maskimpute_benchmark.protocol import canonical_sha256, load_protocol

def test_canonical_hash_ignores_mapping_order():
    assert canonical_sha256({"b": 2, "a": 1}) == canonical_sha256({"a": 1, "b": 2})

def test_protocol_declares_four_non_splatter_mechanisms():
    protocol = load_protocol(Path("study/protocol.json"))
    assert protocol.mechanisms == ("scdesign3", "symsim", "scmultisim", "semisynthetic")
    assert protocol.final_draws_per_condition == 8
    assert protocol.final_model_seeds == 3

def test_protocol_rejects_splatter_as_final(tmp_path):
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps({"schema_version": 1, "final": {"mechanisms": ["splatter"]}}))
    with pytest.raises(ValueError, match="Splatter is development-only"):
        load_protocol(path)
```

- [ ] **Step 2: Verify the tests fail because the package is absent**

Run: `python -m pytest tests/test_protocol.py -q`

Expected: collection fails with `ModuleNotFoundError: maskimpute_benchmark`.

- [ ] **Step 3: Add package metadata and test scoping**

Create `pyproject.toml` with setuptools package discovery, Python `>=3.10`, runtime dependencies `numpy>=2,<3`, `scipy>=1.13,<2`, `anndata>=0.11,<0.12`, and optional `test = ["pytest>=8,<9"]`. Set pytest `testpaths = ["tests"]` so vendored `rds2py` tests are not collected.

- [ ] **Step 4: Implement canonical parsing**

Use frozen dataclasses. `load_protocol` must require schema version 1, the four final mechanisms in the specified order, positive draw/seed counts, disjoint development/final namespaces, and no Splatter final mechanism. Canonical hashing is SHA-256 over UTF-8 JSON serialized with `sort_keys=True`, `separators=(",", ":")`, and `allow_nan=False`.

The tracked JSON must contain:

```json
{
  "schema_version": 1,
  "legacy_data_role": "development_only",
  "development": {"namespace": "dev", "draws_per_condition": 3, "cells": 1000, "genes": 1000},
  "final": {
    "namespace": "final",
    "mechanisms": ["scdesign3", "symsim", "scmultisim", "semisynthetic"],
    "draws_per_condition": 8,
    "model_seeds": 3,
    "cells": 2000,
    "genes": 2000
  },
  "primary_metrics": ["mse", "mse_dropout", "mse_pre_dropout_zero", "gnrmse", "corr_err"],
  "final_timeout_seconds": 21600,
  "max_rss_gib": 48,
  "max_gpu_gib": 14
}
```

- [ ] **Step 5: Run tests and default collection**

Run: `python -m pytest tests/test_protocol.py -q && python -m pytest --collect-only -q`

Expected: protocol tests pass and default collection lists only `tests/`, not `rds2py/tests/`.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml maskimpute_benchmark study/protocol.json tests/test_protocol.py
git commit -m "feat: define publication study protocol"
```

### Task 2: Frozen-round state machine

**Files:**
- Create: `maskimpute_benchmark/study.py`
- Create: `scripts/studyctl.py`
- Modify: `.gitignore`
- Test: `tests/test_study_state.py`

**Interfaces:**
- Consumes: `load_protocol`, `canonical_sha256`, and `file_sha256` from Task 1.
- Produces: `freeze_round(repo, round_dir, config_path, protocol_path)`, `materialize_final(round_dir, seed_count)`, `assert_final_runnable(repo, round_dir)`, `record_final_evaluation(round_dir, result_manifest)`, and `supersede_round(round_dir, reason)`.

- [ ] **Step 1: Write failing state-transition tests**

Create a real temporary Git repository in the fixture, commit a config and protocol, and assert:

```python
def test_final_cannot_materialize_before_freeze(clean_repo):
    with pytest.raises(StudyStateError, match="must be frozen"):
        materialize_final(clean_repo / "artifacts/study/round-001", seed_count=4)

def test_dirty_or_changed_commit_cannot_run_final(clean_repo):
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
    (clean_repo / "tracked.py").write_text("changed\n")
    with pytest.raises(StudyStateError, match="clean frozen commit"):
        assert_final_runnable(clean_repo, round_dir)

def test_evaluation_receipt_is_one_use(clean_repo):
    round_dir = freeze_fixture(clean_repo)
    materialize_final(round_dir, seed_count=4)
    record_final_evaluation(round_dir, {"results_sha256": "a" * 64})
    with pytest.raises(StudyStateError, match="already evaluated"):
        assert_final_runnable(clean_repo, round_dir)
```

- [ ] **Step 2: Verify expected failures**

Run: `python -m pytest tests/test_study_state.py -q`

Expected: import failure for `maskimpute_benchmark.study`.

- [ ] **Step 3: Implement atomic round records**

`freeze_round` must reject dirty repositories, record `method_commit`, config/protocol absolute-independent relative paths and hashes, UTC time, state `frozen`, and round ID. `materialize_final` must use `secrets.randbits(63)` to create unique generator seeds only after freeze, store the seed manifest SHA-256, and transition to `materialized`. `assert_final_runnable` must compare `git rev-parse HEAD`, `git status --porcelain`, and both file hashes with the freeze record. `record_final_evaluation` uses exclusive creation for `evaluation_receipt.json`, then transitions to `evaluated`. `supersede_round` preserves every prior file and transitions to `superseded` with a nonempty reason.

All JSON writes use a temporary sibling followed by `Path.replace`. Add `artifacts/` to `.gitignore`.

- [ ] **Step 4: Implement CLI commands**

Expose `freeze`, `materialize-final`, `verify-final`, `record-evaluation`, and `supersede`. Every command prints the round state and relevant hashes as JSON and returns nonzero on `StudyStateError`.

- [ ] **Step 5: Run unit and CLI smoke tests**

Run: `python -m pytest tests/test_study_state.py -q && python scripts/studyctl.py --help`

Expected: all state tests pass and help lists five subcommands.

- [ ] **Step 6: Commit**

```bash
git add .gitignore maskimpute_benchmark/study.py scripts/studyctl.py tests/test_study_state.py
git commit -m "feat: seal final benchmark rounds"
```

### Task 3: Truth-bearing dataset contract and inference isolation

**Files:**
- Create: `maskimpute_benchmark/schema.py`
- Test: `tests/test_dataset_schema.py`

**Interfaces:**
- Produces: `validate_benchmark_dataset(adata) -> None`, `make_inference_view(adata) -> AnnData`, `benchmark_dataset_sha256(adata) -> str`, and `TruthKind`.

- [ ] **Step 1: Write failing schema tests**

Build a six-cell/four-gene AnnData fixture with integer `X`, `pre_capture_counts`, labels, markers, and provenance. Assert exact-truth validation succeeds; negative/fractional observed counts fail; proxy truth requires `reference_counts`; and `make_inference_view` removes every truth layer plus `group`, `pseudotime`, marker columns, and all `uns` keys except normalization and non-evaluative batch.

```python
def test_inference_view_cannot_expose_truth(exact_dataset):
    view = make_inference_view(exact_dataset)
    assert not set(view.layers) & {"pre_capture_counts", "reference_counts", "expected_counts"}
    assert "group" not in view.obs
    assert "is_marker" not in view.var
    assert set(view.uns) <= {"normalization", "allowed_covariates"}
```

- [ ] **Step 2: Verify schema tests fail**

Run: `python -m pytest tests/test_dataset_schema.py -q`

Expected: import failure for `maskimpute_benchmark.schema`.

- [ ] **Step 3: Implement validation and canonical checksums**

Validate required `obs`, `uns["truth_kind"]`, provenance fields, shape agreement, finite nonnegative integer counts, unique IDs, and truth-kind-specific layers. Hash shapes, CSR arrays or dense C-order bytes, stable obs/var identifiers, truth kind, and canonical provenance; never rely on HDF5 byte identity.

- [ ] **Step 4: Implement isolated views**

Return a deep-copy AnnData containing observed counts, feature IDs, cell IDs, declared batch/covariates only, and normalization metadata. Add `uns["source_dataset_sha256"]` after stripping truth so method outputs can be bound to their input.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_dataset_schema.py -q`

```bash
git add maskimpute_benchmark/schema.py tests/test_dataset_schema.py
git commit -m "feat: isolate benchmark truth from imputation"
```

### Task 4: Complete reconstruction and zero-score metrics

**Files:**
- Create: `maskimpute_benchmark/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces: `entry_masks(observed, truth)`, `reconstruction_metrics(imputed, observed, truth, marker_genes=None, corr_gene_mask=None)`, `zero_score_metrics(p_pre_zero, observed, truth, n_bins=10)`, and `stratified_zero_score_metrics(...)`.

- [ ] **Step 1: Write failing hand-calculated tests**

Use 3×3 matrices with known observed zeros, induced dropouts, pre-dropout zeros, and nonzeros. Assert exact MSE/MAE subset values; common-gene CorrErr; mean/variance distortion; false-positive expression rate; AUROC/AUPRC/Brier/log loss; and calibration bins. Include empty-mask cases and require `reason="no_entries"` rather than silent NaN.

```python
def test_pre_dropout_zero_score_is_evaluated_only_on_observed_zeros(fixture):
    result = zero_score_metrics(fixture.p, fixture.observed, fixture.truth, n_bins=2)
    assert result["n"] == int((fixture.observed == 0).sum())
    assert result["brier"].reason is None

def test_corr_err_uses_prespecified_common_genes(fixture):
    result = reconstruction_metrics(
        fixture.imputed, fixture.observed, fixture.truth,
        corr_gene_mask=np.array([True, True, False]),
    )
    assert result["n_corr_genes"].value == 2
```

- [ ] **Step 2: Verify failures**

Run: `python -m pytest tests/test_metrics.py -q`

Expected: import failure for `maskimpute_benchmark.metrics`.

- [ ] **Step 3: Implement typed metric results**

Use a frozen `MetricValue(value: float | None, n: int, reason: str | None)` and return all prespecified keys. AUROC uses average ranks for ties; AUPRC uses average precision; probabilities are clipped only for log loss. ECE uses equal-frequency bins and returns bin-level counts, mean predictions, observed fractions, and Wilson intervals. Strata are observed-library-size quartiles and truth-expression bins `[0, 1, 2, 4, inf)`; truth strata are evaluator-only.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/test_metrics.py -q`

```bash
git add maskimpute_benchmark/metrics.py tests/test_metrics.py
git commit -m "feat: add publication benchmark metrics"
```

### Task 5: Hierarchical paired inference and multiplicity control

**Files:**
- Create: `maskimpute_benchmark/statistics.py`
- Test: `tests/test_statistics.py`

**Interfaces:**
- Produces: `hierarchical_paired_bootstrap(records, method, comparator, metric, n_boot=10000, seed=20260712)`, `holm_adjust(p_values)`, and `summarize_seed_variance(records)`.

- [ ] **Step 1: Write failing deterministic inference tests**

Create records for two mechanisms, two conditions each, three independent draws, two methods, and three nested model seeds. Assert model seeds are averaged within draw before the observed paired effect; duplicating seed rows does not change the number of independent units; a missing comparator pair is excluded with a reason count; bootstrap output is reproducible; and Holm-adjusted p-values are monotone and bounded.

```python
def test_model_seeds_do_not_inflate_independent_n(records):
    result = hierarchical_paired_bootstrap(records, "maskimpute", "dca", "mse", n_boot=200, seed=7)
    assert result.n_independent_draws == 12
    assert result.n_raw_rows == 72
```

- [ ] **Step 2: Verify failures**

Run: `python -m pytest tests/test_statistics.py -q`

Expected: import failure for `maskimpute_benchmark.statistics`.

- [ ] **Step 3: Implement blocked resampling**

Validate record keys `mechanism`, `condition`, `dataset_id`, `method`, `model_seed`, `metric`, `value`, and `status`. Average valid seeds within method/draw, pair methods on dataset ID, and calculate percent difference `(method-comparator)/abs(comparator)`. Each bootstrap replicate samples mechanisms, conditions within sampled mechanisms, draws within conditions, and raw seeds within method/draw before re-averaging. Return median effect, percentile 95% interval, probability effect `<0`, two-sided sign probability, win count, and exclusions by reason.

- [ ] **Step 4: Implement seed variance and Holm adjustment**

Seed variance is the within-draw sample variance summarized separately from between-draw variance. Holm adjustment sorts finite p-values, multiplies by remaining hypotheses, applies a cumulative maximum, caps at one, and restores original order.

- [ ] **Step 5: Run the infrastructure suite and commit**

Run: `python -m pytest tests/test_protocol.py tests/test_study_state.py tests/test_dataset_schema.py tests/test_metrics.py tests/test_statistics.py -q`

```bash
git add maskimpute_benchmark/statistics.py tests/test_statistics.py
git commit -m "feat: add hierarchical benchmark inference"
```

## Plan verification

After all tasks, run:

```bash
python -m pytest -q
git diff --check 3813a7e..HEAD
```

Expected: only project tests are collected, all pass, and no whitespace errors are reported.
