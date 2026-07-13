# Comparator Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make comparator scheduling and failure receipts scientifically exact while preserving pinned upstream behavior and producing a reproducible SAVER R environment.

**Architecture:** Registry records separate execution scope from integration readiness and expose immutable planner entries without changing the runner. Adapter subprocesses retain native behavior, while the Python boundary classifies invalid outputs and binds exact environment evidence.

**Tech Stack:** Python 3.10+, pytest, NumPy, R 4.6.1, pinned CRAN source tarballs, Bash, Ruff.

## Global Constraints

- Start at `b60c3bc75f4cc90219009febb33826b5055e29ff` in a new isolated worktree.
- Do not edit `maskimpute_benchmark/runner.py` or evaluator/selection files.
- Never clip comparator output or modernize legacy scSDAE.
- Keep SAVER packages in an external or ignored library; track only its lock/build inputs and evidence.

---

### Task 1: Registry execution applicability

**Files:**
- Modify: `maskimpute_benchmark/methods/base.py`
- Modify: `maskimpute_benchmark/methods/registry.py`
- Modify: `study/methods.json`
- Test: `tests/test_method_registry.py`

**Interfaces:**
- Produces: `MethodSpec.execution_scope`, `MethodSpec.applicability_reason`, `ResourceSpec.gpu_mode`, and `MethodRegistry.execution_plan()` returning immutable `MethodPlanEntry` values with `executable` derived from scope/reason.

- [x] Add failing tests asserting required same-input methods are executable, D3Impute/scTsI are `external_reference_only`, scImpute/WEDGE are `historical_not_run`, scGAC/scTACL are unavailable with the exact required reasons, scTACL does not claim raw-count output, and malformed scope/reason combinations fail closed.
- [x] Run `python -m pytest tests/test_method_registry.py -q` and confirm failures come from missing fields/interface.
- [x] Add the closed registry fields and planner validation, retain scCR as GPU-required after proving the supported selected executable exposes CUDA, and update all registry entries consistently.
- [x] Re-run the registry test and confirm it passes.

### Task 1A: Current-method discovery dispositions

**Files:**
- Modify: `study/methods.json`
- Create: `study/method-attempts/sczn.json`
- Create: `study/method-attempts/scgimpute.json`
- Test: `tests/test_method_registry.py`

**Interfaces:**
- Produces: source-pinned `sczn` with exact `upstream_not_packaged_as_callable_method` evidence and a non-executable scGImpute discovery receipt with exact `public_source_not_located`.

- [x] Audit the official scZN repository HEAD, license, input/label dependencies, and output alignment; add a failing receipt/registry test.
- [x] Pin the pristine source revision/tree and record the notebook-only supervised-label boundary without creating a reimplementation.
- [x] Search the scGImpute primary paper and public source surfaces on 2026-07-12; add a failing discovery-receipt test and the exact non-executable result.
- [x] Run the focused registry tests and confirm pass.

### Task 2: afMF native-negative receipt

**Files:**
- Modify: `maskimpute_benchmark/methods/afmf.py`
- Test: `tests/test_priority_method_adapters.py`

**Interfaces:**
- Produces: `run_afmf(...)` raises `AdapterUnavailableError(reason_code="upstream_negative_native_output")` whose detail contains deterministic `negative_count=<n>` and `minimum=<value>` diagnostics and whose command/log hashes remain bound.

- [x] Add a fake pinned-source launcher test that returns a negative native matrix and assert the exact reason/diagnostics and absence of an output snapshot.
- [x] Run the single test and confirm the current generic failure classification fails.
- [x] Let the subprocess serialize finite, shape-correct native output; classify negative values in the parent before finalization without clipping.
- [x] Re-run the afMF tests and confirm pass.

### Task 3: scCR native device selection

**Files:**
- Modify: `maskimpute_benchmark/methods/sccr.py`
- Test: `tests/test_required_legacy_method_adapters.py`

**Interfaces:**
- Produces: `SCCRConfig.device=None` means interpreter-local auto selection (`cuda:0` iff `torch.cuda.is_available()`, else `cpu`); explicit `cpu`/`cuda:<index>` remain auditable overrides.

- [x] Add a CPU-only fake launcher test asserting the command requests `auto`, the receipt binds `device=cpu`, and the registry remains GPU-required after the supported executable's real CUDA probe passes.
- [x] Run the single test and confirm the hard-coded CUDA default fails.
- [x] Implement device selection inside the selected executable and update compatibility text.
- [x] Run the scCR focused tests and confirm pass.

### Task 4: scSDAE GPU0 kernel preflight

**Files:**
- Modify: `maskimpute_benchmark/methods/scsdae.py`
- Test: `tests/test_required_legacy_method_adapters.py`

**Interfaces:**
- Produces: failure of the explicit logical GPU0 TensorFlow kernel probe raises `SCSDaeUnavailableError(reason_code="legacy_gpu_kernel_incompatible")` retaining the full probe command/stdout/stderr hashes and no run receipt.

- [x] Add a fake launcher kernel-probe failure test with the canonical sentinel and assert attempt evidence.
- [x] Run the single test and confirm current classification does not recognize the sentinel.
- [x] Add the exact TensorFlow 1.12 logical-GPU0 matrix kernel and canonical sentinel classification; do not alter the training implementation.
- [x] Re-run scSDAE focused tests and confirm pass.

### Task 5: Locked SAVER library and smoke

**Files:**
- Create: `environments/saver-r.lock.json`
- Create: `scripts/build_saver_r_environment.sh`
- Create: `environments/saver-r.build-receipt.json`
- Modify: `maskimpute_benchmark/methods/saver.py`
- Modify: `study/methods.json`
- Test: `tests/test_core_method_adapters.py`

**Interfaces:**
- Consumes: explicit `library_dir: Path` and `lock_manifest: Path` in `run_saver`.
- Produces: a receipt containing manifest SHA-256 plus exact Rcpp, RcppEigen, foreach, iterators, shape, glmnet, doParallel, Matrix, and SAVER versions, all loaded from the selected external library.

- [x] Add failing tests that reject absent/mismatched libraries and require the complete lock-bound receipt.
- [x] Run the SAVER focused tests and confirm failure.
- [x] Lock source URLs/SHA-256 values and the pinned SAVER source revision/tree, build into `/tmp/maskimpute-saver-r461`, and emit a deterministic build receipt.
- [x] Update the adapter to use only the selected library plus R base/recommended libraries, never install during a benchmark run, and validate package versions against the lock manifest.
- [x] Run a real tiny pinned SAVER smoke twice and confirm deterministic output and bound receipts. If build/network fails, retain exact build evidence and leave registry status non-ready.

### Task 6: Verification and handoff

**Files:**
- Verify only owned files and focused tests.

- [x] Run focused registry/adapter tests, then the complete test suite.
- [x] Run `ruff check` on changed Python/tests, check `git diff --check`, verify `runner.py` and selection/evaluator paths are unchanged, and inspect the final diff.
- [x] Request an independent review if a concurrency slot is available; fix Critical/Important findings.
- [x] Commit the verified branch and report commit SHA plus SAVER/environment limitations.
