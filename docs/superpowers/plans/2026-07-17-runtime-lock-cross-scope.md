# Cross-scope publication runtime-lock implementation plan

**Goal:** Permit the complete publication runtime lock to contain explicitly
bound external-reference-only entries while same-input/scaling/final registries
probe only their executable scope.

**Constraints:** Preserve the full lock SHA, exact default validation, external
workflow live validation, final CLI surface, method applicability semantics, and
the fixed scientific denominator. Do not run a scientific competition or final
round in this task.

## Task 1: Lock-validator contract (RED then GREEN)

Modify `maskimpute_benchmark/runtime_environments.py` and
`tests/test_runtime_environments.py`.

- Add `lock_only_environment_ids: Sequence[str] = ()`.
- RED tests: accepted explicit extra without probing it; duplicate/unsafe/overlap
  rejection; unknown extra rejection; unchanged exact default behavior.
- Require the exact disjoint union of live and lock-only IDs.
- Return sorted live and lock-only inventory-hash tuples separately.

## Task 2: Execution registry retention (RED then GREEN)

Modify `maskimpute_benchmark/runner.py` and
`tests/test_benchmark_runner.py`.

- Extend `ExecutionEnvironmentRegistry.fixed` with the lock-only tuple.
- Persist it in the frozen registry, include it in `registry_sha256`, and pass it
  during `full_revalidate`.
- Derive ready external-reference-only IDs from `MethodRegistry` with a single
  helper; reject malformed or overlapping scope.
- Prove lock-only IDs never enter `executable_paths` or adapter dispatch.

## Task 3: Production boundary propagation (RED then GREEN)

Modify the minimum required call sites in:

- `maskimpute_benchmark/runner.py`
- `maskimpute_benchmark/final_runner.py`
- `maskimpute_benchmark/scaling.py`

and corresponding benchmark-runner, final-runner, and scaling tests.

- Pass the registry-derived tuple at development competition construction.
- Make final and scaling registry loaders receive the validated method registry
  and pass the same derived tuple.
- Preserve the path-free evaluated replay and frozen-final CLI signatures.

## Task 4: Tracked parity and denominator regression

Add a cheap production-shaped test without runtime probes.

- Assert the tracked lock has the exact thirteen sorted IDs.
- Derive eleven live IDs plus exactly `d3impute` and `sctsi` as lock-only from
  `study/methods.json`; require their disjoint union to equal the lock IDs.
- Build the authority against a sixteen-binding fixture and assert exactly 1,744
  entries, with no D3Impute/scTsI entry.
- Retain external-reference workflow tests demonstrating those methods remain
  executable only in the matched-bulk track.

## Task 5: Verification and review

Run focused RED/GREEN nodes, then warning-strict runtime-environment,
benchmark-runner, final-runner, scaling, external-reference, freeze, and method
registry suites. Run Ruff format/check, supported-Python compileall, and
`git diff --check`. Inspect the exact range, commit atomically, and obtain an
independent review before integration.
