# Task 16 report: propagate frozen comparator identities

## Outcome

Implemented the complete Task 16 migration from the accepted Task 15 base
`c97e349fee20101a2f353e8e2674f0f1ae2c5799`.

- Scaling now consumes the exact frozen MAGIC, DCA, and scVI
  `BoundComparatorConfiguration` values. Direct comparator plan entries,
  executor receipts, stored runs, metrics, checkpoints, and publication
  evidence carry the complete readable configuration and compare it directly.
- Scaling rejects registry-default reconstruction and rejects drift in the
  configuration payload, method binding, and comparator authority reference.
- Downstream final, trajectory, and development evidence accepts the frozen
  method-authority wrapper, carries either the complete selected comparator
  configuration or complete nonexecution identity, and omits comparator content
  summaries. The exact direct final execution-request receipt is supported.
- Terminal selected-comparator rows remain present with their original
  `timeout`/`resource_exceeded` statuses, exact reasons, and null endpoints.
- Final trajectory replay decodes and replays the exact frozen method-authority
  wrapper and direct run identities while preserving unrelated legacy run
  provenance.
- Final analysis and publication synthesis expose
  `scheduled_same_input_ids` separately from
  `numerical_comparison_population_ids`. Numerical ranks/effects use only the
  frozen ready population.
- Publication synthesis keeps all scheduled methods visible, reports exact
  frozen nonexecution status/reason values, and assigns per-method superiority
  permission as `allowed`, `control_not_a_superiority_target`,
  `unavailable_uncompared_method`, or `insufficient_completed_cells`.
- Removed the live `SelectionAuthority.required_comparator_ids` compatibility
  property. Production code contains no remaining occurrence of that name.

`evaluation_manifest.py` required no production migration: its remaining
configuration-summary fields cover unrelated legacy MaskImpute reconstruction
and orthogonal provenance, not the direct final/scaling comparator segment.

## TDD evidence

Fresh RED/GREEN boundaries included:

- Scaling exact frozen configurations: RED on the old wrapper/default path;
  GREEN after direct frozen MAGIC/DCA/scVI propagation.
- Publication scheduled/numerical denominator split: RED because the synthesis
  output lacked `scheduled_same_input_ids`; GREEN with unavailable BiAEImpute
  retained and superiority gated.
- Final claim gate: RED when scheduled/numerical authority was supplied but the
  old unavailable path was returned; GREEN after the numerical-only gate
  migration.
- Selection authority: RED because
  `numerical_comparison_population_ids` was absent; GREEN after removing the
  legacy property.
- Downstream frozen configuration wrapper: RED when direct frozen method
  authority was rejected; GREEN after exact selected/nonexecution decoding.
- Downstream terminal rows: RED when selected timeout/resource rows could not
  carry direct identities; GREEN with exact status/reason preservation.
- Downstream stored endpoints: RED while comparator summary fields were still
  emitted; GREEN after complete direct values replaced them.
- Trajectory replay: RED on the frozen wrapper configuration schema, then RED
  on the exact legacy replay field set; GREEN after typed frozen-plan decoding
  and preservation of legacy outer provenance.
- Direct final execution request: RED because downstream accepted only the
  legacy nine-field request; GREEN after adding the exact six-field
  `frozen_comparator_direct` branch (`1 passed in 30.29s`).
- Direct scaling tamper matrix proves rejection at the executor receipt,
  checkpoint run, and stored metric boundaries for payload, method, and
  authority mutations.
- Combined later-cell regression proves a selected comparator final timeout and
  trajectory resource failure retain null endpoints and exact reasons, stay in
  both scheduled/numerical sets, and receive only
  `insufficient_completed_cells` for unsupported superiority.

Focused/adjacent suite evidence during implementation:

- Scaling: `46 passed in 261.69s` before the final added tamper regressions.
- Downstream: `94 passed in 71.47s` before the final direct-request regression.
- Final analysis: `53 passed in 189.34s`.
- Publication synthesis: `58 passed in 2.74s` before the final later-cell
  regression.
- Post-format critical regression set: `11 passed in 81.10s`.

## Final verification

Required warning-strict seven-file suite:

```text
617 passed in 2656.61s (0:44:16)
```

Command:

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_scaling_panel.py tests/test_downstream_evidence.py \
  tests/test_final_analysis.py tests/test_publication_synthesis.py \
  tests/test_selection_authority.py tests/test_benchmark_runner.py \
  tests/test_final_runner.py -q -W error -p no:cacheprovider
```

Additional verification:

- Ruff lint: passed.
- Ruff format check: passed after formatting six changed files.
- `python -m compileall -q maskimpute_benchmark tests`: passed.
- `git diff --check`: passed.
- `rg -n 'required_comparator_ids' maskimpute_benchmark`: no matches.
- No scientific workload, figure, asset, or publication export was run.

## Concerns

None. The existing p_pre_zero score-evidence envelope and unrelated legacy
outer provenance remain unchanged by design.
