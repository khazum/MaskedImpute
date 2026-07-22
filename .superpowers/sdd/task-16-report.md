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

## Independent-review fixes

The Task 16 independent review identified one Critical synthesis boundary and
one Important persisted-decoder boundary. Both are repaired in a separate
follow-up commit:

- Publication synthesis now decodes the exact six-field scheduled status rows
  accepted by publication freeze. Completed controls, selected comparators,
  and intrinsic-terminal unavailable comparators remain distinct. Unavailable
  status/reason evidence is recomputed from the closed configuration terminal
  denominator instead of relying on a fabricated top-level reason.
- The downstream seven-field direct configuration envelope now decodes to a
  typed `FrozenPlanMethodAuthority`. Selected configurations use the readable
  direct bound-comparator decoder; nonexecution authorities use the closed
  bound-configuration decoder for every denominator row and are recursively
  frozen. Method/configuration/kind/requirements, exclusivity, disposition,
  and seed policy are revalidated before the value reaches any plan builder.
- The closed readable comparator decoder also received its missing local
  `BoundComparatorConfiguration` import, exposed by the new persisted decoder
  regression.
- Synthesis fixtures now reuse production-shaped freeze rows; the obsolete
  fabricated `{schema_version, reason}` nonexecution fixture is gone.

### Corrected TDD evidence

Earlier exploratory results obtained with an unsupported interpreter are
discarded. The authoritative review-fix RED/GREEN cycle used only:

```bash
env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m pytest \
  tests/test_publication_synthesis.py tests/test_downstream_evidence.py \
  -k 'scheduled_claim_permissions_accept_production_completed_and_selected_rows or scheduled_claim_permissions_decodes_production_nonexecution_denominator or direct_downstream_configuration_decoder_restores_exact_typed_authority or generic_downstream_builder_accepts_decoded_direct_configuration or persisted_direct_final_and_trajectory_plans_reload_typed_configurations' \
  -q -W error -p no:cacheprovider
```

- Exact `3bb46ae` production behavior with the new tests: `6 failed, 154
  deselected in 100.18s`. The failures proved selected/unavailable synthesis
  incompatibility, the untyped decoder, generic builder rejection, and both
  persisted final and trajectory reload boundaries.
- First corrected implementation run: the two synthesis cases passed; all four
  decoder cases exposed the missing local `BoundComparatorConfiguration`
  import in `decode_direct_bound_comparator_value`.
- After that one-line closed-decoder repair: `6 passed, 154 deselected in
  98.50s`.
- After Ruff formatting: `6 passed, 154 deselected in 101.49s`.

The complete synthesis file initially exposed the old hand-built scheduled
rows and one stale `completed` expectation for a production `selected` row.
After migrating those fixtures, the exact supported-interpreter synthesis run
passed: `61 passed in 47.36s`.

### Review-fix final verification

The exact required warning-strict seven-file suite passed after the fixture
migration:

```text
623 passed in 2747.47s (0:45:47)
```

The command was the same supported-interpreter command recorded above in
`Final verification`, with `-q -W error -p no:cacheprovider` and no additional
environment or warning flags.

Fresh post-format checks also passed:

- Ruff lint and Ruff format check on all five changed Python files.
- `env -u LD_LIBRARY_PATH /tmp/maskimpute-supported/bin/python -m compileall
  -q maskimpute_benchmark tests`.
- `git diff --check`.
- `rg -n 'required_comparator_ids' maskimpute_benchmark` returned no matches.

No scientific workload, figure generation, asset export, ledger edit, or
publication export was run during the review fix.

## Typed-parser closure re-review fix

The Task 16 re-review found that persisted direct authorities still admitted
internally consistent forgeries. The shared bound-configuration validator now
requires the decoded `ComparatorMethodBinding` to equal the complete canonical
projection from `comparator_method_binding(registry.by_id(method_id))`. This
authenticates every method field for both readable selected configurations and
the payload-JSON configurations used by nonexecution authorities.

The nonexecution parser now additionally:

- requires an exact integer `schema_version` and rejects `bool`;
- requires the complete canonical configuration denominator for the method;
- preserves canonical row order; and
- verifies the exact canonical method and authority on every row.

Valid selected, nonexecution, generic-builder, final-plan, and trajectory-plan
roundtrips retain their typed recursively frozen values.

### Strict TDD evidence

All commands used the supported interpreter and warning-strict pytest flags.
Before production edits, the new method-component, omitted-row, reordered-row,
and bool-as-int mutations produced:

```text
45 failed, 2 passed, 99 deselected in 59.94s
```

The two already-rejected cases were the existing method-ID consistency checks;
the remaining 45 failures established the open boundary. After the production
change, the identical focused command produced:

```text
47 passed, 99 deselected in 55.56s
```

The complete downstream evidence file then produced:

```text
146 passed in 88.00s (0:01:28)
```

The targeted adjacent comparator-tuning, scaling, and benchmark-runner command
produced:

```text
17 passed, 295 deselected in 107.06s (0:01:47)
```

After Ruff formatting, the focused mutation and valid roundtrip regression set
produced:

```text
51 passed, 95 deselected in 71.58s (0:01:11)
```

The prior 623-test gate was not rerun, as requested for this narrow closure;
the adjacent warning-strict run exposed no broad impact.

### Typed-parser closure verification

- Ruff formatting completed, Ruff lint passed, and Ruff format check passed on
  all three changed Python files.
- Supported-interpreter `compileall` passed for `maskimpute_benchmark` and
  `tests`.
- `git diff --check` passed.
- The production legacy scan for `required_comparator_ids` returned no matches.
- No hashes, summaries, compatibility/default reconstruction, scientific
  workload, figure generation, asset export, publication export, or ledger edit
  was introduced or run.
