# Publication synthesis trajectory-archive integration plan

**Goal:** Adapt the publication synthesis gate to the current evaluated binding,
primary downstream layout, and separate receipt-bound descriptive trajectory
archive without allowing trajectory results to affect competitive claims.

## Task 1: Repair merged fixtures and primary bindings (RED then GREEN)

Modify `tests/test_publication_synthesis.py` and
`maskimpute_benchmark/publication_synthesis.py`.

- Expand the evaluated-round fixture with all trajectory fields.
- Use `<round>/results/final/execution/execution_manifest.json` for the primary
  source.
- Add a regression that the merged synthesis loads current primary evidence.
- Repair `_validate_downstream_bindings` only to the current contract.

## Task 2: Add separate trajectory evidence fixtures (RED)

- Build a registered trajectory dataset/authority/binding fixture.
- Build the current eight-entry, one-endpoint plan and manifest fixture while
  keeping execution runs, receipt-owned files, and external rows separate.
- Extend `_LoadedPublicationEvidence` and loader seams with distinct primary and
  trajectory directories.
- Add missing archive, rebuilt/persisted plan drift, receipt-binding mismatch,
  denominator/endpoint drift, and exact-namespace RED tests.

## Task 3: Load and validate the trajectory archive (GREEN)

- Rebuild with `build_final_trajectory_downstream_evidence_plan`.
- Derive the `trajectory` external namespace through the production helper.
- Reload and compare the persisted plan and manifest.
- Add `_validate_trajectory_downstream_bindings` and require exact receipt,
  dataset, authority, plan, file-inventory, count, and endpoint relationships.
- Strengthen the primary report binding so its trajectory evidence digest equals
  the evaluated binding, rather than merely matching digest syntax.

## Task 4: Emit a gate-inert descriptive summary

- Add `_trajectory_summary` with status/reason counts and validated bindings.
- Require `role == descriptive_only` and `gate_influence == none`.
- Add paired tests proving any valid trajectory values/statuses leave competitive
  and superiority permissions byte-identical.
- Preserve legitimate terminal upstream statuses as descriptive rows.

## Task 5: Integration gates and review

Run warning-strict publication-synthesis and final-analysis suites plus the
targeted downstream/trajectory/final-runner regressions named in the design
audit. Run broader downstream/null-DE suites, Ruff format/check, compileall, and
`git diff --check`. Inspect the exact base-to-tip range, commit atomically, and
obtain independent review before integration.
