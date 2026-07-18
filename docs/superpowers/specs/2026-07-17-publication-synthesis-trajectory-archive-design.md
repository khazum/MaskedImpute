# Publication synthesis trajectory-archive integration design

## Problem

The initial publication synthesis safety gate was designed before the final
downstream replay gained a separate supplementary-trajectory archive. It also
assumes the obsolete primary source layout and an older evaluated-round binding.
A clean textual merge therefore does not provide a valid publication replay.

The synthesis must independently reconstruct and validate both downstream
archives while ensuring trajectory evidence remains descriptive. Missing or
tampered evidence is a provenance failure, not an unavailable scientific result,
and trajectory results must never rescue the competitive reconstruction gate.

## Selected design

`_load_publication_evidence` rebuilds the primary and supplementary trajectory
plans from the evaluated round. It derives both external output namespaces through
`expected_final_downstream_output_directory`, reloads the persisted plans and
manifests, and requires byte-semantic equality with the rebuilt plans.

Primary evidence uses:

```text
source root: <round>/results/final/execution
source manifest: execution_manifest.json
scope: all
```

Trajectory evidence uses:

```text
source root: <round>/results/trajectory/execution
source manifest: execution_manifest.json
scope: supplementary_trajectory
external suffix: trajectory
```

Both plans carry the same evaluated-round binding. The trajectory plan and
manifest must match the receipt's trajectory execution plan, registered dataset,
authority, binding, evidence digest, file inventory, and planned denominator.
Every external record has exactly one `trajectory_pseudotime_rank_loss` endpoint.
The execution-run denominator, receipt-owned result-file count, and external
endpoint-row count remain distinct domains and are never conflated.

The synthesized trajectory summary records terminal status/reason counts and
validated bindings with:

```json
{"role":"descriptive_only","gate_influence":"none"}
```

It has no numerical threshold and is excluded from `_competitive_gate` and
`_superiority_permissions` inputs.

## Failure semantics

Missing archive bytes, rebuilt/persisted plan drift, namespace drift, checksum or
receipt-binding mismatch, denominator mismatch, endpoint drift, and coherent
source replacement raise `PublicationSynthesisError`. Legitimate terminal
upstream statuses remain reason-coded descriptive observations.

## Verification

Production-shaped tests cover current primary layout, expanded evaluated binding,
the exact external trajectory namespace, missing/tampered archives, plan and
receipt mismatch, all denominator domains, one-endpoint records, terminal-status
retention, and gate-inertness. Existing reconstruction/null-DE safety tests remain
unchanged in meaning. Evaluated downstream replay must continue calling
`load_prepared_final_panel(..., allow_evaluated=True)` without live runtime paths.
