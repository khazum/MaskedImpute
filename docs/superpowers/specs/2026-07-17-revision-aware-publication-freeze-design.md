# Revision-Aware Publication Freeze Design

**Status:** Approved implementation design; implementation intentionally deferred

## Scope and prerequisite

Publication freeze currently assumes one base development input, one base
selection report, and one fixed set of development artifacts. The development
workflow can instead end at base, v28, or v29, and selection-complete evidence
is now an immutable schema-4 promotion at a path distinct from its schema-2/3
source input. Freeze must resolve and bind that exact terminal stage without
falling back to an earlier favorable report.

Implementation starts only after the schema-4 selection-promotion work is
integrated. That work owns `development_selection_stage_paths`, the immutable
`*-downstream.json` inputs, exact source projection, downstream completeness,
and report publication. This design consumes those contracts; it does not
change their paths or overwrite any development evidence.

## Audit findings

The existing freeze implementation is secure for its original fixed layout,
but it is not revision-complete:

- `_FIXED_PATHS` points only to the base selection input/report/evaluation
  manifest and base reconstruction checkpoint.
- `_candidate_configuration` recognizes the base search and v28 authority but
  cannot select the tracked v29 configuration.
- Tracked freeze authorities include the scaling panel but omit
  `study/trajectory_panel.json`, even though final trajectory execution derives
  mandatory evidence directly from it.
- The artifact key set is static. It cannot express the base-to-v28 or
  base-to-v28-to-v29 activation prefix, promoted inputs, downstream archives,
  or revision orthogonal evidence.
- Method execution evidence is derived entirely from the base checkpoint. For
  a selected v28/v29 candidate, that loses the selected candidate's revision
  execution even though the comparator denominator should remain the base
  denominator used by cumulative selection.
- An existing identical `study/frozen_method.json` is idempotent only when it
  exists before `_atomic_write`; a concurrent identical hard-link publication
  is rejected as a conflict.
- Clean validation deliberately does not reopen ignored development evidence.
  Therefore the frozen payload itself must carry a complete stage identity,
  while round freeze must revalidate the raw dynamic package against it.

The outward frozen-method schema remains version 1 so existing final consumers
continue to work. A new versioned `development_stage_receipt` and a dynamic,
exact `artifact_bindings` set carry the stronger evidence contract.

## Closed stage paths

The source/complete/report/downstream paths come only from
`development_selection_stage_paths(stage)`. Revision-specific paths come only
from `revision_stage_paths(version)`. Freeze must not reconstruct suffixes at a
call site or accept a path override.

| Stage | Source input | Promoted input | Report | Downstream |
| --- | --- | --- | --- | --- |
| base | `development_selection_input.json` | `development_selection_input-downstream.json` | `development_selection_report.json` | `downstream/` |
| v28 | `development_selection_input-v28.json` | `development_selection_input-v28-downstream.json` | `development_selection_report-v28.json` | `downstream-v28/` |
| v29 | `development_selection_input-v29.json` | `development_selection_input-v29-downstream.json` | `development_selection_report-v29.json` | `downstream-v29/` |

Each stage also has a fixed evaluation manifest, reconstruction directory and
checkpoint, and orthogonal directory and manifest:

- base uses `evaluation_manifest.json`, `competition-reconstruction/`, and
  `evaluation/orthogonal/orthogonal_outputs.json`;
- v28/v29 use the corresponding fields of `revision_stage_paths(version)` and
  `<orthogonal_directory>/orthogonal_outputs.json`;
- a revision stage additionally binds its tracked `study/vNN_revision.json`
  and the preceding promoted input/report named by its activation paths.

The tracked revision specifications are publication authorities but are not
stage-presence signals: both are prespecified before either revision runs.

## Active-stage resolution

Resolution is based on known operational footprints, not on the first valid or
freezeable report. The footprint for v28 or v29 consists of every owned
operational path: source input, promoted input, report, downstream directory
and manifest, evaluation manifest, reconstruction directory and checkpoint,
and orthogonal directory and manifest. Presence uses `os.path.lexists`, so a
broken symlink or an empty/partial directory is still evidence that the stage
was started and cannot be silently ignored.

The resolver applies this exact order:

1. If any v29 footprint path exists, the active stage is v29.
2. Otherwise, if any v28 footprint path exists, the active stage is v28.
3. Otherwise, the active stage is base.

After choosing, it requires the complete consecutive prefix: `(base)`,
`(base, v28)`, or `(base, v28, v29)`. Every required footprint must be present,
regular/non-symlink where a file is expected, and semantically valid. Any
partial or invalid newest footprint is fatal. There is no `try v29, then v28`
or `try v28, then base` path.

Known stage-family paths with a suffix other than v28/v29 are rejected rather
than guessed. The CLI remains stage-free; repository evidence alone chooses
the stage.

## Exact selection and activation chain

For every stage in the resolved prefix, freeze securely and canonically reads
the source input, promoted input, and report, then performs all of the
following before retaining any result:

1. The source is exact schema 2 for base or schema 3 with consecutive
   `revision_versions` for v28/v29.
2. The promoted input is exact schema 4 with `revision_versions` equal to the
   stage prefix.
3. Schema-4 source projection reopens the fixed source and requires exact
   payload equality plus the source path, file SHA-256, and source result hash.
4. Downstream completeness reopens the fixed stage directory and validates its
   plan, manifest, exact record prefix, source checkpoints, statuses, endpoint
   denominator, and base/revision source bindings. The promoted binding's path,
   manifest file/payload hashes, and plan hash must equal the independently
   loaded archive.
5. `_select_for_repository(..., require_clean=True)` recomputes the report from
   the promoted input, and the canonical report file must equal it exactly.

The reports must form this state machine:

| Active stage | Required preceding reports | Required active report |
| --- | --- | --- |
| base | none | `trigger=freeze_candidate`, one selected configuration |
| v28 | base has `trigger=v28` and no selection; exact v28 activation revalidates | `trigger=freeze_candidate`, selected v28 configuration |
| v29 | base has `trigger=v28`; v28 has `trigger=v29`; exact v28 and v29 activations revalidate | `trigger=freeze_candidate`, selected v29 configuration |

`validate_revision_activation` is called independently for each revision and
its returned input/report paths and hashes must equal the retained preceding
stage bindings. An active `downgrade_claim`, a revision trigger at the terminal
stage, a selection at a preceding stage, or an active selected version older
than the resolved stage blocks publication. Claim downgrade is a separate
workflow outcome, never an instruction to freeze an earlier method.

## Dynamic closed inventory

The static `_FIXED_PATHS` split becomes:

- common tracked authorities: runtime lock, method registry, selection
  contract, development search, **both** v28 and v29 revision authorities,
  ablation registry, scaling panel, trajectory panel, protocol, and SAVER
  authorities;
- common development evidence: dataset status, count-score manifest, retained
  calibration, and the conditional external-reference checkpoint;
- one exact stage-qualified binding group for every stage in the resolved
  prefix.

Each stage contributes these unique artifact keys:

```text
<stage>_selection_source_input
<stage>_selection_complete_input
<stage>_selection_report
<stage>_evaluation_manifest
<stage>_reconstruction_checkpoint
<stage>_orthogonal_manifest
<stage>_downstream_plan
<stage>_downstream_manifest
```

`<stage>` is exactly `base`, `v28`, or `v29`. Each artifact binding remains the
closed `{path, sha256}` object accepted by `_artifact_bindings`. There are no
legacy aliases for the active input/report: one logical core file has one
stage-qualified key. Validation requires exact equality with the key set
derived from the receipt's stage prefix, plus the conditional external
checkpoint. Missing, extra, duplicate-path, unsafe, or wrong-stage bindings
are rejected.

Core manifests bind their transitive files, but the frozen receipt also closes
each stage-owned reconstruction, orthogonal, and downstream directory with a
no-symlink tree receipt over relative paths, entry types, modes, file bytes,
and an explicit rejection of symbolic links and special files. Only directories
and unique regular files are permitted. The receipt stores one SHA-256 per fixed
directory.
This catches unreferenced additions and coherent whole-manifest replacement
without flattening thousands of record bindings into `artifact_bindings`.

The frozen payload adds this exact logical structure:

```text
development_stage_receipt = {
  schema_version: 1,
  active_stage: "base" | "v28" | "v29",
  revision_versions: [] | ["v28"] | ["v28", "v29"],
  stage_order: ["base", ...],
  stages: [
    {
      stage,
      source_input_artifact,
      complete_input_artifact,
      report_artifact,
      evaluation_manifest_artifact,
      reconstruction_checkpoint_artifact,
      orthogonal_manifest_artifact,
      downstream_plan_artifact,
      downstream_manifest_artifact,
      source_result_sha256,
      complete_result_sha256,
      downstream_plan_sha256,
      downstream_manifest_sha256,
      reconstruction_tree_sha256,
      orthogonal_tree_sha256,
      downstream_tree_sha256,
      activation
    }
  ],
  artifact_names: [...],
  inventory_sha256
}
```

Base `activation` is null. Revision activation is an exact object containing
the version/trigger, revision-authority artifact key, preceding complete-input
and report artifact keys, and the activation input file/result/report hashes.
`artifact_names` is sorted and contains precisely the stage-qualified
development keys. `inventory_sha256` hashes the unsigned receipt body plus
those exact artifact bindings and tree receipts; the outer frozen
`payload_sha256` authenticates the whole object again.

Preparation reads and hashes the dynamic inventory, performs semantic replay,
then re-resolves the active stage and rereads every core file/tree receipt. A
stage appearing, disappearing, or changing during preparation is fatal.

## v29 configuration authority

Candidate lookup is generalized to the base development search plus the exact
tracked v28 and v29 revision specifications. The selected ID must occur once,
its configuration hash must recompute, and its method version must match both
the active stage and the selected assessment. v29 therefore receives the same
tracked-file and semantic validation as v28; it is not reconstructed from a
report payload.

Both revision files remain bound even for a base freeze because they are
prespecified tracked publication authorities. Only activated revisions appear
in `revision_versions` and the stage prefix.

`study/trajectory_panel.json` is always bound alongside
`study/scaling_panel.json`. Final execution regenerates its mandatory
receipt-bound trajectory dataset directly from that authority, so relying only
on the enclosing Git commit would be weaker than the existing scaling contract.
It is tracked authority, not an activation footprint.

## Execution-evidence denominator

Cumulative revision selection retains the base comparator rows and adds only
the revision candidate rows. Frozen method execution evidence must mirror that
denominator:

- build the full same-input comparator evidence from the base reconstruction
  checkpoint;
- for base selection, retain it unchanged;
- for v28/v29 selection, independently validate the selected stage checkpoint
  and replace only the `maskimpute` evidence row with that checkpoint's exact
  selected-candidate execution evidence;
- keep observed, learned comparators, and capacity-matched control bound to the
  base checkpoint; keep external-reference-only methods on the production
  external checkpoint.

The selected stage checkpoint must contain exactly one selected MaskImpute
configuration matching the tracked revision configuration. Other revision-run
rows cannot replace the base comparator denominator. Per-method evidence names
the stage-qualified checkpoint artifact, so the mixed provenance is explicit
and self-authenticating.

## Immutable and idempotent publication

`prepare_frozen_method` remains create-only. It writes canonical bytes to a
same-directory temporary file, fsyncs the file, and attempts an atomic hard
link. If the destination already exists before or during the link:

- a secure, pinned read of byte-identical canonical content succeeds
  idempotently;
- different, unsafe, non-regular, multiply linked, or noncanonical content is
  rejected and never changed.

The temporary is removed and the directory is fsynced. A failure after a
successful link may leave only the complete intended destination, so a retry
is safe. There is no overwrite, rename-over-existing, unlink, or repair path.

The stage receipt makes idempotence stage-sensitive. A later v28/v29 footprint,
changed evidence, or attempted downgrade changes the recomputed bytes and
therefore conflicts with an existing receipt. Before opening a final round,
`freeze_publication_round` re-resolves raw evidence and compares its complete
stage/inventory receipt with the committed frozen method. Clean
`validate_frozen_method` may still operate after ignored development evidence
is removed, but it validates the exact dynamic key schema, tracked authorities,
configuration, receipt hashes, and payload self-authentication.

## Failure semantics and verification

Tests must cover base, v28, and v29 success; every partial newest footprint;
invalid and symlinked newest evidence; exact activation/report transitions;
schema-2/3 fallback attempts; schema-4 source/downstream tampering; v29 tracked
authority/configuration tampering; scaling/trajectory authority tampering;
missing/extra dynamic bindings; tree additions, special files, symlinks, and
coherent replacements; selected-stage candidate evidence with base comparator
evidence; stage appearance during preparation; stage drift between preparation
and round freeze; identical sequential and concurrent publication; and
conflicting concurrent publication retention.

No test or implementation step runs development or final scientific evidence.
