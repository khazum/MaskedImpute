# Schema-4 Selection Promotion Design

**Status:** Approved implementation design

## Problem and decision

The base, v28, and v29 builders immutably publish schema-2/3 selection inputs
before downstream evidence exists. Reusing those paths for schema 4 would
require overwriting evidence that already authorizes the downstream run. The
production workflow will therefore retain each schema-2/3 input as immutable
source evidence and publish the evidence-bearing schema-4 input to a distinct
fixed path:

| Stage | Immutable source input | Selection-complete input | Downstream evidence |
| --- | --- | --- | --- |
| base | `development_selection_input.json` | `development_selection_input-downstream.json` | `downstream/` |
| v28 | `development_selection_input-v28.json` | `development_selection_input-v28-downstream.json` | `downstream-v28/` |
| v29 | `development_selection_input-v29.json` | `development_selection_input-v29-downstream.json` | `downstream-v29/` |

The alternatives rejected were replacing the source input in place, which
breaks immutability and loses the exact upstream bytes, and allowing selectors
to choose either schema 2/3 or schema 4, which permits a silent bypass of the
downstream completeness gate.

## Production API and command

A selection-promotion module owns the closed stage-to-path mapping and exposes
an API that accepts only the repository and the stage (`None`, `v28`, or
`v29`). It securely reads the fixed source input, calls the existing downstream
attachment validator against the fixed evidence directory, verifies the exact
schema and revision chain, and immutably publishes canonical schema 4 at the
fixed destination. The result reports source, downstream-manifest, promoted
payload, and promoted-file hashes.

`scripts/promote_development_selection_input.py` has no scientific, stage, or
path options. It discovers the latest exact source stage using the existing
fail-closed revision detector and promotes only that stage. Operators run it
immediately after `run_development_downstream_evidence.py` and before the
stage's selection command.

## Evidence binding and consumers

The schema-4 `downstream_evidence` object additionally binds the exact source
selection path, source file SHA-256, and source `result_sha256`. Schema-4
validation re-reads that source and requires it to equal the legacy projection
of the promoted payload byte-for-semantics, including the exact base or
consecutive revision identity. The existing manifest file and payload hashes
continue to bind the downstream side of the promotion.

Base selection, v28 selection, v29 selection, and revision activation consume
only the corresponding downstream-suffixed input. There is no fallback to the
schema-2/3 source path. Revision builders continue to write the source paths,
and downstream stage discovery continues to inspect them, preserving the
acyclic order: source input, downstream evidence, schema-4 promotion,
selection report, then optional revision activation.

## Publication and failure semantics

Publication freeze uses the base downstream-suffixed selection path until the
separate revision-aware freeze work selects an applicable stage. Publication
never treats a schema-2/3 source input as selection-complete.

Publishing uses a same-directory temporary file, file and directory `fsync`,
and an atomic hard-link create. A conflicting destination is rejected without
modification; an identical existing destination makes retry idempotent. A
failure before the link leaves no destination or temporary file, while a
failure after the link leaves the complete immutable bytes.

## Verification

Tests cover each stage's fixed paths, schema/revision rejection, exact source
and manifest tamper rejection, wrong-round evidence, conflicting destination,
idempotent retry, simulated publication interruption, selector and activation
use of only promoted paths, and a CLI with no stage or path override. Focused
warning-strict tests, Ruff, compileall, and `git diff --check` complete the
change.
