# Direct-identity fair-comparator design

## Status

Approved in conversation on 2026-07-20. This design supersedes only the
identity and provenance mechanism in the fair-comparator subproject described
by `2026-07-18-fair-comparator-tuning-design.md`. The scientific grid,
development-only selection rule, execution budgets, readiness thresholds,
complete scheduled denominator, and publication target remain unchanged.

This document does not authorize a scientific run. Implementation, focused
verification, independent review, calibration repair, licensing, and the
remaining publication gates must finish first.

## Decision

The fair-comparator segment will use explicit typed snapshots and direct
equality. It will not compute, emit, require, or validate content digests.
Comparator identity consists of readable schema revisions, closed typed
configuration payloads, ordered plan entries, explicit dataset descriptors,
and complete execution records.

The following names are forbidden in fair-comparator schemas and generated
artifacts: `hash`, `digest`, `checksum`, `fingerprint`, `sha`, and algorithm
variants of those names. Fair-comparator code paths must not call `hashlib`,
`canonical_sha256`, file-digest helpers, or equivalent content-summary
functions.

Direct equality means comparing the actual typed values or canonical encoded
bytes. Replacing one digest algorithm with another, renaming a digest field,
or storing an encoded digest under a neutral name would violate this design.

## Boundary

The hash-free segment begins when `study/comparator_tuning.json` is loaded and
includes:

- comparator authority parsing and typed payload decoding;
- fair-comparator references in `study/selection_contract.json` and
  `study/development_search.json`;
- development plan construction for comparator-tuning rows;
- comparator requests, transaction intents, checkpoint records, budget replay,
  smoke receipts, log receipts, and selection receipts;
- development-evaluation projection of the selected comparator map; and
- the typed selected-comparator map handed to publication-freeze consumers.

Existing publication, dataset, calibration, runtime, revision, final-runner,
scaling, and archive provenance mechanisms outside this segment are unchanged.
Shared modules may retain those legacy mechanisms for their legacy modes, but
the fair-comparator branch must not invoke them or place their fields in a
fair-comparator artifact. The handoff to the legacy publication pipeline is a
plain typed selected-comparator map; any pre-existing outer publication
provenance applied later is outside this subproject and must not be expanded by
this migration.

Previously generated fair-comparator artifacts using the superseded schema are
invalid and are not upgraded in place. No compatibility alias or dual-write
period is permitted.

## Authority model

`study/comparator_tuning.json` remains the sole tracked comparator-grid
authority. Its top-level identity is a readable `authority_revision`, initially
`fair-comparator-direct-v1`, alongside its integer schema version. Each of its
34 configuration rows contains exactly:

- `method_id`;
- `configuration_id`;
- `is_upstream_default`; and
- the complete adapter `payload`.

The existing metric, collapse, selection, readiness, budget, storage, smoke,
path, and denominator policies remain closed fields in the same document.
Configuration-level and document-level digest fields are removed.

The loader validates the document against the approved method order,
configuration order, full expected payload table, policy constants, and typed
adapter dataclasses. It compares canonical JSON bytes directly where encoded
identity matters. Duplicate keys, nonfinite numbers, signed negative zero,
unknown or missing fields, bool-as-int values, type coercion, partial payloads,
and post-decode normalization drift fail closed.

`study/selection_contract.json` and `study/development_search.json` bind the
authority using its fixed path, schema version, and `authority_revision`.
Loading either document also loads and fully validates the referenced authority;
the revision string alone is never treated as proof of content. Any scientific
change requires a reviewed revision bump and a matching update to the closed
expected-value table.

Comparator linkage to the method registry uses a closed, non-digest execution
projection: method ID, execution scope, integration status, adapter key,
environment identifier, declared source reference, and resource class. The
projection is embedded in the plan and compared field by field with the loaded
registry. Comparator code does not consume registry content-digest fields.

## Typed run identity

Every comparator attempt carries one immutable `ComparatorRunIdentity` with:

- workflow schema and authority revision;
- method ID and full method execution projection;
- configuration ID and full typed payload;
- stage and entry kind;
- dataset, mechanism, view, and biological-draw identifiers;
- mask seed, model seed, and draw index; and
- the scheduled entry ordinal.

The identity is serialized as a closed JSON object and compared directly.
There is no separate request, configuration, method, source, entry, or plan
digest. A request resolves exactly one authority row by method ID,
configuration ID, and full payload equality before dispatch. The adapter
receives only the decoded authoritative payload. Its effective configuration is
re-encoded after the attempt and must equal the request payload exactly.

Unknown configuration IDs, duplicate matches, registry-default fallback,
payload patches, and caller-supplied relabeling remain errors.

## Plan and checkpoint model

Fair-comparator plans use `identity_mode = "direct-v1"`. The plan stores its
complete ordered entry projection rather than a plan digest. The corresponding
checkpoint and transaction-intent schemas contain:

- the identity mode and authority revision;
- the complete ordered plan snapshot;
- explicit prepared-input descriptors;
- the contiguous record prefix;
- the single replayed budget ledger; and
- the comparator-selection completeness status derived from records.

The legacy digest-based checkpoint schema remains available only for legacy
plans. Construction of a direct-identity plan must not eagerly calculate legacy
identity fields. `CheckpointStore` selects exactly one schema from the plan's
identity mode and rejects mixed fields.

On load, the stored plan snapshot must equal the current plan snapshot exactly.
Every record must equal the corresponding plan entry on all identity fields,
and the records must form a non-duplicated contiguous prefix. Excess records,
reordered rows, payload drift, dataset drift, or caller-supplied completeness
state fail closed.

Task 7's budget rules remain binding:

- candidate-search and comparator-tuning entries count toward configuration
  limits;
- comparator configurations share one method-level budget scope;
- MaskImpute retains separate method-and-kind scopes;
- the stored budget mapping must equal central replay exactly;
- durable records are never selectively retried; and
- only unfinished transaction intent may recover.

Comparator-selection completeness remains derived solely from the full
comparator-tuning denominator. `failed`, `timeout`, `resource_exceeded`, and
`unavailable` are intrinsic terminal outcomes. `budget_exhausted`,
`blocked_authority`, and `infrastructure_error` block completeness. Only a grid
whose rows are all completed or intrinsic terminal is selectable.

## Prepared-input descriptors

Every prepared development input has an explicit descriptor containing:

- dataset and source/accession identifiers;
- preprocessing contract revision;
- matrix shape and dtype;
- ordered cell IDs and gene IDs;
- ordered batch labels;
- total count, nonzero count, minimum, and maximum; and
- the applicable mechanism, mask seed, and view identifiers.

The descriptor is stored once per prepared input and compared field by field on
resume. Ordered axis IDs are not replaced with a summary value. Count summaries
are validation invariants, not content identities.

Within one execution, every method/configuration for a scheduled input receives
the same immutable prepared object. Across a resume, the input is
deterministically re-prepared from the tracked source reference and must match
the complete descriptor. This design deliberately makes no byte-identity claim
for a large matrix without storing a second matrix copy.

## Logs, smoke, and selection receipts

Development log receipts retain only canonical stream name, original byte
count, capture policy, and terminal reason. Raw stream content and content
summaries are not retained.

The smoke receipt retains authority revision, all 34 full comparator identities,
status/reason, runtime, measured resource peaks and provenance, and the fixed
projection multiplier. Scientific matrices, evaluator targets, metrics, and
adapter outputs remain discarded. All configurations must complete.

The comparator-selection receipt contains:

- schema and authority revision;
- the complete scheduled tuning denominator;
- every terminal record needed for collapse and ranking;
- per-method collapsed units, Pareto membership, rank tuple, and tie-break
  values;
- one selected configuration ID and full payload per selectable method; and
- the record-derived readiness decision and complete status counts.

Receipt validation recomputes collapse, Pareto membership, ranks, selections,
and readiness from the embedded records, then compares the actual values
directly. Create-only and concurrent-identical publication compare complete
canonical bytes directly; conflicting existing bytes are preserved and
rejected.

## Error handling

All direct-identity parsers use closed schemas and normalized domain errors.
They reject mixed old/new schemas, omitted fields, extra fields, duplicate JSON
keys, invalid primitive types, nonfinite values, ambiguous numeric encodings,
unsafe paths, and inconsistent caller-supplied derived values.

No failed validation deletes or rewrites existing evidence. No loader repairs a
superseded artifact. A mismatch requires a fresh checkpoint or receipt after a
reviewed authority amendment.

## Verification

Implementation is test-driven. The first regressions must fail against the
current digest-based fair-comparator implementation and cover:

1. the tracked authority and comparator sections of tracked contracts contain
   none of the forbidden field-name tokens;
2. fair-comparator authority, plan, request, checkpoint, intent, log, smoke,
   and selection entry points do not call content-digest helpers;
3. all 34 payloads decode and re-encode to directly equal typed values;
4. payload mutation, signed negative zero, order changes, configuration
   relabeling, and method-projection drift are rejected without a digest field;
5. direct-identity checkpoint mutation, record reordering, excess records,
   input-descriptor drift, and budget-ledger drift are rejected;
6. the exact Task 7 terminal/blocking status partition and resume behavior stay
   green;
7. smoke and selection receipts contain full identities and recompute every
   derived decision;
8. old fair-comparator schemas and mixed-mode artifacts are rejected rather
   than migrated; and
9. legacy non-comparator tests demonstrate that their existing provenance and
   checkpoint modes remain unchanged.

Static checks must be scoped to the fair-comparator module, comparator-specific
schema sections, and comparator branches of shared modules. They must not force
an unrelated repository-wide provenance rewrite.

No real comparator, smoke, tuning, evaluator, competition, or final workload is
run while implementing this migration. Verification uses fixtures, fake
executors, and synthetic artifacts until all plan tasks and publication gates
authorize scientific execution.

## Migration sequence

1. Amend the governing fair-comparator design and implementation plan to cite
   this decision and replace their digest-based requirements.
2. Convert the tracked comparator authority and its two tracked references to
   direct identity.
3. Convert typed decoding, method linkage, requests, and all 34 configuration
   paths.
4. Add the direct plan/checkpoint mode and convert the 2,896-row development
   plan, retaining Task 7's exact replay logic.
5. Convert log, smoke, selection, development-evaluation, and selected-map
   handoff schemas.
6. Remove superseded comparator aliases and prove legacy modes are unchanged.
7. Independently review the complete migration before continuing the remaining
   fair-comparator tasks.

The existing Task 7 commit remains useful for its scientific budget and
completeness behavior, but it is not accepted as the final Task 7 boundary until
the direct-identity migration is applied and independently re-reviewed.

## Acceptance criteria

The migration is complete when the fair-comparator segment contains no content
digest computation or schema fields, all comparator identity checks use closed
typed values or direct canonical-byte equality, the exact 34-configuration and
2,896-row authorities remain unchanged scientifically, Task 7 budget replay is
preserved, old comparator artifacts fail closed, adjacent legacy modes pass
unchanged, and an independent exact-commit review has no Critical or Important
findings.
