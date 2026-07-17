# External-reference development evidence design

## Purpose

Replace the self-asserted external-reference checkpoint with producer-owned,
byte-verifiable evidence from the pinned D3Impute and scTsI adapters on the
canonical Tung iPSC single-cell and matched measured-bulk source. This track is
development-only. Its bulk reference must never be exposed to MaskImpute or to
the same-input competition.

## Fixed scientific design

- Methods: `d3impute` and `sctsi`, selected from `study/methods.json` only.
- Dataset: `tung-ipsc-ercc-bulk-replicates`, freshly prepared through
  `prepare_real_orthogonal_panel`.
- Input: the prepared Tung single-cell `MethodInput`.
- Reference: all measured Tung bulk-per-sample profiles, aligned gene-for-gene,
  with raw counts and the source-file SHA-256 bound into each adapter's native
  matched-bulk reference object.
- Adapters and parameters: the existing pinned adapter entry points and their
  default fixed configurations. There are no CLI switches for methods, data,
  reference, seeds, parameters, output location, or evaluation endpoints.
- Endpoints: the exact nonempty `tung_concordance_units` records for ERCC
  recovery, technical-replicate concordance, and bulk-pseudobulk concordance.
  Because the two bulk-derived endpoints reuse the matched-bulk adapter input,
  every endpoint record labels that overlap. Technical-replicate concordance
  is the only endpoint that does not reuse the matched-bulk adapter input, but
  it still uses technical lanes from the same experiment and pseudobulk
  weighted by the observed single-cell library sizes. It is therefore not an
  independent validation cohort. All three endpoints are descriptive same-
  experiment checks, not independent efficacy validation.

The only operational CLI locators are one absolute, non-symlink executable per
fixed method and one absolute, non-symlink scTsI R-library directory. Duplicate
or unknown locators are rejected.

## Producer and immutable artifact

The focused `external_reference_development` module requires both registry
environments to be final `ready` dispositions bound to the exact runtime lock,
then validates source, executable, R-library, and source-checkout authority
before any attempt. It publishes immutable input metadata and matrices before
invoking an adapter. Runtime closure and pristine source are revalidated
immediately before and after every attempt.

Each completed attempt stores native output, independently converted evaluator
output, stdout, stderr, environment receipt, compatibility disclosures, and
the three endpoint-unit tables. A known adapter-unavailable outcome stores its
terminal reason code, a separately hashed detail, command/log evidence, and an
explicit unavailable disposition for all three endpoints. Runtime/source drift
or an unexpected exception aborts without a completed checkpoint. Existing
output is never overwritten.

The canonical checkpoint is written last and binds the plan, method registry,
runtime lock, source receipts, operational locator identities, all persisted
files, and all records.

## Validator and publication freeze

The public loader does not trust checkpoint metrics. It freshly reloads the
registry, runtime lock, source checkouts, and canonical Tung panel; opens every
artifact as a unique regular non-symlink file; verifies byte sizes and SHA-256
digests; reconstructs both adapter references; reconstructs output snapshots;
reruns native-to-evaluator conversion; and recomputes every Tung endpoint unit.
It rejects absent/empty metrics, extra files, hand-authored checkpoints,
reference/input/output/log tampering, locator or runtime mismatch, and source
drift.

Publication freeze calls only this production loader. It binds external
eligibility to the canonical Tung source ID rather than the 16 simulated
same-input dataset IDs. Structured adapter exception details are reduced to an
existing safe terminal reason code plus a separately persisted detail hash;
the frozen integration-reason grammar remains strict.

The populated publication worktree necessarily retains ignored execution
assets. Freeze permits only the existing roots `artifacts/envs`,
`artifacts/external`, `artifacts/method-sources`, and
`artifacts/study/development`. Every permitted root is enumerated as a closed
tree and its paths, entry types, modes, file bytes, and symlink targets are
bound by a SHA-256 receipt in `freeze.json`; the receipt is recomputed at every
round transition. Ignored files outside those roots remain fatal. In
particular, `__pycache__`, `.pytest_cache`, and similar caches must be removed
before freeze, and publication commands should run with bytecode writing
disabled.
