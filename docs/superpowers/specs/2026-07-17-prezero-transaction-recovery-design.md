# Pre-zero Provenance and Development Transaction Recovery Design

## Purpose

Close two fail-open persistence paths in realized `p_pre_zero` evidence. A
conversion-terminal MaskImpute run must not be able to lose an already-produced
score artifact through a coherently rehashed record, and a development crash
must not leave immutable artifacts that block a scientifically equivalent retry
with different diagnostic logs.

## Provenance-derived score requirement

Persisted score presence is not authority for whether a score should exist.
`CheckpointStore` will derive a trusted `expected_matrix_present` value from the
plan entry and stored execution provenance. A score is mandatory only when the
entry is MaskImpute, the frozen configuration requires the count score, and at
least one execution fact proves the adapter ran far enough to realize it:

- the run completed;
- native-output provenance is present; or
- an unavailable run has the canonical
  `evaluator_conversion_<category>_detail_<sha256>` disposition emitted only
  after a completed adapter execution.

The expected value, rather than `evidence.matrix.shape`, controls exact score
authority derivation. `validate_stored_prezero_evidence` will require observed
and expected matrix presence to agree before validating matrix, policy,
storage, and reports. This preserves legitimate pre-execution terminal rows
without matrices while rejecting coordinated removal from conversion-terminal
rows. Development and final storage share the same validator, so both use the
same authority rule.

## Development transaction protocol

The existing temporary store remains the pre-publication semantic validation
boundary. After validation, `CheckpointStore.append` will recover any earlier
interrupted transaction, publish a canonical transaction intent, publish the
flat immutable run artifacts, atomically replace `checkpoint.json`, and remove
the intent only after the checkpoint is durable.

Each intent is bound to the plan checksum, plan ordinal, run ID, checkpoint
position, and the exact closed set of possible artifact paths for that attempt.
The intent is published before the first artifact. Therefore interruption after
stdout, stderr, native output, evaluator output, or `p_pre_zero` output always
leaves enough information for deterministic recovery.

Recovery reads and validates the canonical checkpoint prefix. If the intended
record is committed with the matching run ID, recovery retains its artifacts
and only closes the intent. Otherwise it proves that every candidate is absent
from all committed artifact receipts, re-reads the unchanged checkpoint prefix,
and removes only existing regular, nonsymlink, current-user-owned files with a
single hard link. Unknown intent fields, unsafe paths, noncanonical JSON,
changed checkpoint bytes, linked files, or committed references fail closed.
Empty transaction and run directories may be removed after their contents are
closed.

No artifact is overwritten during recovery or retry. A retry may change stdout
or stderr while retaining identical scientific outputs because recovery first
removes only the uncommitted prior attempt's artifacts.

## Failure handling

Invalid attempts fail during temporary validation and create no final output.
Publication-time exceptions leave the intent and any closed artifacts for the
next append to recover. A crash after checkpoint publication but before intent
completion is treated as committed: artifacts remain, the intent is removed,
and execution continues from the canonical checkpoint prefix.

Final-result transaction behavior is unchanged. The shared score-presence
contract only strengthens final record loading.

## Verification

Regression tests will:

- coordinately remove matrix, policy, storage, and score reports from real
  conversion-terminal development and final records, rebind all mutable
  checksums, and require rejection;
- interrupt development publication after each artifact boundary;
- construct a new store, retry identical scientific output with different
  stdout/stderr, and require successful checkpointing with no stale intent or
  orphan artifact; and
- retain the existing exact matrix, policy, report, cache-isolation, invalid
  append, and final transaction-recovery coverage.
