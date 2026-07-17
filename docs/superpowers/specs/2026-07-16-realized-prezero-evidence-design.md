# Realized p_pre_zero Evidence Design

## Scope

This supplement implements publication-design items 4 and 5 from
`2026-07-12-genome-biology-study-design.md`. It does not run development or
final data and does not create numerical publication claims.

## Decision

The realized `p_pre_zero` matrix emitted by a completed MaskImpute adapter is
the sole score-evidence authority. The evaluator copies that matrix from the
`AdapterExecution` boundary and never reconstructs it from count-score,
calibration, diagnostic, or execution-request receipts. Receipts are used only
to validate and bind the score/calibration policy that produced the matrix.

Two alternatives are rejected:

1. Storing only derived metrics prevents independent recomputation and cannot
   detect a matrix/metric mismatch.
2. Recomputing the score from a receipt can differ from the probability used by
   the fitted decoder and violates the realized-evidence requirement.

## Evidence contract

Every evaluated attempt has exactly one `p_pre_zero_evidence` record.

- A completed `method_id == "maskimpute"` attempt must expose a finite
  float64 matrix in `[0, 1]` aligned to the evaluator's retained cell and gene
  identities. Missing or malformed evidence fails closed.
- Every other method has an explicit `not_applicable` evidence row with reason
  `method_does_not_emit_p_pre_zero`.
- Every noncompleted MaskImpute attempt retains its run status and reason in an
  explicit unavailable evidence row. It is never omitted from the denominator.
- The semantic matrix digest domain-separates and binds raw content, shape,
  dtype, run ID, method, dataset ID and checksum, mechanism, biological draw,
  technical view, model seed, configuration ID/checksum, method-input checksum,
  retained-cell checksum, and score-policy checksum.
- The policy record binds the realized score source, score artifact/input/config
  checksums, calibration artifact and algorithm, calibration scope, and stated
  equivalence reason. Development LODO and final retained-calibration receipts
  remain independently validated by their existing authorities.

## Metric contract

The evaluator emits one overall observed-zero record, four tie-preserving
library-size strata, and four fixed truth-expression strata. Each record has
machine-readable AUROC, AUPRC, Brier score, log loss, calibration intercept,
calibration slope, and ECE fields plus deterministic reliability bins.

Only `exact_pre_capture` truth produces score estimates. `exact_continuous`
truth emits `undefined_for_continuous_truth`; `proxy_high_depth` emits
`proxy_truth_not_exact`; and orthogonal truth emits `truth_unavailable`.
Unavailable/nonapplicable attempts preserve the observed-zero denominator and
their terminal reason in every metric field.

## Persistence and validation

Development checkpoints and final records store the matrix as deterministic
zlib-compressed little-endian raw float64 bytes. A receipt records the encoding,
shape, dtype, compressed SHA-256 and byte count, uncompressed SHA-256 and byte
count, semantic matrix digest, and evidence payload digest. Load/resume uses
bounded decompression and rejects oversized input, trailing streams, truncated
streams, nonfinite/out-of-range values, checksum drift, policy drift, metric
drift, identity drift, and partial bindings.

Final transaction intents include the score artifact. Final disk preflight adds
one zlib `compressBound` allowance for each remaining executable MaskImpute run,
without success-conditioned exclusions. Failure rows require no score matrix but
remain in the result denominator.

## Verification

Tests prove adapter/evaluator propagation, explicit non-MaskImpute and failure
rows, all four truth mechanisms, deterministic metrics and reliability bins,
development fresh/resume/tamper behavior, final fresh/resume/tamper/zip-bomb
behavior, and final disk-preflight accounting. Existing final failure-denominator
and execution-request integrity tests must remain green.
