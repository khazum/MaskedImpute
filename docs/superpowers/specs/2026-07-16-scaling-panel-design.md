# Publication Scaling Panel Design

**Status:** Binding implementation companion to the Genome Biology study design

## Purpose

The scaling panel measures whether the frozen method and representative learned
comparators remain executable as cell count grows. It is not an additional
biological-replicate panel and cannot affect model selection.

## Closed design

- Mechanism/view: SymSim, moderate capture condition.
- Sizes: 10,000, 25,000, 50,000, and 100,000 cells.
- Genes: 500 at every size.
- Methods: observed counts, frozen MaskImpute, DCA, scVI, and MAGIC.
- Seed policy: one domain-separated biological/measurement seed triple per size;
  stochastic methods use model seed 42. This quantifies computational scaling,
  not seed or biological uncertainty.
- The candidate configuration is loaded only from the committed frozen-method
  receipt. Comparator defaults, runtime lock, score/calibration authority,
  implementation source hash, and the tracked scaling contract are plan-bound.

## Accuracy and resource endpoints

All four truth matrices fit the 48 GiB evaluator budget. Accuracy therefore uses
all four sizes, but only a bounded metric implementation: overall, induced-
dropout, exact pre-capture-zero, and observed-nonzero MSE; gNRMSE; mean and
variance distortion; per-gene empirical Wasserstein distance; and gene-gene
correlation distortion. Cell-cell correlations and pairwise cell distances are
deliberately excluded because their dense quadratic implementation does not fit
the larger sizes. Realized `p_pre_zero` score matrices and their calibration
analysis are also excluded: they are retained and evaluated in the main final
panel, while duplicating them here would violate the bounded scaling-storage
contract. These exclusions are fixed before execution and are not based on
results.

Every method-size row retains runtime, peak process-tree RSS, peak GPU memory,
terminal status, reason, exact logs, output hashes, and the complete bounded
accuracy metric denominator. The method registry's timeout, RSS, and GPU
ceilings and the parent-side Linux process-tree/nvidia-smi measurement labels
are copied into each hash-bound plan entry. Completed rows cannot exceed those
ceilings or change those labels; timeout and resource-exceeded rows must have
the corresponding reason and measurements. A resource-exceeded reason must
name a ceiling that was actually crossed; the other ceiling may have been
crossed at the same time. Timeouts, resource failures, and unavailable runs
remain in the result set.

## Storage and resume policy

Every completed run retains both its native method matrix and its evaluator
matrix as canonical little-endian float64 bytes compressed with zlib level 6.
Their shapes, scales, encodings, compressed and uncompressed byte counts, and
byte hashes are receipted. An independently serialized executor receipt binds
the native snapshot identity, terminal status and reason, runtime and resource
telemetry, and both log identities before evaluator conversion occurs.

The complete panel has a fixed raw-matrix evidence bound of 7,400,000,000 bytes
(6.89 GiB): 3.7 GB each for native and evaluator matrices (five methods times
185,000 cells times 500 genes times eight bytes). Each compressed matrix is
individually capped by zlib's documented bound, giving a panel-wide compressed
matrix ceiling of 7,402,258,990 bytes. Excluding the separately retained
moderate H5AD inputs, all per-run files have an aggregate ceiling of
10,128,556,590 bytes (9.43 GiB): the compressed matrix ceiling plus forty
64-MiB logs and twenty 2-MiB executor receipts. Only the moderate H5AD input is
retained. The paired severe H5AD and native simulator files are hashed,
receipted, and then deleted.

Each run is first written into a private staging directory. The complete set of
files is validated there, the directory is atomically renamed into its final
run path, and only then is the checkpoint replaced. A retry removes only a
closed, allow-listed orphan transaction left by a crash; malformed or symlinked
paths are rejected. Every checkpoint and artifact path component must remain
inside the output root and must not be a symbolic link. A canonical checkpoint
binds a strict plan prefix, dataset receipts, logs, metrics, code, frozen
method, runtime, and tracked authorities. Resume refuses changed bytes or a
changed implementation hash.

On each fresh process resume, and once more before terminal publication, the
store derives the expected seed triple, ephemeral protocol, dataset ID,
independent-unit ID, path, and design digest from the tracked scaling contract
and study protocol. It independently reruns the deterministic tracked SymSim
generator in a temporary root and requires exact equality of the moderate and
severe semantic hashes, truth hash, provenance/configuration embedded in those
semantic hashes, and both native-manifest/file-inventory receipts. It then
reopens each retained moderate H5AD once, recomputes its semantic and truth
hashes and QC/input identities, decodes each completed evaluator artifact,
and recomputes all ten metric rows. Before metric replay, it also decodes the
native artifact, rebuilds the method-output snapshot, reruns the tracked output
converter, and requires byte-exact equality with the retained evaluator
matrix. All replayed values must equal the checkpoint exactly.

After that full validation, appends advance an in-memory immutable checkpoint
snapshot transactionally: only the new receipt or run and its artifacts are
validated before the canonical checkpoint is replaced. Historical H5ADs are
not recursively reopened or rehashed on every append. Before an append, a
compare-before-replace check requires the on-disk checkpoint bytes to equal the
version that populated the cache. Returned snapshots are detached deep copies,
so caller mutation cannot alter cached authority. Self-consistent checkpoint
hashes alone are not evidence of a valid scaling row.
