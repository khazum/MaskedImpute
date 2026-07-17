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
the larger sizes. This exclusion is fixed before execution and is not based on
results.

Every method-size row retains runtime, peak process-tree RSS, peak GPU memory,
terminal status, reason, exact logs, output hashes, and the complete bounded
accuracy metric denominator. Timeouts, resource failures, and unavailable runs
remain in the result set.

## Storage and resume policy

Method output matrices are hashed and evaluated but never persisted. Only the
moderate H5AD input is retained. The paired severe H5AD and native simulator
files are hashed, receipted, and then deleted. A canonical checkpoint binds a
strict plan prefix, dataset receipts, logs, metrics, code, frozen method,
runtime, and tracked authorities. Resume refuses changed bytes or a changed
implementation hash.
