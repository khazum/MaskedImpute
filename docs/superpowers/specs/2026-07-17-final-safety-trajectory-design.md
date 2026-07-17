# Frozen Trajectory and Final Null-DE Evidence Design

**Status:** Binding implementation companion to the Genome Biology study design

## Purpose

Close the two remaining scientific-evidence gaps without weakening the untouched
final-holdout contract:

1. execute the already registered exact-latent trajectory panel with the frozen
   method and frozen comparator configurations; and
2. evaluate null differential-expression false-positive rates on every frozen
   final reconstruction output.

These are safety and utility analyses. They do not reopen development, alter the
four-mechanism reconstruction denominator, or create additional independent
replicates for the primary comparison.

## Trajectory boundary

The registered synthetic trajectory panel is known before method freeze and is
therefore supplementary, not an unseen final reconstruction dataset. It remains
one independent biological draw. Its result may support a descriptive
trajectory-recovery statement only; it cannot support population-level
generalization or an inferential superiority claim.

Trajectory method execution occurs only after the candidate, comparator
versions, configurations, calibration, method registry, runtime lock, and final
execution claim are frozen. It occurs inside the claimed final round and before
the one-use evaluation receipt is issued. The receipt must therefore bind every
trajectory plan, record, log, compressed evaluator output, and manifest byte.
The trajectory panel is never added to development selection or to the 40-dataset
primary final denominator.

The exact registered trajectory authority and regenerated semantic dataset hash
are revalidated before planning and again on resume. `pseudotime`, the root,
group labels, and all other evaluator targets are removed before constructing
the method input. Every method receives the same retained cells, genes, observed
integer counts, normalization contract, and truth-free covariates.

The supplementary execution denominator is derived from the frozen method
receipt rather than accepted from command-line configuration. It uses the same
method applicability rules, selected MaskImpute configuration, capacity-matched
control, comparator defaults, runtime environments, time/resource limits, and
three nested seeds for stochastic methods as the primary final plan. Methods
requiring an unavailable matched bulk reference retain an explicit non-run row.
All failures, timeouts, resource exceedances, and unavailable methods remain in
the denominator.

MaskImpute fits its count score on the truth-free trajectory counts and applies
the retained all-development calibrator. This is external inference, never a
development holdout fold. The score policy must explicitly name the registered
trajectory authority and be independently rederived on resume.

Trajectory files live under
`results/trajectory/execution/` within the claimed round. Publication and resume
use the same atomic, immutable record and compressed-output contract as the
primary final store. A trajectory execution manifest records its supplementary
scope, registered authority hashes, frozen primary-plan hash, execution claim,
runtime/environment authority, exact method denominator, all terminal statuses,
and all result-file receipts. The primary final evaluation receipt names both
the primary execution manifest and this trajectory manifest.

After the round is evaluated, the evaluator-only downstream stage revalidates
the receipt, source manifest, every source record/output, and the registered
trajectory dataset, then deterministically replays the fixed trajectory
endpoint. Its external output is namespaced by round ID and evaluation-receipt
payload hash. Loading the evidence replays the endpoint again; coordinated
replacement of an output and its local hashes is not accepted.

## Frozen-final null-DE boundary

Null-DE evidence is a read-only post-evaluation analysis of the 40 primary final
datasets. It never mutates the evaluated round. Its source is the same exact
evaluated-round binding used by final downstream evidence: repository identity,
round ID, freeze/materialization/claim/receipt hashes, primary execution
manifest, exact dataset bindings, and all record/output receipts.

For each final dataset, the evaluator derives one deterministic balanced split
from:

- the evaluation-receipt payload hash;
- the source dataset semantic hash and stable cell IDs;
- the mechanism, biological draw, and technical view; and
- the fixed algorithm identifier `final-null-de-v1`.

The split is stratified by evaluator-only group labels and is identical for all
methods and model seeds on that dataset. Split assignments never depend on a
method output. The fixed testable-gene mask is derived once from observed counts
using the existing `fixed_null_de_gene_mask` procedure and is reused for observed
counts and every method. The evaluator applies the existing stratum-adjusted,
two-sided OLS null-DE test and reports gene-level FPR at nominal alpha 0.05.

Every primary final source record produces exactly one null-DE row. Completed
outputs are decoded and revalidated before evaluation. Upstream terminal states
produce a reason-coded terminal row with no numeric value. A dataset whose
prespecified split or fixed gene denominator is mathematically unavailable also
retains a reason-coded row; it is never silently dropped.

The immutable external evidence plan and manifest bind the evaluated round, the
exact primary downstream/source checkpoint identity, evaluator source hash,
split/gene-mask hashes, every record file, and the complete denominator. Resume
accepts only a validated record prefix. Loading a completed manifest re-reads
the primary final source and independently recomputes each split, gene mask, and
FPR before accepting stored rows.

## Analysis and claims

Final null-DE summaries first average model seeds within a biological draw and
paired technical views within that draw. Biological draws are the independent
units. Report MaskImpute's maximum and draw-level distribution of FPR, its
paired difference from observed counts, and all unavailable/failure counts.

The prespecified safety gate remains:

- MaskImpute FPR is at most 0.06; and
- it is no more than 0.01 above observed counts.

The publication synthesis must mark the safety gate unavailable or failed when
the exact denominator is incomplete. It may use the word `competitive` only
when the frozen reconstruction claim gate and every safety gate, including
final null-DE, pass. The single trajectory draw is always labeled descriptive.
No trajectory result can rescue a failed reconstruction or safety gate.

## Required verification

Tests must cover:

- truth-free trajectory method views and exact registered authority replay;
- frozen configuration/applicability/seed derivation with no user overrides;
- primary and trajectory output-directory separation;
- interrupted trajectory resume and receipt coverage of every result byte;
- rejection of changed trajectory authority, dataset, plan, record, output,
  runtime binding, or score/calibration policy;
- final null-DE split equality across methods/seeds and stable-ID permutation;
- fixed observed-count gene masks and rejection of output-dependent masks;
- exact 40-dataset primary source coverage and one row per source run;
- reason-coded upstream and mathematically unavailable rows;
- resume-prefix validation and deterministic replay on completed load;
- rejection of symlinked paths and outputs inside the frozen repository for the
  external null-DE stage; and
- conservative claim-gate behavior for pass, fail, and incomplete evidence.

No test fixture may weaken the production final binding. Explicit test-only
source kinds must be impossible to select through production entry points.
