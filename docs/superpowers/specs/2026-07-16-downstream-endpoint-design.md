# Evaluator-Only Downstream Endpoint Design

**Status:** Approved implementation design

## Boundary

Downstream evaluation starts only after a method has returned an imputed
cell-by-gene matrix on the persisted evaluator scale, `log2(CP10k + 1)`.
`MethodOutput` contains only that matrix and stable cell and gene identifiers;
the evaluator does not renormalize it. Group labels, group-specific marker
truth, held-out counts, pseudotime, and the trajectory root live in a separate
`EvaluatorTargets` object. The evaluator aligns these objects by stable IDs;
it never invokes a method and cannot add evaluator fields to method input.

## Complete endpoint schema

Every evaluation returns exactly these eight rows, in this order:

1. `marker_rank_loss`
2. `clustering_ari_loss`
3. `clustering_nmi_loss`
4. `positive_de_marker_recall`
5. `positive_de_false_discovery_rate`
6. `heldout_gene_profile_rank_loss`
7. `heldout_cell_profile_rank_loss`
8. `trajectory_pseudotime_rank_loss`

Each row records its direction, `completed` or `unavailable` status, an
explicit reason for unavailable values, one independent biological-draw unit,
and a descriptive denominator with its unit. Cell, gene, marker, group, and
hypothesis counts are descriptive denominators, never independent replicates.

## Estimands

- Marker rank loss is the macro-average over groups of the mean normalized,
  tie-aware rank of that group's true markers. Genes are ranked by the
  one-vs-rest difference in mean `log2(CP10k + 1)` expression. Zero is best.
- Clustering uses stable-ID ordering, deterministic full-SVD PCA, and
  deterministic k-means with seed `20260716`. Candidate cluster counts are
  the truth-free fixed grid 2--10, restricted only by cell and distinct-profile
  counts; minimum Davies--Bouldin index selects the model with smaller `k` as
  the tie-break. It returns `1 - ARI` and `1 - NMI`; group labels are accepted
  only after model selection and are used only for scoring.
- Positive-control differential expression uses one-sided Welch tests for
  every group-by-gene hypothesis. Equal constants receive p=1 and separated
  constants p=0. Benjamini-Hochberg correction is one global family containing
  all `number of groups * number of genes` hypotheses in the biological draw.
  Recall uses all group-specific true-marker hypotheses; false-discovery rate
  uses the BH discoveries at alpha 0.05.
- Held-out rank losses compare independent count-split profiles. The gene
  endpoint averages Spearman correlations across cells for genes variable in
  the held-out split; the cell endpoint averages correlations across genes for
  cells variable in the held-out split. A method-collapsed profile receives
  correlation zero instead of becoming outcome-dependently unavailable.
- Trajectory loss is `1 - Spearman rho` between genuine evaluator pseudotime
  and a root-oriented multiscale diffusion distance. The deterministic graph
  is built from stable-ID-ordered common-scale full-SVD coordinates. Exact
  k-nearest-neighbor distances are computed in fixed row blocks and retained
  only as a sparse symmetric graph, so the 2,700-cell panel never constructs
  a dense cell-by-cell matrix. A validated evaluator-known root must be the
  unique minimum of genuine pseudotime. Group labels are never converted into
  pseudotime. Clustering and trajectory share the same within-denominator SVD.

## Simulator adapter audit

The existing development outputs were inspected directly. SymSim, SERGIO,
SPARSim, and semisynthetic outputs all contain evaluator-only `obs.group`.
SymSim, SERGIO, and SPARSim contain group-specific Boolean marker columns;
semisynthetic outputs do not. Only semisynthetic outputs contain
`layers["heldout_counts"]`. None of the four outputs contains genuine
`obs.pseudotime`, so trajectory evaluation must be unavailable with
`genuine_pseudotime_not_available_in_simulator_output`. The separately tracked
`study/trajectory_panel.json` registers a 2,700-cell deterministic synthetic
panel with exact latent pseudotime, a prespecified unique root, a mechanism
outside the four reconstruction mechanisms, and bound authority and semantic
dataset hashes. Its pseudotime and group fields are absent from method views.

## Production evidence

`downstream_evidence.py` consumes the sealed development checkpoint or final
execution manifest introduced by the score-evidence stage. It validates raw
little-endian development outputs and bounded `zlib_raw_f64_v1` final outputs,
including file, uncompressed, and runner content receipts. A plan binds every
source record, evaluator artifact, raw and semantic dataset hash, retained cell
identity, gene identity, biological draw, technical view, and model seed.

Each source denominator produces one immutable canonical record containing
exactly eight endpoint rows. Noncompleted upstream runs retain their original
status and reason on all eight rows and use the fixed
`upstream_run_not_completed` reason code. Record prefixes are resumable;
resume and completed-manifest validation re-read all source and dataset
bindings. Development selection schema 4 requires the resulting manifest to
cover exactly the reconstruction-selection denominator before selection can
proceed; schemas 2 and 3 remain readable as pre-downstream legacy artifacts.

When v28 or v29 is activated, development evidence is one ordered bundle over
the base checkpoint and every consecutive activated revision checkpoint.
Revision checkpoints repeat comparators and the capacity control, so the
bundle retains the base selection-primary denominator once and adds only each
revision's own candidate rows. Each source separately binds checkpoint file
and payload hashes, rebuilt plan and input hashes, exact run statuses,
denominator identity, and the corresponding evaluator-manifest reconstruction
object. Schema-4 validation cross-checks all of those source bindings against
the independently rebuilt revision evaluation; metric-level `unavailable`
rows therefore cannot be mistaken for a failed completed run. Fixed outputs
are `downstream`, `downstream-v28`, and `downstream-v29`, with the runner
selecting the latest present fixed combined revision input and refusing an
invalid latest input rather than falling back.

## Verification

Focused tests cover hand-calculated values, stable-ID permutation invariance,
truth-free clustering model selection under changed truth labels and group
counts, deterministic single-SVD reuse, the 2,700-cell block bound, every
missing/degenerate reason path, numerical terminal rows, global BH,
method collapse, genuine trajectory authority, development/final storage
contracts, tamper detection, resume, fixed eight-row failure preservation, and
schema-4 selection completeness.
