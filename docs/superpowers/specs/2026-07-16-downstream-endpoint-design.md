# Evaluator-Only Downstream Endpoint Design

**Status:** Approved implementation design

## Boundary

Downstream evaluation starts only after a method has returned an imputed
cell-by-gene matrix. `MethodOutput` contains only the matrix and stable cell
and gene identifiers. Group labels, group-specific marker truth, held-out
counts, pseudotime, and the trajectory root live in a separate
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
  one-vs-rest difference in mean log1p-CP10k expression. Zero is best.
- Clustering uses stable-ID ordering, log1p-CP10k, deterministic full-SVD PCA,
  and deterministic k-means with seed `20260716`. It returns `1 - ARI` and
  `1 - NMI`; group labels are used only to score the completed clustering.
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
  is built from stable-ID-ordered log1p-CP10k and full-SVD coordinates. A
  validated evaluator-known root must be the unique minimum of genuine
  pseudotime. Group labels are never converted into pseudotime.

## Simulator adapter audit

The existing development outputs were inspected directly. SymSim, SERGIO,
SPARSim, and semisynthetic outputs all contain evaluator-only `obs.group`.
SymSim, SERGIO, and SPARSim contain group-specific Boolean marker columns;
semisynthetic outputs do not. Only semisynthetic outputs contain
`layers["heldout_counts"]`. None of the four outputs contains genuine
`obs.pseudotime`, so trajectory evaluation must be unavailable with
`genuine_pseudotime_not_available_in_simulator_output`. A separate validated
trajectory-truth constructor covers future genuine trajectory datasets.

## Verification

Focused tests cover hand-calculated values, stable-ID permutation invariance,
determinism, every missing/degenerate reason path, fixed-schema completeness,
the global BH denominator, method collapse, genuine trajectory validation,
and structural separation of method output from evaluator-only truth.
