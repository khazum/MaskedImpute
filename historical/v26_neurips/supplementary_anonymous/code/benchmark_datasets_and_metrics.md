# Synthetic Benchmark Datasets and Evaluation Metrics

This note summarizes the synthetic datasets and evaluation metrics used to compare imputation/denoising approaches.

## 1) Available datasets

All datasets are generated with Splatter (Splat `groups` and `paths`) and saved as `SingleCellExperiment` RDS files via:

synthetic_datasets/generate_simulated_benchmark.R`

### Global simulation settings

- Genes: `G = 1000`
- Tuning split size: `N = 1000` cells
- Test split size: `N = 5000` cells
- Scaling split sizes: `N ∈ {10000, 20000, ..., 100000}`
- One deterministic dataset per scenario/split (fixed seeds)

### Dataset layout

- Tuning: `simulated_data/tune/<scenario_id>/sce.rds`
- Test: `simulated_data/test/<scenario_id>/sce.rds`
- Scaling: `simulated_data/scale/groups_balanced_moderate_drop/n<cells>/sce.rds`

### Scenario catalog (13 total)

1. `groups_balanced_nodrop` (NB-only baseline; no explicit dropout)
2. `groups_balanced_moderate_drop`
3. `groups_balanced_moderate_drop_group`
4. `groups_imbalanced_moderate_drop`
5. `groups_imbalanced_moderate_drop_group`
6. `groups_rare_high_drop`
7. `groups_rare_high_drop_group`
8. `batch_effects_moderate_drop`
9. `batch_effects_moderate_drop_group`
10. `paths_linear_moderate_drop`
11. `paths_linear_moderate_drop_group`
12. `paths_branching_moderate_drop`
13. `paths_branching_moderate_drop_group`

Interpretation:

- One NB-only baseline + six scenario families, each with:
  - experiment-level dropout variant, and
  - group-level dropout variant (distinct per-group dropout parameters)
- Scaling/timing uses only `groups_balanced_moderate_drop`.

### Data stored per `sce.rds`

- Assays:
  - `counts` (observed simulated counts)
  - `TrueCounts` (pre-dropout ground truth)
  - `logcounts` (CP10k + `log2(1 + x)` on `counts`)
  - `logTrueCounts` (CP10k + `log2(1 + x)` on `TrueCounts`)
- Gene filtering: retain genes expressed in at least 3 cells (from observed `counts`)
- Metadata includes observed and true library sizes used for CP10k normalization.

### Extra assays available in `sce.rds`

In addition to the four benchmark-evaluation assays above, the generated SCE files can include these extra Splatter assays:

- `BaseCellMeans`
- `BatchCellMeans`
- `BCV`
- `CellMeans`
- `DropProb` (present for dropout-enabled scenarios)
- `Dropout` (present for dropout-enabled scenarios; stored as integer 0/1 mask)

## 2) Metrics used to compare methods

Let:

- \(X^\star_{\log}\): ground-truth log-normalized matrix (`logTrueCounts`)
- \(X'_{\log}\): reconstructed/imputed log-normalized matrix
- `DEFacGroup*`: rowData columns with per-group differential-expression factors (\(1\) = not DE)
- \(N\): number of cells, \(G\): number of genes
- \(D = \{(i,j): x_{ij}=0 \land x^\star_{ij}>0\}\) (dropout-like entries)
- \(B = \{(i,j): x^\star_{ij}=0\}\) (biozero entries)
- \(\mathcal{J}_{M}=\{j:\exists g,\;|DEFacGroup_g(j)-1|>0\}\) (marker genes)
- \(C=\{(i,j): j\in\mathcal{J}_{M}\}\) (all entries from marker genes)
- \(|D|\), \(|B|\), \(|C|\): cardinalities of \(D\), \(B\), \(C\)
- \(\varepsilon = 10^{-8}\): numerical-stability constant

### (i) Mean Squared Error (MSE)

\[
\mathrm{MSE}=\frac{1}{NG}\sum_{i,j}\left(x^\star_{ij}-x'_{ij}\right)^2
\]

Subset variants:

\[
\mathrm{dropout\mbox{-}MSE}=\frac{1}{|D|}\sum_{(i,j)\in D}\left(x^\star_{ij}-x'_{ij}\right)^2
\]
\[
\mathrm{biozero\mbox{-}MSE}=\frac{1}{|B|}\sum_{(i,j)\in B}\left(x^\star_{ij}-x'_{ij}\right)^2
\]
\[
\mathrm{marker\mbox{-}MSE}=\frac{1}{|C|}\sum_{(i,j)\in C}\left(x^\star_{ij}-x'_{ij}\right)^2
\]

Edge case rule: if \(|D|=0\), \(|B|=0\), or \(|C|=0\), report the corresponding subset metric as `NA` for that dataset.

### (ii) Mean Absolute Error (MAE)

\[
\mathrm{MAE}=\frac{1}{NG}\sum_{i,j}\left|x^\star_{ij}-x'_{ij}\right|
\]

Subset variants:

\[
\mathrm{dropout\mbox{-}MAE}=\frac{1}{|D|}\sum_{(i,j)\in D}\left|x^\star_{ij}-x'_{ij}\right|
\]
\[
\mathrm{biozero\mbox{-}MAE}=\frac{1}{|B|}\sum_{(i,j)\in B}\left|x^\star_{ij}-x'_{ij}\right|
\]
\[
\mathrm{marker\mbox{-}MAE}=\frac{1}{|C|}\sum_{(i,j)\in C}\left|x^\star_{ij}-x'_{ij}\right|
\]

Edge case rule: if \(|D|=0\), \(|B|=0\), or \(|C|=0\), report the corresponding subset metric as `NA` for that dataset.

### (iii) Per-gene normalized RMSE (gNRMSE)

For each gene \(j\):

\[
\mathrm{RMSE}_j=\sqrt{\frac{1}{N}\sum_i\left(x^\star_{ij}-x'_{ij}\right)^2}, \quad
s_j=\mathrm{sd}\left(\{x^\star_{ij}\}_{i=1}^{N}\right)
\]

Aggregate:

\[
\mathrm{gNRMSE}=\frac{1}{G}\sum_{j=1}^{G}\frac{\mathrm{RMSE}_j}{\max(s_j,\varepsilon)}
\]
\[
\mathrm{marker\mbox{-}gNRMSE}=\frac{1}{|\mathcal{J}_{M}|}\sum_{j\in \mathcal{J}_{M}}\frac{\mathrm{RMSE}_j}{\max(s_j,\varepsilon)}
\]

Edge case rule: genes with zero (or near-zero) ground-truth variance use \(\max(s_j,\varepsilon)\) in the denominator to avoid division-by-zero blow-up.
If \(|\mathcal{J}_{M}|=0\), report `marker-gNRMSE = NA` for that dataset.

Output columns use `*_marker` for marker-gene subset metrics.

### (iv) Gene-gene correlation distortion (CorrErr)

Compute Pearson gene-gene correlations across cells from \(X^\star_{\log}\) and \(X'_{\log}\). Correlations are computed only on genes with non-zero variance in both matrices:

- \(\mathcal{J}_{\mathrm{corr}}=\{j:\mathrm{sd}(X^\star_{\log,\cdot j})>0 \land \mathrm{sd}(X'_{\log,\cdot j})>0\}\)
- \(G_{\mathrm{corr}}=|\mathcal{J}_{\mathrm{corr}}|\)

- \(R^\star\): true correlation matrix
- \(R'\): reconstructed correlation matrix

\[
\mathrm{CorrErr}=\frac{2}{G_{\mathrm{corr}}(G_{\mathrm{corr}}-1)}\sum_{j<k,\;j,k\in\mathcal{J}_{\mathrm{corr}}}\left|R^\star_{jk}-R'_{jk}\right|
\]

Edge case rule: if \(G_{\mathrm{corr}}<2\), report `CorrErr = NA` for that dataset.

Lower CorrErr indicates less distortion of gene-gene dependency structure.

### (v) Runtime scaling

- Evaluate wall-clock runtime on the scaling split:
  - \(N \in \{10000, 20000, \ldots, 100000\}\)
  - fixed representative scenario: `groups_balanced_moderate_drop`
- Reports scalability independently from changing biological scenario structure.

## 3) Fair comparison protocol

- Standardize preprocessing and evaluation across methods.
- Use tuning split (`N=1000`) only for model selection/hyperparameters.
- Report final reconstruction metrics on held-out test split (`N=5000`).
