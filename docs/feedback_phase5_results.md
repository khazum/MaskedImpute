# Phase 5 Results Update

## Scope

Phase 5 now uses a fairer real-data label-agreement evaluation. These experiments are downstream structure-preservation checks, not direct imputation-accuracy measurements, because real datasets do not provide pre-dropout ground truth.

## Fairness Improvements Implemented

1. Replaced supervised `*_top1000markers.rds` inputs with original/full matrices for Baron and Zeisel.
2. Added PBMC68k using the 10x RDS artifact and barcode-level annotations from `temp/single-cell-3prime-paper/pbmc68k_analysis/68k_pbmc_barcodes_annotation.tsv`.
3. Selected the top 1,000 HVGs label-free from observed logcounts using mean-binned dispersion.
4. Replaced the single fixed clustering setting with a PCA/SNN/Leiden grid: `n_pcs={20,50,100}`, `k={10,15,30}`, `resolution={0.2,0.5,1.0,1.5,2.0}`.
5. Repeated evaluation over five seeds and report grid mean/SD plus a cluster-count-matched summary.
6. Added batch-association diagnostics where available and DCA wall-clock budgeting.

## Datasets

- Baron human pancreas: `datasets_original/baron-human.rds`, 8,569 cells, 20,125 genes before HVG selection, 14 labels, donor column `human` used as batch diagnostic.
- Zeisel mouse cortex: `datasets_original/zeisel.rds`, 3,005 cells, 19,972 genes before HVG selection, 9 labels, cell-id prefix used as a plate-like batch diagnostic.
- PBMC68k: `temp/pbmc68k_data.rds`, 68,579 cells before subsampling, 32,738 genes before HVG selection, 11 published barcode annotations. Evaluation uses a fixed random 12,000-cell subset for tractable repeated benchmarking.

## Methods

Computed methods:

- Observed raw logcounts
- DCA
- scVI
- ALRA
- MAGIC
- MaskImpute

DCA was run with a 300-second per-dataset budget. No DCA run timed out on the 1,000-HVG matrices; statuses are recorded in `results_real_data/real_data_method_runs.tsv` and future timeouts will be reported explicitly.

## Outputs

- Grid metrics: `results_real_data/real_data_clustering_grid.tsv`
- Method run status/runtime: `results_real_data/real_data_method_runs.tsv`
- Summary metrics: `results_real_data/real_data_clustering_summary.tsv`
- Paper table: `paper/generated/real_data_table.tex`
- Runner: `scripts/run_phase5_real_data.py`
- PBMC extractor: `scripts/extract_pbmc68k_hvg.R`
- Log: `logs_parallel_runs/phase5_real_data_grid.log`

Command used:

```bash
CUDA_VISIBLE_DEVICES=0 .venv_scvi/bin/python scripts/run_phase5_real_data.py \
  --methods baseline,dca,scvi,alra,magic,maskimpute \
  --seeds 42,43,44,45,46 \
  --n-hvg 1000 \
  --n-pcs-grid 20,50,100 \
  --n-neighbors-grid 10,15,30 \
  --resolution-grid 0.2,0.5,1.0,1.5,2.0 \
  --pbmc-max-cells 12000 \
  --cell-sample-strategy random \
  --dca-timeout-sec 300
```

## Summary Results

ARI/NMI grid values are mean over seeds and grid settings. Matched values choose the grid setting whose cluster count is closest to the number of annotated labels for each seed.

| Dataset | Method | ARI grid | NMI grid | ARI matched | NMI matched | Batch ARI |
|---|---|---:|---:|---:|---:|---:|
| Baron | Observed | 0.508 | 0.771 | 0.667 | 0.830 | 0.147 |
| Baron | DCA | 0.442 | 0.746 | 0.607 | 0.818 | 0.160 |
| Baron | scVI | 0.424 | 0.739 | 0.607 | 0.816 | 0.162 |
| Baron | ALRA | 0.390 | 0.721 | 0.619 | 0.811 | 0.116 |
| Baron | MAGIC | 0.315 | 0.688 | 0.523 | 0.779 | 0.156 |
| Baron | MaskImpute | 0.456 | 0.754 | 0.684 | 0.845 | 0.163 |
| Zeisel | Observed | 0.409 | 0.629 | 0.757 | 0.738 | 0.038 |
| Zeisel | DCA | 0.381 | 0.611 | 0.655 | 0.670 | 0.042 |
| Zeisel | scVI | 0.374 | 0.604 | 0.650 | 0.675 | 0.042 |
| Zeisel | ALRA | 0.327 | 0.610 | 0.601 | 0.676 | 0.040 |
| Zeisel | MAGIC | 0.248 | 0.587 | 0.623 | 0.692 | 0.055 |
| Zeisel | MaskImpute | 0.366 | 0.621 | 0.688 | 0.700 | 0.042 |
| PBMC68k | Observed | 0.214 | 0.424 | 0.216 | 0.452 | 0.000 |
| PBMC68k | DCA | 0.212 | 0.430 | 0.277 | 0.486 | 0.001 |
| PBMC68k | scVI | 0.216 | 0.424 | 0.254 | 0.460 | 0.000 |
| PBMC68k | ALRA | 0.169 | 0.412 | 0.281 | 0.462 | 0.000 |
| PBMC68k | MAGIC | 0.145 | 0.374 | 0.206 | 0.449 | 0.000 |
| PBMC68k | MaskImpute | 0.187 | 0.402 | 0.220 | 0.419 | 0.000 |

## Interpretation

- The observed matrix remains very competitive, which is expected for label-agreement metrics because clustering can reward preservation of raw marker-like structure.
- MaskImpute has the strongest cluster-count-matched Baron result and the strongest matched Zeisel NMI among imputation methods.
- PBMC68k does not show a MaskImpute clustering advantage; this should be presented as a limitation of the downstream label-agreement evaluation, not hidden.
- The paper should avoid claiming real-data clustering superiority. The synthetic ground-truth benchmark remains the main evidence for imputation accuracy.
