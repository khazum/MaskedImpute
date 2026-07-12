# Anonymous Supplementary Material for MaskImpute

This archive contains anonymized code, configuration files, generated result tables, and scripts supporting the submitted MaskImpute paper.

## Contents

- `code/`: MaskImpute implementation and benchmark runner entry points.
- `scripts/`: table/figure generation, real-data evaluation, phase-6 diagnostics, and verification scripts.
- `synthetic_datasets/`: synthetic benchmark generation script.
- `configs/`: phase-6 ablation and sensitivity configurations.
- `paper_generated/`: generated LaTeX tables and benchmark summary JSON used by the paper.
- `results/`: compact generated result summaries used by the paper.
- `docs/`: reproduction and verification notes.

## Minimal Reproduction Path

1. Create the Python/R environments described in the paper and code comments.
2. Generate or obtain the benchmark RDS files according to `synthetic_datasets/generate_simulated_benchmark.R` and `benchmark_datasets_and_metrics.md`.
3. Run MaskImpute through `code/run_imputation.py` or the parallel runner.
4. Regenerate paper assets with:

```bash
python scripts/generate_phase3_benchmark_assets.py
python scripts/generate_phase6_assets.py
python scripts/verify_phase7_paper_claims.py
```

The full benchmark requires GPU resources for neural baselines and was run on a DGX H100-class system as described in the paper. Public real datasets are referenced in the manuscript; they are not redistributed in this archive.
