# Phase 7: Final Integration and Verification

## Scope

Phase 7 regenerated paper artifacts, rebuilt the manuscript PDF, and verified that the active paper is internally consistent with generated results.

## Commands Run

```bash
.venv_scvi/bin/python scripts/generate_phase3_benchmark_assets.py \
  > logs_parallel_runs/phase7_generate_phase3.log 2>&1

.venv_scvi/bin/python scripts/generate_phase6_assets.py \
  > logs_parallel_runs/phase7_generate_phase6.log 2>&1

.venv_scvi/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from scripts.run_phase5_real_data import write_tex_summary
summary = pd.read_csv('results_real_data/real_data_clustering_summary.tsv', sep='\t')
write_tex_summary(summary, Path('paper/generated/real_data_table.tex'))
PY

conda install -n magic311 -c conda-forge perl tectonic pypdf -y
conda install -n magic311 -c defaults texlive-core=20240312 -y

conda run -n magic311 bash -lc 'cd paper && tectonic --keep-logs main.tex' \
  > logs_parallel_runs/phase7_compile_tectonic.log 2>&1

.venv_scvi/bin/python scripts/verify_phase7_paper_claims.py
conda run -n magic311 bash -lc \
  "python -c 'from pypdf import PdfReader; print(len(PdfReader(\"paper/main.pdf\").pages))'"
```

## Generated/Verified Artifacts

- Main PDF: `paper/main.pdf`
- Synthetic benchmark tables and figure: `paper/generated/benchmark_*.tex`, `paper/generated/benchmark_summary.json`, `paper/figures/mse_5000_bars.pdf`, `paper/figures/mse_5000_bars.png`
- Real-data table: `paper/generated/real_data_table.tex`
- Phase-6 tables: `paper/generated/phase6_ablation_table.tex`, `paper/generated/phase6_calibration_table.tex`, `paper/generated/phase6_downstream_table.tex`, `paper/generated/phase6_sensitivity_table.tex`
- Compatibility aliases: `paper/generated/ablation_table.tex`, `paper/generated/downstream_table.tex`

## Verification Results

- PDF compilation succeeded with Tectonic in `magic311`; output PDF is `paper/main.pdf`.
- PDF page count: `23` pages including references, appendix, and checklist.
- Claim verification passed: `scripts/verify_phase7_paper_claims.py` checks headline synthetic reductions, Biozero-MSE comparisons, real-data claims, phase-6 calibration claims, source references, citation keys, and obsolete terminology.
- Active paper sources and generated tables contain no remaining `MaskClass`, `MaskedImpute`, `Balanced-MSE`, `balanced_mse`, `UAI`, or `uai` terms.
- All `\input{...}` and `\includegraphics{...}` targets referenced from `paper/main.tex` exist.
- All citation keys used by the active paper are present in `paper/references.bib`.
- Compile log contains no undefined-reference or missing-citation warnings.

## Notes

- `latexmk`/`pdflatex` in the conda TeX Live package still cannot generate `pdflatex.fmt` because the package lacks the expected `mktexlsr.pl`/full TeX Live script tree. Tectonic was installed via conda and used for the successful PDF build.
- Tectonic reports layout warnings only: underfull/overfull boxes in the appendix/checklist and UTF-8 replacement warnings from downloaded TeX package files. No fatal errors or missing references were reported.
- Runtime scaling was not recomputed, consistent with the earlier instruction not to update scaling results during this pass.
