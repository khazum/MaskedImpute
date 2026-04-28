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
  > logs_parallel_runs/phase7_neurips_compile.log 2>&1

.venv_scvi/bin/python scripts/verify_phase7_paper_claims.py
conda run -n magic311 python scripts/verify_neurips_submission.py
```

## Generated/Verified Artifacts

- Main PDF: `paper/main.pdf`
- Anonymous supplementary code/results archive: `supplementary_maskimpute_anonymous.zip`
- Synthetic benchmark tables and figure: `paper/generated/benchmark_*.tex`, `paper/generated/benchmark_summary.json`, `paper/figures/mse_5000_bars.pdf`, `paper/figures/mse_5000_bars.png`
- Real-data table: `paper/generated/real_data_table.tex`
- Phase-6 tables: `paper/generated/phase6_ablation_table.tex`, `paper/generated/phase6_calibration_table.tex`, `paper/generated/phase6_downstream_table.tex`, `paper/generated/phase6_sensitivity_table.tex`
- Compatibility aliases: `paper/generated/ablation_table.tex`, `paper/generated/downstream_table.tex`

## Verification Results

- PDF compilation succeeded with Tectonic in `magic311`; output PDF is `paper/main.pdf`.
- PDF page count: `25` pages including references, appendix, and checklist.
- Main content pages: `9`; references start on page `10`, appendices start on page `12`, and the NeurIPS checklist starts on page `19`.
- Appendix floats are placed in section order with `[H]` floats and `\FloatBarrier`/`\clearpage` boundaries, avoiding empty appendix headers.
- PDF size: approximately `0.62` MB, below the 50 MB submission limit.
- Supplementary archive size: approximately `0.254` MB, below the 100 MB supplementary ZIP limit.
- Claim verification passed: `scripts/verify_phase7_paper_claims.py` checks headline synthetic reductions, Biozero-MSE comparisons, real-data claims, phase-6 calibration claims, source references, citation keys, and obsolete terminology.
- NeurIPS submission verification passed: `scripts/verify_neurips_submission.py` checks PDF size, main content page count, required ordering, anonymous submission style options, checklist presence, and obvious deanonymization/acknowledgment text.
- Active paper sources and generated tables contain no remaining `MaskClass`, `MaskedImpute`, `Balanced-MSE`, `balanced_mse`, `UAI`, or `uai` terms.
- All `\input{...}` and `\includegraphics{...}` targets referenced from `paper/main.tex` exist.
- All citation keys used by the active paper are present in `paper/references.bib`.
- Compile log contains no undefined-reference or missing-citation warnings.

## Notes

- `latexmk`/`pdflatex` in the conda TeX Live package still cannot generate `pdflatex.fmt` because the package lacks the expected `mktexlsr.pl`/full TeX Live script tree. Tectonic was installed via conda and used for the successful PDF build.
- Tectonic reports layout warnings only: underfull/overfull boxes in the appendix/checklist and UTF-8 replacement warnings from downloaded TeX package files. No fatal errors or missing references were reported.
- Runtime scaling was not recomputed, consistent with the earlier instruction not to update scaling results during this pass.
- Recommended contribution type for the submission form: `Use-Inspired`, because the contribution is a method and benchmark tied to a real-world scRNA-seq use case.
