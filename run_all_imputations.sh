#!/usr/bin/env bash
set -euo pipefail

# Ensure the magic311 conda environment is active.
if [[ "${CONDA_DEFAULT_ENV:-}" != "magic311" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate magic311
  else
    echo "conda not found; activate the magic311 env before running." >&2
    exit 1
  fi
fi

BASE_DIR="synthetic_datasets/rds_splat_output"
OUT_BASE_R="results_imputation_r"
OUT_BASE_PY="results_imputation_py"

SIZES=(1000) # 5000 10000 15000 20000 25000)

for size in "${SIZES[@]}"; do
  in_dir="${BASE_DIR}/cells_${size}"
  if [[ ! -d "${in_dir}" ]]; then
    echo "Skipping missing dataset folder: ${in_dir}" >&2
    continue
  fi

  out_r="${OUT_BASE_R}/cells_${size}"
  out_py="${OUT_BASE_PY}/cells_${size}"
  mkdir -p "${out_r}" "${out_py}"

  echo "== cells_${size} =="
  #Rscript run_imputation.R "${in_dir}" "${out_r}"
  python run_imputation.py "${in_dir}" "${out_py}" --methods low_mse
done
