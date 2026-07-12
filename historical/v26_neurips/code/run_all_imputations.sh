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

BASE_DIR="${BASE_DIR:-simulated_data/test}"
OUT_BASE_R="${OUT_BASE_R:-results_imputation_r}"
OUT_BASE_PY="${OUT_BASE_PY:-results_imputation_py}"
NREPEATS="${NREPEATS:-5}"

if [[ ! -d "${BASE_DIR}" ]]; then
  echo "Missing dataset root: ${BASE_DIR}" >&2
  exit 1
fi

for scenario_dir in "${BASE_DIR}"/*; do
  [[ -d "${scenario_dir}" ]] || continue
  scenario="$(basename "${scenario_dir}")"
  in_file="${scenario_dir}/sce.rds"
  if [[ ! -f "${in_file}" ]]; then
    echo "Skipping missing dataset file: ${in_file}" >&2
    continue
  fi

  out_r="${OUT_BASE_R}/all_methods/test/${scenario}"
  out_py="${OUT_BASE_PY}/all_methods/test/${scenario}"
  mkdir -p "${out_r}" "${out_py}"

  echo "== ${scenario} =="
  conda run -n r45_bio Rscript run_imputation.R "${in_file}" "${out_r}" 8 "${NREPEATS}" all
  python run_imputation.py "${in_file}" "${out_py}" --methods all --n-repeat "${NREPEATS}"
done
