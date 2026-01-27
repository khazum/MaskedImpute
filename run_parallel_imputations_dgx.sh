#!/usr/bin/env bash
set -euo pipefail

# Activate magic311 conda env if needed.
if [[ "${CONDA_DEFAULT_ENV:-}" != "magic311" ]]; then
  if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate magic311
  else
    echo "conda not found; activate the magic311 env before running." >&2
    exit 1
  fi
fi

BASE_DIR="${BASE_DIR:-synthetic_datasets/rds_splat_output}"
OUT_R="${OUT_R:-results_imputation_r}"
OUT_PY="${OUT_PY:-results_imputation_py}"
SIZES=(${SIZES:-1000 5000 10000 15000 20000 25000})

NCORES="${NCORES:-$(nproc)}"
NREPEATS="${NREPEATS:-10}"
MAGIC_JOBS="${MAGIC_JOBS:-${NCORES}}"

run_r_method() {
  local method="$1"
  local numa_node="${2:-}"
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}" >&2
      continue
    fi
    local out_dir="${OUT_R}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "[R/${method}] cells_${size}"
    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        Rscript run_imputation.R "${in_dir}" "${out_dir}" "${NCORES}" "${NREPEATS}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="" Rscript run_imputation.R "${in_dir}" "${out_dir}" "${NCORES}" "${NREPEATS}" "${method}"
    fi
  done
}

run_py_method() {
  local method="$1"
  local gpu="$2"
  local numa_node="${3:-}"
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}" >&2
      continue
    fi
    local out_dir="${OUT_PY}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "[PY/${method}] cells_${size} (GPU ${gpu})"
    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="${gpu}" numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" python run_imputation.py "${in_dir}" "${out_dir}" "${method}"
    fi
  done
}

run_py_cpu_method() {
  local method="$1"
  local numa_node="${2:-}"
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}" >&2
      continue
    fi
    local out_dir="${OUT_PY}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "[PY/${method}] cells_${size} (CPU)"
    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}"
    else
      CUDA_VISIBLE_DEVICES="" python run_imputation.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}"
    fi
  done
}

echo "Starting parallel runs on $(hostname)"

# CPU-only methods
run_r_method baseline 0 &
run_r_method saver 1 &
run_r_method ccimpute 1 &
run_py_cpu_method magic 0 &

# GPU methods (one per GPU)
run_py_method dca 0 0 &
run_py_method autoclass 1 1 &
run_py_method low_mse 2 0 &
run_py_method balanced_mse 3 1 &

wait
echo "All runs complete."
