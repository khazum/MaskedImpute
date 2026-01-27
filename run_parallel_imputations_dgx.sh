#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4

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
SIZES=(${SIZES:-1000 5000 10000 15000 20000})
LOG_DIR="logs_parallel_runs"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------
# RESOURCE CALCULATION
# ------------------------------------------------------------------
TOTAL_CORES="${NCORES:-$(nproc)}"
# We are launching 4 CPU-heavy methods (Baseline, Saver, CCImpute, Magic)
# and 4 GPU methods (which need lighter CPU support).
# We divide cores by 5 to leave headroom for the GPU jobs and OS.
THREADS_PER_JOB=4

# Ensure at least 1 thread per job
if [[ "$THREADS_PER_JOB" -lt 1 ]]; then THREADS_PER_JOB=1; fi

echo "Total Cores: $TOTAL_CORES"
echo "Concurrent CPU Jobs: 4"
echo "Allocating $THREADS_PER_JOB cores per CPU job to avoid thrashing."

NREPEATS="${NREPEATS:-5}"
MAGIC_JOBS="${THREADS_PER_JOB}"

run_r_method() {
  local method="$1"
  local numa_node="${2:-}"
  local log_file="${LOG_DIR}/r_${method}.log"
  
  echo "Started [R/${method}] - logging to $log_file"
  
  (
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}"
      continue
    fi
    local out_dir="${OUT_R}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "Processing cells_${size}..."
    
    # Use calculated THREADS_PER_JOB instead of full NCORES
    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        Rscript run_imputation.R "${in_dir}" "${out_dir}" "${THREADS_PER_JOB}" "${NREPEATS}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="" Rscript run_imputation.R "${in_dir}" "${out_dir}" "${THREADS_PER_JOB}" "${NREPEATS}" "${method}"
    fi
  done
  echo "Finished [R/${method}]"
  ) > "$log_file" 2>&1
}

run_py_method() {
  local method="$1"
  local gpu="$2"
  local numa_node="${3:-}"
  local log_file="${LOG_DIR}/py_gpu_${method}.log"

  echo "Started [PY/${method}] (GPU ${gpu}) - logging to $log_file"

  (
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}"
      continue
    fi
    local out_dir="${OUT_PY}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "Processing cells_${size}..."

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="${gpu}" numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" python run_imputation.py "${in_dir}" "${out_dir}" "${method}"
    fi
  done
  echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1
}

run_py_cpu_method() {
  local method="$1"
  local numa_node="${2:-}"
  local log_file="${LOG_DIR}/py_cpu_${method}.log"
  
  echo "Started [PY/${method}] (CPU) - logging to $log_file"

  (
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}"
      continue
    fi
    local out_dir="${OUT_PY}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "Processing cells_${size}..."

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}"
    else
      CUDA_VISIBLE_DEVICES="" python run_imputation.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}"
    fi
  done
  echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1
}

echo "Starting parallel runs on $(hostname)..."
echo "Monitor progress with: tail -f ${LOG_DIR}/*.log"

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
