#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

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
LOG_DIR="logs_parallel_runs"

mkdir -p "$LOG_DIR"

# ------------------------------------------------------------------
# RESOURCE CALCULATION
# ------------------------------------------------------------------
TOTAL_CORES="${NCORES:-$(nproc)}"
CORES_PER_SOCKET=$((TOTAL_CORES / 2))
CPU_JOBS_PER_NODE="${CPU_JOBS_PER_NODE:-4}"
THREADS_PER_JOB="${THREADS_PER_JOB:-$((CORES_PER_SOCKET / CPU_JOBS_PER_NODE))}"
GPU_CPU_THREADS="${GPU_CPU_THREADS:-4}"

if [[ "$THREADS_PER_JOB" -lt 1 ]]; then THREADS_PER_JOB=1; fi
if [[ "$GPU_CPU_THREADS" -lt 1 ]]; then GPU_CPU_THREADS=1; fi

echo "Total Cores: $TOTAL_CORES"
echo "Cores per socket: $CORES_PER_SOCKET"
echo "CPU jobs per socket: $CPU_JOBS_PER_NODE"
echo "Threads per CPU job: $THREADS_PER_JOB"
echo "Threads per GPU job: $GPU_CPU_THREADS"

NREPEATS="${NREPEATS:-5}"
MAGIC_JOBS="${THREADS_PER_JOB}"

run_r_job() {
  local method="$1"
  local size="$2"
  local numa_node="$3"
  local log_file="${LOG_DIR}/r_${method}_cells_${size}.log"

  (
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}"
      exit 0
    fi
    local out_dir="${OUT_R}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "Processing ${method} cells_${size}..."

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
        OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}" \
        numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        Rscript run_imputation.R "${in_dir}" "${out_dir}" "${THREADS_PER_JOB}" "${NREPEATS}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
        OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}" \
        Rscript run_imputation.R "${in_dir}" "${out_dir}" "${THREADS_PER_JOB}" "${NREPEATS}" "${method}"
    fi
  ) > "$log_file" 2>&1 &
}

run_py_cpu_job() {
  local method="$1"
  local size="$2"
  local numa_node="$3"
  local log_file="${LOG_DIR}/py_cpu_${method}_cells_${size}.log"

  (
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "Skipping missing dataset folder: ${in_dir}"
      exit 0
    fi
    local out_dir="${OUT_PY}/${method}/cells_${size}"
    mkdir -p "${out_dir}"
    echo "Processing ${method} cells_${size}..."

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
        OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}" \
        numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}"
    else
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${THREADS_PER_JOB}" MKL_NUM_THREADS="${THREADS_PER_JOB}" \
        OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}" NUMEXPR_NUM_THREADS="${THREADS_PER_JOB}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}"
    fi
  ) > "$log_file" 2>&1 &
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
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_CPU_THREADS}" MKL_NUM_THREADS="${GPU_CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${GPU_CPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_CPU_THREADS}" TORCH_NUM_THREADS="${GPU_CPU_THREADS}" \
        numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_CPU_THREADS}" MKL_NUM_THREADS="${GPU_CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${GPU_CPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_CPU_THREADS}" TORCH_NUM_THREADS="${GPU_CPU_THREADS}" \
        python run_imputation.py "${in_dir}" "${out_dir}" "${method}"
    fi
  done
  echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1
}

echo "Starting parallel runs on $(hostname)..."
echo "Monitor progress with: tail -f ${LOG_DIR}/*.log"

# CPU-only methods: schedule per-size jobs to maximize parallelism.
CPU_TASKS_NODE0=()
CPU_TASKS_NODE1=()
for size in "${SIZES[@]}"; do
  CPU_TASKS_NODE0+=("baseline:${size}")
  CPU_TASKS_NODE0+=("magic:${size}")
  CPU_TASKS_NODE1+=("saver:${size}")
  CPU_TASKS_NODE1+=("ccimpute:${size}")
done

launch_cpu_tasks() {
  local numa_node="$1"
  shift
  local -a tasks=("$@")
  local running=0
  for task in "${tasks[@]}"; do
    local method="${task%%:*}"
    local size="${task##*:}"
    if [[ "${method}" == "magic" ]]; then
      run_py_cpu_job "${method}" "${size}" "${numa_node}"
    else
      run_r_job "${method}" "${size}" "${numa_node}"
    fi
    running=$((running + 1))
    if [[ "${running}" -ge "${CPU_JOBS_PER_NODE}" ]]; then
      wait -n
      running=$((running - 1))
    fi
  done
  wait
}

launch_cpu_tasks 0 "${CPU_TASKS_NODE0[@]}" &
launch_cpu_tasks 1 "${CPU_TASKS_NODE1[@]}" &

# GPU methods (one per GPU)
run_py_method dca 0 0 &
run_py_method autoclass 1 1 &
run_py_method low_mse 2 0 &
run_py_method balanced_mse 3 1 &

wait
echo "All runs complete."
