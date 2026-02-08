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

DATA_DIR="${DATA_DIR:-datasets}"
OUT_R="${OUT_R:-results_clustering_r}"
OUT_PY="${OUT_PY:-results_clustering_py}"
LOG_DIR="${LOG_DIR:-logs_parallel_clustering}"

mkdir -p "$LOG_DIR"

CPU_THREADS="${CPU_THREADS:-8}"
GPU_THREADS="${GPU_THREADS:-8}"
NREPEATS="${NREPEATS:-5}"
MAGIC_JOBS="${MAGIC_JOBS:-$CPU_THREADS}"
CCIMPUTE_CORES="${CCIMPUTE_CORES:-8}"
SAVER_CORES="${SAVER_CORES:-8}"

export MASKEDIMPUTE_PYTHON="$(command -v python)"

echo "CPU threads per job: $CPU_THREADS"
echo "GPU threads per job: $GPU_THREADS"
echo "MAGIC jobs: $MAGIC_JOBS"
echo "ccImpute cores (R): $CCIMPUTE_CORES"
echo "SAVER cores (R): $SAVER_CORES"
echo "Repeats (default): $NREPEATS"

declare -a JOB_PIDS=()
declare -a JOB_NAMES=()
declare -a JOB_LOGS=()

numa_node_for_method() {
  case "$1" in
    baseline|magic|dca)
      echo 0
      ;;
    saver|ccimpute|autoclass|balanced_mse)
      echo 1
      ;;
    *)
      echo ""
      ;;
  esac
}

gpu_for_method() {
  case "$1" in
    autoclass)
      echo 1
      ;;
    balanced_mse)
      echo 3
      ;;
    *)
      echo ""
      ;;
  esac
}

run_r_method() {
  local method="$1"
  local numa_node="$2"
  local ncores="$3"
  local log_file="${LOG_DIR}/r_${method}.log"

  echo "Started [R/${method}] (NUMA ${numa_node:-none}, cores ${ncores}) - logging to $log_file"

  (
    local out_dir="${OUT_R}/${method}"
    mkdir -p "${out_dir}"
    echo "Processing ${method} datasets from ${DATA_DIR}..."

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${ncores}" MKL_NUM_THREADS="${ncores}" \
        OPENBLAS_NUM_THREADS="${ncores}" NUMEXPR_NUM_THREADS="${ncores}" \
        numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        Rscript run_clustering.R "${DATA_DIR}" "${out_dir}" "${ncores}" "${NREPEATS}" "${method}"
    else
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${ncores}" MKL_NUM_THREADS="${ncores}" \
        OPENBLAS_NUM_THREADS="${ncores}" NUMEXPR_NUM_THREADS="${ncores}" \
        Rscript run_clustering.R "${DATA_DIR}" "${out_dir}" "${ncores}" "${NREPEATS}" "${method}"
    fi
    echo "Finished [R/${method}]"
  ) > "$log_file" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("R/${method}")
  JOB_LOGS+=("${log_file}")
}

run_py_cpu_method() {
  local method="$1"
  local numa_node="$2"
  local log_file="${LOG_DIR}/py_cpu_${method}.log"

  echo "Started [PY/${method}] (CPU-only, NUMA ${numa_node:-none}) - logging to $log_file"

  (
    local out_dir="${OUT_PY}/${method}"
    mkdir -p "${out_dir}"
    echo "Processing ${method} datasets from ${DATA_DIR}..."
    local -a extra_args=()
    if [[ "${method}" == "dca" ]]; then
      extra_args+=(--dca-threads "${CPU_THREADS}")
    fi

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
        numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_clustering.py "${DATA_DIR}" "${out_dir}" "${method}" "${extra_args[@]}" --n-jobs "${MAGIC_JOBS}" --n-repeat "${NREPEATS}"
    else
      CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
        python run_clustering.py "${DATA_DIR}" "${out_dir}" "${method}" "${extra_args[@]}" --n-jobs "${MAGIC_JOBS}" --n-repeat "${NREPEATS}"
    fi
    echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("PY/${method}")
  JOB_LOGS+=("${log_file}")
}

run_py_gpu_method() {
  local method="$1"
  local gpu="$2"
  local numa_node="$3"
  local log_file="${LOG_DIR}/py_gpu_${method}.log"

  echo "Started [PY/${method}] (GPU ${gpu}, NUMA ${numa_node:-none}) - logging to $log_file"

  (
    local out_dir="${OUT_PY}/${method}"
    mkdir -p "${out_dir}"
    echo "Processing ${method} datasets from ${DATA_DIR}..."

    if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_THREADS}" MKL_NUM_THREADS="${GPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${GPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_THREADS}" TORCH_NUM_THREADS="${GPU_THREADS}" \
        numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
        python run_clustering.py "${DATA_DIR}" "${out_dir}" "${method}" --n-repeat "${NREPEATS}"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_THREADS}" MKL_NUM_THREADS="${GPU_THREADS}" \
        OPENBLAS_NUM_THREADS="${GPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_THREADS}" TORCH_NUM_THREADS="${GPU_THREADS}" \
        python run_clustering.py "${DATA_DIR}" "${out_dir}" "${method}" --n-repeat "${NREPEATS}"
    fi
    echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("PY/${method}")
  JOB_LOGS+=("${log_file}")
}

echo "Starting parallel clustering runs on $(hostname)..."
echo "Monitor progress with: tail -f ${LOG_DIR}/*.log"

CPU_R_METHODS=(baseline saver ccimpute)
CPU_PY_METHODS=(magic dca)
GPU_METHODS=(autoclass balanced_mse)

for method in "${CPU_R_METHODS[@]}"; do
  ncores="${CPU_THREADS}"
  if [[ "${method}" == "saver" ]]; then
    ncores="${SAVER_CORES}"
  elif [[ "${method}" == "ccimpute" ]]; then
    ncores="${CCIMPUTE_CORES}"
  fi
  run_r_method "${method}" "$(numa_node_for_method "${method}")" "${ncores}"
done

for method in "${CPU_PY_METHODS[@]}"; do
  run_py_cpu_method "${method}" "$(numa_node_for_method "${method}")"
done

for method in "${GPU_METHODS[@]}"; do
  gpu_id="$(gpu_for_method "${method}")"
  numa_node="$(numa_node_for_method "${method}")"
  if [[ -z "${gpu_id}" ]]; then
    echo "Skipping ${method}: no GPU mapping configured."
    continue
  fi
  run_py_gpu_method "${method}" "${gpu_id}" "${numa_node}"
done

fail_count=0
declare -A PID_TO_INDEX=()
for i in "${!JOB_PIDS[@]}"; do
  PID_TO_INDEX["${JOB_PIDS[$i]}"]="$i"
done

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  echo "This script requires bash support for 'wait -n' and 'wait -p'." >&2
  exit 2
fi

remaining="${#JOB_PIDS[@]}"
while [[ "${remaining}" -gt 0 ]]; do
  done_pid=""
  if wait -n -p done_pid; then
    status=0
  else
    status=$?
  fi
  if [[ -z "${done_pid}" ]]; then
    echo "Internal error: wait -n returned empty PID." >&2
    exit 2
  fi
  idx="${PID_TO_INDEX[$done_pid]}"
  name="${JOB_NAMES[$idx]}"
  log_file="${JOB_LOGS[$idx]}"
  if [[ "${status}" -eq 0 ]]; then
    echo "[OK] ${name} completed."
  else
    fail_count=$((fail_count + 1))
    echo "[FAIL] ${name} exited with status ${status}. See ${log_file}"
  fi
  remaining=$((remaining - 1))
done

if [[ "${fail_count}" -gt 0 ]]; then
  echo "All clustering runs finished with ${fail_count} failed method(s)."
  exit 1
fi
echo "All clustering runs completed successfully."
