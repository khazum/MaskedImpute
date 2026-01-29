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
OUT_R="${OUT_R:-results_clustering_r}"
OUT_PY="${OUT_PY:-results_clustering_py}"
SIZES=(${SIZES:-1000 5000 10000 15000 20000 25000 50000 75000 100000})
LOG_DIR="${LOG_DIR:-logs_parallel_clustering}"

mkdir -p "$LOG_DIR"

CPU_THREADS="${CPU_THREADS:-8}"
GPU_THREADS="${GPU_THREADS:-8}"
NREPEATS="${NREPEATS:-5}"
MAGIC_JOBS="${MAGIC_JOBS:-$CPU_THREADS}"
CCIMPUTE_CORES="${CCIMPUTE_CORES:-8}"

export MASKEDIMPUTE_PYTHON="$(command -v python)"

echo "CPU threads per job: $CPU_THREADS"
echo "GPU threads per job: $GPU_THREADS"
echo "MAGIC jobs: $MAGIC_JOBS"
echo "ccImpute cores (R): $CCIMPUTE_CORES"
echo "Repeats (default): $NREPEATS"

repeats_for_size() {
  if [[ "$1" == "5000" ]]; then
    echo 10
  else
    echo "${NREPEATS}"
  fi
}

numa_node_for_method() {
  case "$1" in
    baseline|magic|dca|low_mse|experiment)
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
    dca)
      echo 0
      ;;
    autoclass)
      echo 1
      ;;
    low_mse)
      echo 2
      ;;
    balanced_mse)
      echo 3
      ;;
    experiment)
      echo 4
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
    for size in "${SIZES[@]}"; do
      local in_dir="${BASE_DIR}/cells_${size}"
      if [[ ! -d "${in_dir}" ]]; then
        echo "Skipping missing dataset folder: ${in_dir}"
        continue
      fi
      local out_dir="${OUT_R}/${method}/cells_${size}"
      mkdir -p "${out_dir}"
      echo "Processing ${method} cells_${size}..."
      local repeats
      repeats="$(repeats_for_size "${size}")"

      if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${ncores}" MKL_NUM_THREADS="${ncores}" \
          OPENBLAS_NUM_THREADS="${ncores}" NUMEXPR_NUM_THREADS="${ncores}" \
          numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
          Rscript run_clustering.R "${in_dir}" "${out_dir}" "${ncores}" "${repeats}" "${method}"
      else
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${ncores}" MKL_NUM_THREADS="${ncores}" \
          OPENBLAS_NUM_THREADS="${ncores}" NUMEXPR_NUM_THREADS="${ncores}" \
          Rscript run_clustering.R "${in_dir}" "${out_dir}" "${ncores}" "${repeats}" "${method}"
      fi
    done
    echo "Finished [R/${method}]"
  ) > "$log_file" 2>&1 &
}

run_py_cpu_method() {
  local method="$1"
  local numa_node="$2"
  local log_file="${LOG_DIR}/py_cpu_${method}.log"

  echo "Started [PY/${method}] (CPU-only, NUMA ${numa_node:-none}) - logging to $log_file"

  (
    for size in "${SIZES[@]}"; do
      local in_dir="${BASE_DIR}/cells_${size}"
      if [[ ! -d "${in_dir}" ]]; then
        echo "Skipping missing dataset folder: ${in_dir}"
        continue
      fi
      local out_dir="${OUT_PY}/${method}/cells_${size}"
      mkdir -p "${out_dir}"
      echo "Processing ${method} cells_${size}..."
      local repeats
      repeats="$(repeats_for_size "${size}")"

      if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
          numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
          python run_clustering.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}" --n-repeat "${repeats}"
      else
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
          python run_clustering.py "${in_dir}" "${out_dir}" "${method}" --n-jobs "${MAGIC_JOBS}" --n-repeat "${repeats}"
      fi
    done
    echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1 &
}

run_py_gpu_method() {
  local method="$1"
  local gpu="$2"
  local numa_node="$3"
  local log_file="${LOG_DIR}/py_gpu_${method}.log"

  echo "Started [PY/${method}] (GPU ${gpu}, NUMA ${numa_node:-none}) - logging to $log_file"

  (
    for size in "${SIZES[@]}"; do
      local in_dir="${BASE_DIR}/cells_${size}"
      if [[ ! -d "${in_dir}" ]]; then
        echo "Skipping missing dataset folder: ${in_dir}"
        continue
      fi
      local out_dir="${OUT_PY}/${method}/cells_${size}"
      mkdir -p "${out_dir}"
      echo "Processing ${method} cells_${size}..."
      local repeats
      repeats="$(repeats_for_size "${size}")"

      local -a extra_args=()
      if [[ "${method}" == "dca" ]]; then
        extra_args+=(--dca-threads "${GPU_THREADS}")
      fi

      if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_THREADS}" MKL_NUM_THREADS="${GPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${GPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_THREADS}" TORCH_NUM_THREADS="${GPU_THREADS}" \
          numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
          python run_clustering.py "${in_dir}" "${out_dir}" "${method}" "${extra_args[@]}" --n-repeat "${repeats}"
      else
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_THREADS}" MKL_NUM_THREADS="${GPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${GPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_THREADS}" TORCH_NUM_THREADS="${GPU_THREADS}" \
          python run_clustering.py "${in_dir}" "${out_dir}" "${method}" "${extra_args[@]}" --n-repeat "${repeats}"
      fi
    done
    echo "Finished [PY/${method}]"
  ) > "$log_file" 2>&1 &
}

echo "Starting parallel clustering runs on $(hostname)..."
echo "Monitor progress with: tail -f ${LOG_DIR}/*.log"

CPU_R_METHODS=(baseline saver ccimpute)
CPU_PY_METHODS=(magic)
GPU_METHODS=(dca autoclass low_mse balanced_mse experiment)

for method in "${CPU_R_METHODS[@]}"; do
  ncores="${CPU_THREADS}"
  if [[ "${method}" == "ccimpute" ]]; then
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

wait
echo "All clustering runs complete."
