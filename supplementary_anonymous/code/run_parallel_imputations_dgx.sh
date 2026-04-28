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

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  echo "This script requires bash support for 'wait -n' and 'wait -p'." >&2
  exit 2
fi

RSCRIPT_CMD=(conda run -n "${R_CONDA_ENV:-r45_bio}" Rscript)
DEFAULT_DATA_ROOT=""
if [[ -d "simulated_data/test" ]]; then
  DEFAULT_DATA_ROOT="simulated_data/test"
elif [[ -d "synthetic_datasets/simulated_data/test" ]]; then
  DEFAULT_DATA_ROOT="synthetic_datasets/simulated_data/test"
else
  DEFAULT_DATA_ROOT="simulated_data/test"
fi
DATA_ROOT="${DATA_ROOT:-${DEFAULT_DATA_ROOT}}"
SPLIT_NAME="${SPLIT_NAME:-test}"
SCENARIOS_RAW="${SCENARIOS:-all}"
OUT_R="${OUT_R:-results_imputation_r}"
OUT_PY="${OUT_PY:-results_imputation_py}"
LOG_DIR="${LOG_DIR:-logs_parallel_runs}"

CPU_THREADS="${CPU_THREADS:-8}"
GPU_THREADS="${GPU_THREADS:-8}"
NREPEATS="${NREPEATS:-5}"
MAGIC_JOBS="${MAGIC_JOBS:-$CPU_THREADS}"
SAVER_CORES="${SAVER_CORES:-8}"
CCIMPUTE_CORES="${CCIMPUTE_CORES:-8}"
DCA_THREADS="${DCA_THREADS:-$CPU_THREADS}"
if [[ -x ".venv_scvi/bin/python" ]]; then
  SCVI_PYTHON="${SCVI_PYTHON:-.venv_scvi/bin/python}"
else
  SCVI_PYTHON="${SCVI_PYTHON:-python}"
fi
SCVI_MAX_EPOCHS="${SCVI_MAX_EPOCHS:-400}"
SCVI_BATCH_SIZE="${SCVI_BATCH_SIZE:-256}"
SCVI_LATENT="${SCVI_LATENT:-10}"
SCVI_HIDDEN="${SCVI_HIDDEN:-128}"
SCVI_LAYERS="${SCVI_LAYERS:-2}"
ALRA_MAX_RANK="${ALRA_MAX_RANK:-100}"

mkdir -p "$LOG_DIR"

discover_scenarios() {
  local data_root="$1"
  local raw="$2"
  local -n out_ref="$3"
  out_ref=()

  if [[ ! -d "${data_root}" ]]; then
    echo "Missing data root: ${data_root}" >&2
    return 1
  fi

  if [[ -z "${raw}" || "${raw}" == "all" ]]; then
    while IFS= read -r -d '' scenario_dir; do
      local scenario
      scenario="$(basename "${scenario_dir}")"
      if [[ -f "${scenario_dir}/sce.rds" ]]; then
        out_ref+=("${scenario}")
      fi
    done < <(find "${data_root}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
  else
    local -a requested=()
    IFS=', ' read -r -a requested <<< "${raw}"
    local scenario
    for scenario in "${requested[@]}"; do
      [[ -z "${scenario}" ]] && continue
      if [[ ! -f "${data_root}/${scenario}/sce.rds" ]]; then
        echo "Skipping unknown scenario '${scenario}' (missing ${data_root}/${scenario}/sce.rds)." >&2
        continue
      fi
      out_ref+=("${scenario}")
    done
  fi

  if [[ "${#out_ref[@]}" -eq 0 ]]; then
    echo "No scenarios selected under ${data_root}" >&2
    return 1
  fi

  return 0
}

declare -a SCENARIOS=()
if ! discover_scenarios "${DATA_ROOT}" "${SCENARIOS_RAW}" SCENARIOS; then
  exit 1
fi
if [[ "${#SCENARIOS[@]}" -eq 0 ]]; then
  echo "No scenarios available to run." >&2
  exit 1
fi

echo "Data root: ${DATA_ROOT}"
echo "Split name: ${SPLIT_NAME}"
echo "Scenarios (${#SCENARIOS[@]}): ${SCENARIOS[*]}"
echo "CPU threads per job: ${CPU_THREADS}"
echo "GPU threads per job: ${GPU_THREADS}"
echo "MAGIC jobs: ${MAGIC_JOBS}"
echo "DCA threads: ${DCA_THREADS}"
echo "ccImpute cores (R): ${CCIMPUTE_CORES}"
echo "SAVER cores (R): ${SAVER_CORES}"
echo "scVI python: ${SCVI_PYTHON}"
echo "scVI epochs/batch: ${SCVI_MAX_EPOCHS}/${SCVI_BATCH_SIZE}"
echo "Repeats per method: ${NREPEATS}"
echo "Rscript: ${RSCRIPT_CMD[*]}"

declare -a JOB_PIDS=()
declare -a JOB_NAMES=()
declare -a JOB_LOGS=()

method_display_name() {
  case "$1" in
    balanced_mse)
      echo "MaskImpute"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

numa_node_for_method() {
  case "$1" in
    baseline|magic|dca)
      echo 0
      ;;
    alra|saver|ccimpute|autoclass|scvi|balanced_mse)
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
    scvi)
      echo 2
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

  echo "Started [R/${method}] (NUMA ${numa_node:-none}, cores ${ncores}) - logging to ${log_file}"

  (
    local processed=0
    local scenario
    for scenario in "${SCENARIOS[@]}"; do
      local in_path="${DATA_ROOT}/${scenario}/sce.rds"
      if [[ ! -f "${in_path}" ]]; then
        echo "Skipping missing dataset file: ${in_path}"
        continue
      fi
      processed=$((processed + 1))
      local out_dir="${OUT_R}/${method}/${SPLIT_NAME}/${scenario}"
      mkdir -p "${out_dir}"
      echo "Processing ${method} ${scenario}..."

      if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${ncores}" MKL_NUM_THREADS="${ncores}" \
          OPENBLAS_NUM_THREADS="${ncores}" NUMEXPR_NUM_THREADS="${ncores}" \
          numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
          "${RSCRIPT_CMD[@]}" run_imputation.R "${in_path}" "${out_dir}" "${ncores}" "${NREPEATS}" "${method}"
      else
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${ncores}" MKL_NUM_THREADS="${ncores}" \
          OPENBLAS_NUM_THREADS="${ncores}" NUMEXPR_NUM_THREADS="${ncores}" \
          "${RSCRIPT_CMD[@]}" run_imputation.R "${in_path}" "${out_dir}" "${ncores}" "${NREPEATS}" "${method}"
      fi
    done
    if [[ "${processed}" -eq 0 ]]; then
      echo "No runnable datasets found for R/${method}. Check DATA_ROOT and SCENARIOS." >&2
      exit 1
    fi
    echo "Finished [R/${method}]"
  ) > "${log_file}" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("R/${method}")
  JOB_LOGS+=("${log_file}")
}

run_py_cpu_method() {
  local method="$1"
  local numa_node="$2"
  local log_file="${LOG_DIR}/py_cpu_${method}.log"
  local display_name
  display_name="$(method_display_name "${method}")"

  echo "Started [PY/${display_name}] (method key ${method}, CPU-only, NUMA ${numa_node:-none}) - logging to ${log_file}"

  (
    local processed=0
    local scenario
    for scenario in "${SCENARIOS[@]}"; do
      local in_path="${DATA_ROOT}/${scenario}/sce.rds"
      if [[ ! -f "${in_path}" ]]; then
        echo "Skipping missing dataset file: ${in_path}"
        continue
      fi
      processed=$((processed + 1))
      local out_dir="${OUT_PY}/${method}/${SPLIT_NAME}/${scenario}"
      mkdir -p "${out_dir}"
      echo "Processing ${method} ${scenario}..."

      local -a extra_args=()
      if [[ "${method}" == "dca" ]]; then
        extra_args+=(--dca-threads "${DCA_THREADS}")
      elif [[ "${method}" == "alra" ]]; then
        extra_args+=(--alra-max-rank "${ALRA_MAX_RANK}")
      fi

      if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
          numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
          python run_imputation.py "${in_path}" "${out_dir}" "${method}" "${extra_args[@]}" --n-jobs "${MAGIC_JOBS}" --n-repeat "${NREPEATS}"
      else
        CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
          python run_imputation.py "${in_path}" "${out_dir}" "${method}" "${extra_args[@]}" --n-jobs "${MAGIC_JOBS}" --n-repeat "${NREPEATS}"
      fi
    done
    if [[ "${processed}" -eq 0 ]]; then
      echo "No runnable datasets found for PY/${method}. Check DATA_ROOT and SCENARIOS." >&2
      exit 1
    fi
    echo "Finished [PY/${method}]"
  ) > "${log_file}" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("PY/${display_name}")
  JOB_LOGS+=("${log_file}")
}

run_py_gpu_method() {
  local method="$1"
  local gpu="$2"
  local numa_node="$3"
  local log_file="${LOG_DIR}/py_gpu_${method}.log"
  local py_cmd="python"
  local -a extra_args=()
  local display_name
  display_name="$(method_display_name "${method}")"

  if [[ "${method}" == "scvi" ]]; then
    py_cmd="${SCVI_PYTHON}"
    extra_args+=(--scvi-max-epochs "${SCVI_MAX_EPOCHS}")
    extra_args+=(--scvi-batch-size "${SCVI_BATCH_SIZE}")
    extra_args+=(--scvi-latent "${SCVI_LATENT}")
    extra_args+=(--scvi-hidden "${SCVI_HIDDEN}")
    extra_args+=(--scvi-layers "${SCVI_LAYERS}")
  fi

  echo "Started [PY/${display_name}] (method key ${method}, GPU ${gpu}, NUMA ${numa_node:-none}) - logging to ${log_file}"

  (
    local processed=0
    local scenario
    for scenario in "${SCENARIOS[@]}"; do
      local in_path="${DATA_ROOT}/${scenario}/sce.rds"
      if [[ ! -f "${in_path}" ]]; then
        echo "Skipping missing dataset file: ${in_path}"
        continue
      fi
      processed=$((processed + 1))
      local out_dir="${OUT_PY}/${method}/${SPLIT_NAME}/${scenario}"
      mkdir -p "${out_dir}"
      echo "Processing ${method} ${scenario}..."

      if [[ -n "${numa_node}" ]] && command -v numactl >/dev/null 2>&1; then
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_THREADS}" MKL_NUM_THREADS="${GPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${GPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_THREADS}" TORCH_NUM_THREADS="${GPU_THREADS}" \
          numactl --cpunodebind="${numa_node}" --membind="${numa_node}" \
          "${py_cmd}" run_imputation.py "${in_path}" "${out_dir}" "${method}" --n-repeat "${NREPEATS}" "${extra_args[@]}"
      else
        CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${GPU_THREADS}" MKL_NUM_THREADS="${GPU_THREADS}" \
          OPENBLAS_NUM_THREADS="${GPU_THREADS}" NUMEXPR_NUM_THREADS="${GPU_THREADS}" TORCH_NUM_THREADS="${GPU_THREADS}" \
          "${py_cmd}" run_imputation.py "${in_path}" "${out_dir}" "${method}" --n-repeat "${NREPEATS}" "${extra_args[@]}"
      fi
    done
    if [[ "${processed}" -eq 0 ]]; then
      echo "No runnable datasets found for PY/${method}. Check DATA_ROOT and SCENARIOS." >&2
      exit 1
    fi
    echo "Finished [PY/${method}]"
  ) > "${log_file}" 2>&1 &
  JOB_PIDS+=("$!")
  JOB_NAMES+=("PY/${display_name}")
  JOB_LOGS+=("${log_file}")
}

echo "Starting parallel imputation runs on $(hostname)..."
echo "Monitor progress with: tail -f ${LOG_DIR}/*.log"

CPU_R_METHODS=(baseline saver ccimpute)
CPU_PY_METHODS=(magic dca alra)
GPU_METHODS=(autoclass scvi balanced_mse)

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
  echo "All runs finished with ${fail_count} failed method(s)."
  exit 1
fi
echo "All runs completed successfully."
