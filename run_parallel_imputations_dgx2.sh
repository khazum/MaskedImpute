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

RSCRIPT_CMD=(conda run -n r45_bio Rscript)

BASE_DIR="${BASE_DIR:-synthetic_datasets/rds_splat_output}"
OUT_R="${OUT_R:-results_imputation_r}"
OUT_PY="${OUT_PY:-results_imputation_py}"
SIZES=(${SIZES:-1000 5000 10000 15000 20000 25000 50000 75000 100000})
LOG_DIR="${LOG_DIR:-logs_parallel_runs}"

mkdir -p "$LOG_DIR"

NREPEATS="${NREPEATS:-4}"
PARALLEL_DATASETS="${PARALLEL_DATASETS:-4}"
PROCESS_CPUS="${PROCESS_CPUS:-8}"
DCA_THREADS="${DCA_THREADS:-$PROCESS_CPUS}"
SAVER_CORES="${SAVER_CORES:-8}"
NUMA_NODE="${NUMA_NODE:-0}"
METHODS_RAW="${METHODS:-dca,saver}"

if ! help wait 2>/dev/null | grep -q -- "-n"; then
  echo "This script requires bash support for 'wait -n' and 'wait -p'." >&2
  exit 2
fi

parse_methods() {
  local raw="$1"
  if [[ -z "$raw" || "$raw" == "all" ]]; then
    echo "dca saver"
    return
  fi

  local -a methods=()
  IFS=',' read -r -a methods <<< "$raw"
  local -a out=()
  local m
  for m in "${methods[@]}"; do
    m="${m//[[:space:]]/}"
    [[ -z "$m" ]] && continue
    case "$m" in
      dca|saver)
        out+=("$m")
        ;;
      *)
        echo "Unknown method '$m'. Allowed: dca,saver or all." >&2
        exit 1
        ;;
    esac
  done

  if [[ "${#out[@]}" -eq 0 ]]; then
    echo "No valid methods selected." >&2
    exit 1
  fi
  echo "${out[*]}"
}

read -r -a METHODS <<< "$(parse_methods "$METHODS_RAW")"

echo "Methods: ${METHODS[*]}"
echo "Repeats: $NREPEATS"
echo "Parallel datasets per method: $PARALLEL_DATASETS"
echo "CPU threads per process: $PROCESS_CPUS"
echo "DCA threads: $DCA_THREADS"
echo "SAVER cores: $SAVER_CORES"
echo "NUMA node: $NUMA_NODE"
echo "Rscript: ${RSCRIPT_CMD[*]}"

run_with_affinity() {
  local -a cmd=("$@")
  if command -v numactl >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS="${PROCESS_CPUS}" MKL_NUM_THREADS="${PROCESS_CPUS}" \
    OPENBLAS_NUM_THREADS="${PROCESS_CPUS}" NUMEXPR_NUM_THREADS="${PROCESS_CPUS}" \
    numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
    "${cmd[@]}"
  else
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS="${PROCESS_CPUS}" MKL_NUM_THREADS="${PROCESS_CPUS}" \
    OPENBLAS_NUM_THREADS="${PROCESS_CPUS}" NUMEXPR_NUM_THREADS="${PROCESS_CPUS}" \
    "${cmd[@]}"
  fi
}

wait_for_one() {
  local method="$1"
  local -n pids_ref="$2"
  local -n names_ref="$3"
  local -n logs_ref="$4"
  local -n fails_ref="$5"

  local done_pid=""
  local status=0
  if wait -n -p done_pid; then
    status=0
  else
    status=$?
  fi

  if [[ -z "$done_pid" ]]; then
    echo "[${method}] Internal error: wait -n returned empty PID." >&2
    fails_ref=$((fails_ref + 1))
    return
  fi

  local idx=-1
  local i
  for i in "${!pids_ref[@]}"; do
    if [[ "${pids_ref[$i]}" == "$done_pid" ]]; then
      idx="$i"
      break
    fi
  done

  if [[ "$idx" -lt 0 ]]; then
    echo "[${method}] Internal warning: finished PID ${done_pid} not tracked." >&2
    fails_ref=$((fails_ref + 1))
    return
  fi

  local name="${names_ref[$idx]}"
  local log_file="${logs_ref[$idx]}"
  if [[ "$status" -eq 0 ]]; then
    echo "[OK] ${name} completed."
  else
    fails_ref=$((fails_ref + 1))
    echo "[FAIL] ${name} exited with status ${status}. See ${log_file}"
  fi

  unset 'pids_ref[idx]'
  unset 'names_ref[idx]'
  unset 'logs_ref[idx]'
  pids_ref=("${pids_ref[@]}")
  names_ref=("${names_ref[@]}")
  logs_ref=("${logs_ref[@]}")
}

run_dca_controller() {
  local -a pids=()
  local -a names=()
  local -a logs=()
  local fail_count=0

  local size
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "[dca] Skipping missing dataset folder: ${in_dir}"
      continue
    fi
    local out_dir="${OUT_PY}/dca/cells_${size}"
    local log_file="${LOG_DIR}/py_dca_cells_${size}.log"
    mkdir -p "${out_dir}"

    (
      echo "Processing dca cells_${size}..."
      run_with_affinity \
        python run_imputation.py "${in_dir}" "${out_dir}" dca \
        --dca-threads "${DCA_THREADS}" \
        --n-repeat "${NREPEATS}"
    ) > "${log_file}" 2>&1 &

    pids+=("$!")
    names+=("PY/dca/cells_${size}")
    logs+=("${log_file}")

    if [[ "${#pids[@]}" -ge "${PARALLEL_DATASETS}" ]]; then
      wait_for_one "dca" pids names logs fail_count
    fi
  done

  while [[ "${#pids[@]}" -gt 0 ]]; do
    wait_for_one "dca" pids names logs fail_count
  done

  if [[ "${fail_count}" -gt 0 ]]; then
    echo "[dca] finished with ${fail_count} failed dataset run(s)."
    return 1
  fi
  echo "[dca] completed successfully."
}

run_saver_controller() {
  local -a pids=()
  local -a names=()
  local -a logs=()
  local fail_count=0

  local size
  for size in "${SIZES[@]}"; do
    local in_dir="${BASE_DIR}/cells_${size}"
    if [[ ! -d "${in_dir}" ]]; then
      echo "[saver] Skipping missing dataset folder: ${in_dir}"
      continue
    fi
    local out_dir="${OUT_R}/saver/cells_${size}"
    local log_file="${LOG_DIR}/r_saver_cells_${size}.log"
    mkdir -p "${out_dir}"

    (
      echo "Processing saver cells_${size}..."
      run_with_affinity \
        "${RSCRIPT_CMD[@]}" run_imputation.R "${in_dir}" "${out_dir}" "${SAVER_CORES}" "${NREPEATS}" saver
    ) > "${log_file}" 2>&1 &

    pids+=("$!")
    names+=("R/saver/cells_${size}")
    logs+=("${log_file}")

    if [[ "${#pids[@]}" -ge "${PARALLEL_DATASETS}" ]]; then
      wait_for_one "saver" pids names logs fail_count
    fi
  done

  while [[ "${#pids[@]}" -gt 0 ]]; do
    wait_for_one "saver" pids names logs fail_count
  done

  if [[ "${fail_count}" -gt 0 ]]; then
    echo "[saver] finished with ${fail_count} failed dataset run(s)."
    return 1
  fi
  echo "[saver] completed successfully."
}

declare -a METHOD_PIDS=()
declare -a METHOD_NAMES=()

start_controller() {
  local method="$1"
  local controller_log="${LOG_DIR}/${method}_controller.log"
  echo "Started [${method}] controller - logging to ${controller_log}"
  (
    if [[ "$method" == "dca" ]]; then
      run_dca_controller
    elif [[ "$method" == "saver" ]]; then
      run_saver_controller
    else
      echo "Unsupported method controller: ${method}" >&2
      exit 1
    fi
  ) > "${controller_log}" 2>&1 &
  METHOD_PIDS+=("$!")
  METHOD_NAMES+=("${method}")
}

for method in "${METHODS[@]}"; do
  start_controller "$method"
done

overall_fail=0
declare -A METHOD_PID_TO_NAME=()
for i in "${!METHOD_PIDS[@]}"; do
  METHOD_PID_TO_NAME["${METHOD_PIDS[$i]}"]="${METHOD_NAMES[$i]}"
done

remaining="${#METHOD_PIDS[@]}"
while [[ "$remaining" -gt 0 ]]; do
  done_pid=""
  if wait -n -p done_pid; then
    status=0
  else
    status=$?
  fi

  name="${METHOD_PID_TO_NAME[$done_pid]:-unknown}"
  if [[ "$status" -eq 0 ]]; then
    echo "[OK] controller ${name} completed."
  else
    overall_fail=$((overall_fail + 1))
    echo "[FAIL] controller ${name} exited with status ${status}. See ${LOG_DIR}/${name}_controller.log"
  fi
  remaining=$((remaining - 1))
done

if [[ "$overall_fail" -gt 0 ]]; then
  echo "All runs finished with ${overall_fail} failed controller(s)."
  exit 1
fi

echo "All requested method controllers completed successfully."
