#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-synthetic_datasets/simulated_data/test}"
SPLIT_NAME="${SPLIT_NAME:-test}"
OUT_PY="${OUT_PY:-results_imputation_py}"
LOG_DIR="${LOG_DIR:-logs_parallel_runs/phase4_scvi_alra}"
SCVI_PYTHON="${SCVI_PYTHON:-.venv_scvi/bin/python}"
MAGIC_PYTHON="${MAGIC_PYTHON:-conda run -n magic311 python}"
NREPEATS="${NREPEATS:-5}"
CPU_THREADS="${CPU_THREADS:-8}"
SCVI_GPUS="${SCVI_GPUS:-0,1,2,3}"
SCVI_MAX_EPOCHS="${SCVI_MAX_EPOCHS:-400}"
SCVI_BATCH_SIZE="${SCVI_BATCH_SIZE:-256}"
SCVI_LATENT="${SCVI_LATENT:-10}"
SCVI_HIDDEN="${SCVI_HIDDEN:-128}"
SCVI_LAYERS="${SCVI_LAYERS:-2}"
ALRA_MAX_RANK="${ALRA_MAX_RANK:-100}"
ALRA_JOBS="${ALRA_JOBS:-4}"

mkdir -p "${LOG_DIR}"

mapfile -t SCENARIOS < <(find "${DATA_ROOT}" -mindepth 1 -maxdepth 1 -type d -name '*' | sort | xargs -r -n1 basename)
if [[ "${#SCENARIOS[@]}" -eq 0 ]]; then
  echo "No scenarios under ${DATA_ROOT}" >&2
  exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${SCVI_GPUS}"
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "SCVI_GPUS is empty" >&2
  exit 1
fi

echo "Running Phase 4 baselines on ${#SCENARIOS[@]} scenarios"
echo "Data root: ${DATA_ROOT}"
echo "scVI python: ${SCVI_PYTHON}"
echo "ALRA python: ${MAGIC_PYTHON}"
echo "scVI GPUs: ${SCVI_GPUS}"
echo "Repeats: ${NREPEATS}"

run_alra_one() {
  local scenario="$1"
  local in_path="${DATA_ROOT}/${scenario}/sce.rds"
  local out_dir="${OUT_PY}/alra/${SPLIT_NAME}/${scenario}"
  local log_file="${LOG_DIR}/alra_${scenario}.log"
  mkdir -p "${out_dir}"
  echo "[ALRA] ${scenario} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" \
    ${MAGIC_PYTHON} run_imputation.py "${in_path}" "${out_dir}" alra --n-repeat "${NREPEATS}" \
      --alra-max-rank "${ALRA_MAX_RANK}" > "${log_file}" 2>&1
}

run_scvi_one() {
  local scenario="$1"
  local gpu="$2"
  local in_path="${DATA_ROOT}/${scenario}/sce.rds"
  local out_dir="${OUT_PY}/scvi/${SPLIT_NAME}/${scenario}"
  local log_file="${LOG_DIR}/scvi_${scenario}.log"
  mkdir -p "${out_dir}"
  echo "[scVI gpu=${gpu}] ${scenario} -> ${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS="${CPU_THREADS}" MKL_NUM_THREADS="${CPU_THREADS}" \
    OPENBLAS_NUM_THREADS="${CPU_THREADS}" NUMEXPR_NUM_THREADS="${CPU_THREADS}" TORCH_NUM_THREADS="${CPU_THREADS}" \
    "${SCVI_PYTHON}" run_imputation.py "${in_path}" "${out_dir}" scvi --n-repeat "${NREPEATS}" \
      --scvi-max-epochs "${SCVI_MAX_EPOCHS}" --scvi-batch-size "${SCVI_BATCH_SIZE}" \
      --scvi-latent "${SCVI_LATENT}" --scvi-hidden "${SCVI_HIDDEN}" --scvi-layers "${SCVI_LAYERS}" \
      > "${log_file}" 2>&1
}

# Run ALRA with bounded CPU parallelism.
alra_running=0
for scenario in "${SCENARIOS[@]}"; do
  run_alra_one "${scenario}" &
  alra_running=$((alra_running + 1))
  if [[ "${alra_running}" -ge "${ALRA_JOBS}" ]]; then
    wait -n
    alra_running=$((alra_running - 1))
  fi
done
wait

echo "ALRA finished; starting scVI"

# Run scVI with one process per listed GPU.
declare -a PIDS=()
declare -a NAMES=()
idx=0
active=0
for scenario in "${SCENARIOS[@]}"; do
  gpu="${GPU_IDS[$((idx % ${#GPU_IDS[@]}))]}"
  run_scvi_one "${scenario}" "${gpu}" &
  PIDS+=("$!")
  NAMES+=("${scenario}")
  idx=$((idx + 1))
  active=$((active + 1))
  if [[ "${active}" -ge "${#GPU_IDS[@]}" ]]; then
    wait -n
    active=$((active - 1))
  fi
done
wait

echo "Phase 4 scVI/ALRA runs complete."
