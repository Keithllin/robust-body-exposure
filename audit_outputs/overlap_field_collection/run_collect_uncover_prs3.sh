#!/usr/bin/env bash
set -eo pipefail

ROOT="/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure"
LOG_DIR="${ROOT}/audit_outputs/overlap_field_collection"
mkdir -p "${LOG_DIR}"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robe
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"

cd "${ROOT}"

NUM_PROCESSES="${NUM_PROCESSES:-16}"
ROLLOUTS="${ROLLOUTS:-10000}"
POST_RELEASE_STEPS="${POST_RELEASE_STEPS:-3}"
TARGET_LIMBS="${TARGET_LIMBS:-2, 4, 5, 8, 10, 11, 12, 13, 14, 15}"

DATASET_DIR="${ROOT}/DATASETS/Uncover_Data/TL_${TARGET_LIMBS}_Uncover_Data_${ROLLOUTS}_states_New_Grasp"
RAW_DIR="${DATASET_DIR}/raw"

log() {
  echo "[$(date -Iseconds)] $*"
}

count_pkls() {
  local raw_dir="$1"
  if [[ -d "${raw_dir}" ]]; then
    find "${raw_dir}" -maxdepth 1 -name '*.pkl' | wc -l
  else
    echo 0
  fi
}

log "Config: rollouts=${ROLLOUTS}, post_release_steps=${POST_RELEASE_STEPS}, num_processes=${NUM_PROCESSES}"
log "Dataset: ${DATASET_DIR}"

EXISTING="$(count_pkls "${RAW_DIR}")"
REMAIN="$((ROLLOUTS - EXISTING))"
log "Existing=${EXISTING}/${ROLLOUTS}, remaining=${REMAIN}"

if [[ "${REMAIN}" -le 0 ]]; then
  log "Already complete: ${EXISTING}/${ROLLOUTS} in ${RAW_DIR}"
  exit 0
fi

log "Starting uncover collection (remaining=${REMAIN})"
python3 assistive-gym-fem/assistive_gym/gnn_dc_uncover.py \
  --env RobeReversible-v1 \
  --rollouts "${REMAIN}" \
  --target_limb_list "${TARGET_LIMBS}" \
  --post-release-steps "${POST_RELEASE_STEPS}" \
  --num-processes "${NUM_PROCESSES}" \
  --output-dataset-dir "${DATASET_DIR}" \
  2>&1 | tee -a "${LOG_DIR}/uncover_10k_prs${POST_RELEASE_STEPS}.log"

log "Complete: $(count_pkls "${RAW_DIR}") pkls in ${RAW_DIR}"
