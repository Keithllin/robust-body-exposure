#!/usr/bin/env bash
# Fine-tune an existing recover GNN on accepted low-drag data.
set -euo pipefail

RROOT="${RROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the 70k recover model directory}"
LOWDRAG_ROOT="${LOWDRAG_ROOT:-$RROOT/DATASETS/Recover_Data/TL_LowerBody_lowdrag_unfold_10k/accepted_low_drag_unfolding}"
BASELINE_ROOT="${BASELINE_ROOT:-$RROOT/DATASETS/Recover_Data/TL_All_Recover_mixed_70k_plus_line_nb_20fam}"
EDGE_MODE="${EDGE_MODE:-radius}"
VOXEL_SIZE="${VOXEL_SIZE:-nan}"
EDGE_THRESHOLD="${EDGE_THRESHOLD:-0.04}"
OUTPUT_DIR="${OUTPUT_DIR:-$RROOT/trained_models/FINAL_MODELS/Recover/LowDrag_FT_70k_${EDGE_MODE}_$(date +%Y%m%d_%H%M%S)}"
BASE_EPOCH="${BASE_EPOCH:--1}"
FT_LR="${FT_LR:-2e-5}"
FT_EPOCHS="${FT_EPOCHS:-50}"
REPLAY_RATIO="${REPLAY_RATIO:-0.5}"
NPROC="${NPROC:-4}"
NUM_WORKERS="${NUM_WORKERS:-8}"

cd "$RROOT"
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
if command -v conda >/dev/null 2>&1; then
  conda activate "${ROBE_CONDA_ENV:-robe}" 2>/dev/null || true
fi
export PYTHONPATH="$RROOT/code:$RROOT/assistive-gym-fem:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

python3 -u code/finetune_recover_low_drag.py \
  --base-model "$BASE_MODEL" \
  --lowdrag-root "$LOWDRAG_ROOT" \
  --baseline-root "$BASELINE_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --base-epoch "$BASE_EPOCH" \
  --holdout-fraction 0.20 \
  --baseline-replay-ratio "$REPLAY_RATIO" \
  --baseline-heldout-count 1000 \
  --state-split-seed 1001 \
  --baseline-sample-seed 2001 \
  --process-workers "$NPROC" \
  --num-workers "$NUM_WORKERS" \
  --batch-size 50 \
  --epochs "$FT_EPOCHS" \
  --learning-rate "$FT_LR" \
  --edge-threshold "$EDGE_THRESHOLD" \
  --voxel-size "$VOXEL_SIZE" \
  --edge-mode "$EDGE_MODE"
