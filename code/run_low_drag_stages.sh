#!/usr/bin/env bash
# Staged launcher for lower-body constrained low-drag unfolding collection.
#
# Accept = not-dragging (ΔE_upper<=tau_final) + not too_short + grasp ok.
# No-ops (low ΔC_lower) are ACCEPTED by default (unfolding-not-dragging goal).
# Cutoff = stop-in-place (no EE pullback); tau_online only stops translate.
set -euo pipefail

RROOT="${RROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STAGE="${1:-0}"   # 0 | 1 | 2 | 3 | viz
POOL="${POOL:-$RROOT/DATASETS/Recover_Data/recover_source_varaware}"
MANIFEST="${MANIFEST:-$POOL/cma_evaluations/Combined_1k_plus_boost_varaware/source_manifest.jsonl}"
RUN_ID="${RUN_ID:-lowdrag_lb}"
TAU_ONLINE="${TAU_ONLINE:-2}"
TAU_FINAL="${TAU_FINAL:-30}"
ETA="${ETA:-3}"          # diagnostic only unless REQUIRE_LOWER_GAIN=1
LMIN="${LMIN:-0.02}"
NPROC="${NPROC:-16}"
MONITOR_STRIDE="${MONITOR_STRIDE:-1}"
# Per-state noise count (plus 1 exact). Override with N_NOISE=...
N_NOISE="${N_NOISE:-}"

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

collect() {
  local out="$1"
  local max_states="$2"
  local max_accepted="$3"
  local nproc="$4"
  local n_noise="$5"
  mkdir -p "$out"
  python3 -u code/collect_lower_body_low_drag.py \
    --pool-root "$POOL" \
    --manifest "$MANIFEST" \
    --output-dir "$out" \
    --run-id "$RUN_ID" \
    --target-limbs 4 5 10 11 12 14 \
    --max-states "$max_states" \
    --max-accepted "$max_accepted" \
    --n-noise "$n_noise" \
    --tau-online "$TAU_ONLINE" \
    --tau-final "$TAU_FINAL" \
    --eta-lower-gain "$ETA" \
    --min-action-length "$LMIN" \
    --monitor-stride "$MONITOR_STRIDE" \
    --uncover-prs 3 \
    --recover-prs 3 \
    --num-processes "$nproc" \
    --record-trajectory \
    --resume
}

viz() {
  local ds="$1"
  python3 -u code/viz_low_drag_pilot.py --dataset-dir "$ds" --output-dir "$ds/viz"
}

case "$STAGE" in
  0)
    OUT="$RROOT/DATASETS/Recover_Data/TL_LowerBody_lowdrag_unfold_stage0_v2"
    collect "$OUT" 8 0 "$NPROC" "${N_NOISE:-3}"
    viz "$OUT"
    echo "Stage 0 done: $OUT"
    ;;
  1)
    # Visual / threshold pilot (~100 states x 4 acts). Already largely done.
    OUT="$RROOT/DATASETS/Recover_Data/TL_LowerBody_lowdrag_unfold_stage1"
    collect "$OUT" 100 0 "$NPROC" "${N_NOISE:-3}"
    viz "$OUT"
    echo "Stage 1 done: $OUT"
    ;;
  2)
    # Distribution pilot: denser noise, ~2k accepted, measure accept rate before 10k.
    # 682 lower states x (1+15) ≈ 11k attempts; expect ~7–9k if rate~70–80%, stop at 2k.
    OUT="$RROOT/DATASETS/Recover_Data/TL_LowerBody_lowdrag_unfold_stage2"
    collect "$OUT" 0 2000 "$NPROC" "${N_NOISE:-15}"
    viz "$OUT"
    echo "Stage 2 done: $OUT"
    echo "Check accept rate = accepted / (accepted+dragging+...). If <~50%, raise N_NOISE for stage 3."
    ;;
  3)
    # Full ~10k accepted. 682 x (1+20) ≈ 14.3k attempts → ~10k at ~70% accept.
    OUT="$RROOT/DATASETS/Recover_Data/TL_LowerBody_lowdrag_unfold_10k"
    collect "$OUT" 0 10000 "$NPROC" "${N_NOISE:-20}"
    echo "Stage 3 collection finished: $OUT"
    echo "If short of 10k: rerun with N_NOISE=28 RUN_ID=lowdrag_lb_b (or --resume after raising n-noise needs new run_id slots)"
    ;;
  viz)
    DS="${2:?usage: $0 viz /path/to/dataset}"
    viz "$DS"
    ;;
  *)
    echo "Usage: $0 [0|1|2|3|viz <dataset_dir>]"
    echo "Env: TAU_ONLINE TAU_FINAL ETA LMIN MONITOR_STRIDE NPROC N_NOISE RUN_ID POOL MANIFEST RROOT"
    echo "Defaults: tau_final=30, no-ops accepted, stage2 n_noise=15, stage3 n_noise=20, NPROC=16"
    exit 1
    ;;
esac
