#!/usr/bin/env bash
# Monitor uncover prs3 collection: progress, stall, memory, process health.
# Run in a separate terminal while collection is active:
#   bash audit_outputs/overlap_field_collection/watch_uncover_prs3.sh
#
# Writes heartbeat lines to uncover_watch.log every INTERVAL seconds.
# After an unexpected reboot, check the last heartbeat timestamp vs last Trial Completed.

set -uo pipefail

ROOT="/mnt/data/MudkipUsersSu2025/kpputhuveetil/git/robe/robust-body-exposure"
LOG_DIR="${ROOT}/audit_outputs/overlap_field_collection"
WATCH_LOG="${LOG_DIR}/uncover_watch.log"

ROLLOUTS="${ROLLOUTS:-10000}"
TARGET_LIMBS="${TARGET_LIMBS:-2, 4, 5, 8, 10, 11, 12, 13, 14, 15}"
POST_RELEASE_STEPS="${POST_RELEASE_STEPS:-3}"
INTERVAL="${INTERVAL:-300}"
STALL_MINUTES="${STALL_MINUTES:-45}"

RAW_DIR="${ROOT}/DATASETS/Uncover_Data/TL_${TARGET_LIMBS}_Uncover_Data_${ROLLOUTS}_states_New_Grasp/raw"
COLLECT_LOG="${LOG_DIR}/uncover_10k_prs${POST_RELEASE_STEPS}.log"
MAIN_LOG="${LOG_DIR}/run_collect_uncover_prs3_main.log"

pkl_count() {
  if [[ -d "${RAW_DIR}" ]]; then
    find "${RAW_DIR}" -maxdepth 1 -name '*.pkl' 2>/dev/null | wc -l
  else
    echo 0
  fi
}

worker_count() {
  pgrep -fc "gnn_dc_uncover.py" 2>/dev/null || echo 0
}

gnn_rss_gb() {
  local pids
  pids="$(pgrep -f "gnn_dc_uncover.py" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    echo "0.0"
    return
  fi
  ps -o rss= -p ${pids} 2>/dev/null | awk '{sum+=$1} END {printf "%.1f", sum/1024/1024}'
}

last_trial_line() {
  local log_file="$1"
  if [[ -f "${log_file}" ]]; then
    grep -E '^[0-9]+ - Trial Completed:' "${log_file}" 2>/dev/null | tail -1
  fi
}

log_mtime_iso() {
  local log_file="$1"
  if [[ -f "${log_file}" ]]; then
    stat -c '%y' "${log_file}" 2>/dev/null | cut -d. -f1
  else
    echo "missing"
  fi
}

mem_summary() {
  free -g | awk '/^Mem:/ {printf "mem_used=%sGi mem_avail=%sGi", $3, $7}'
}

prev_count=""
prev_count_ts=""
stall_warned=0

mkdir -p "${LOG_DIR}"
echo "[$(date -Iseconds)] watch started target=${ROLLOUTS} raw=${RAW_DIR}" | tee -a "${WATCH_LOG}"

while true; do
  ts="$(date '+%F %T')"
  count="$(pkl_count)"
  workers="$(worker_count)"
  rss="$(gnn_rss_gb)"
  mem="$(mem_summary)"
  swap="$(free -h | awk '/^Swap:/ {print $3"/"$2}')"
  load="$(awk '{print $1" "$2" "$3}' /proc/loadavg)"
  remain="$((ROLLOUTS - count))"
  trial="$(last_trial_line "${COLLECT_LOG}")"
  if [[ -z "${trial}" ]]; then
    trial="$(last_trial_line "${MAIN_LOG}")"
  fi
  log_mtime="$(log_mtime_iso "${COLLECT_LOG}")"

  line="${ts} | pkls=${count}/${ROLLOUTS} remain=${remain} | workers=${workers} gnn_rss=${rss}GB | ${mem} swap=${swap} load=${load}"
  echo "${line}" | tee -a "${WATCH_LOG}"
  if [[ -n "${trial}" ]]; then
    echo "  last_trial: ${trial}" | tee -a "${WATCH_LOG}"
  fi
  echo "  collect_log_mtime: ${log_mtime}" | tee -a "${WATCH_LOG}"

  avail_gb="$(free -g | awk '/^Mem:/ {print $7}')"
  if [[ "${avail_gb}" -lt 20 ]]; then
    echo "  WARNING: available memory below 20Gi (${avail_gb}Gi)" | tee -a "${WATCH_LOG}"
  fi

  if [[ "${remain}" -gt 0 && "${workers}" -eq 0 ]]; then
    echo "  ALERT: collection incomplete but no gnn_dc_uncover.py process running" | tee -a "${WATCH_LOG}"
  fi

  if [[ -n "${prev_count}" && "${count}" -eq "${prev_count}" && "${remain}" -gt 0 ]]; then
    now_epoch="$(date +%s)"
    prev_epoch="$(date -d "${prev_count_ts}" +%s 2>/dev/null || echo "${now_epoch}")"
    stalled_min="$(( (now_epoch - prev_epoch) / 60 ))"
    if [[ "${stalled_min}" -ge "${STALL_MINUTES}" && "${stall_warned}" -eq 0 ]]; then
      echo "  ALERT: no new pkls for ${stalled_min}min (stall threshold=${STALL_MINUTES}min)" | tee -a "${WATCH_LOG}"
      stall_warned=1
    fi
  else
    prev_count="${count}"
    prev_count_ts="${ts}"
    stall_warned=0
  fi

  if [[ "${remain}" -le 0 ]]; then
    echo "[$(date -Iseconds)] collection complete (${count}/${ROLLOUTS})" | tee -a "${WATCH_LOG}"
    exit 0
  fi

  sleep "${INTERVAL}"
done
