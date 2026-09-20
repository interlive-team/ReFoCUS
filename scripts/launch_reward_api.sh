#!/bin/bash
# Launches one InternVL3 reward-API server per GPU for ReFoCUS training and
# waits until all of them report healthy. The servers keep running after this
# script exits; their pids and /vqa urls are written under API_DIR so that
# train_refocus.sh can be pointed at them:
#
#   uv run bash scripts/launch_reward_api.sh
#   REMOTE_EVAL_URLS="$(cat work/reward_api/urls.txt)" STAGE=1 ... uv run bash scripts/train_refocus.sh
#   xargs -I{} kill -9 -- -{} < work/reward_api/pids.txt
#
# Each server is started in its own process group (pid == pgid) because it forks
# workers that share the listening socket and ignore SIGTERM; kill the group.
#
# Optional env:
#   REWARD_MODEL  e.g. OpenGVLab/InternVL3-8B      (default: OpenGVLab/InternVL3-2B, the paper's)
#   GPUS        e.g. "0,1,2,3,4,5,6,7"           (default: all visible)
#   BASE_PORT   server i listens on BASE_PORT+i  (default: 34500)
#   API_DIR     logs, pids.txt, urls.txt         (default: <repo>/work/reward_api)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DECORD_EOF_RETRY_MAX=102400

REWARD_MODEL="${REWARD_MODEL:-OpenGVLab/InternVL3-2B}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
BASE_PORT="${BASE_PORT:-34500}"
API_DIR="${API_DIR:-${REPO_ROOT}/work/reward_api}"
HEALTH_TIMEOUT=600
mkdir -p "${API_DIR}"

IFS=',' read -r -a DEVICES <<< "${GPUS}"
PIDS=()
URLS=()
for i in "${!DEVICES[@]}"; do
  DEV="${DEVICES[$i]}"
  PORT=$((BASE_PORT + i))
  if ss -ltn | awk '{print $4}' | grep -Eq "[:.]${PORT}\$"; then
    echo "port ${PORT} is already in use" >&2
    exit 1
  fi
  echo ">> GPU ${DEV} -> http://localhost:${PORT} (${REWARD_MODEL})"
  CUDA_VISIBLE_DEVICES="${DEV}" setsid nohup python "${REPO_ROOT}/train/reward_api/InternVL2_5.py" \
    --model "${REWARD_MODEL}" \
    --max_batch_size 2 \
    --max_delay 0.1 \
    --port "${PORT}" \
    > "${API_DIR}/api_gpu${DEV}.log" 2>&1 &
  PIDS+=("$!")
  URLS+=("http://localhost:${PORT}/vqa")
done
printf '%s\n' "${PIDS[@]}" > "${API_DIR}/pids.txt"
echo "${URLS[*]}" > "${API_DIR}/urls.txt"

echo ">> Waiting for ${#DEVICES[@]} servers (timeout ${HEALTH_TIMEOUT}s)"
start=$(date +%s)
for i in "${!DEVICES[@]}"; do
  until curl --silent --fail "http://localhost:$((BASE_PORT + i))/health" > /dev/null; do
    if (( $(date +%s) - start >= HEALTH_TIMEOUT )); then
      echo "servers not ready in ${HEALTH_TIMEOUT}s; see ${API_DIR}/api_gpu*.log" >&2
      for pid in "${PIDS[@]}"; do kill -9 -- "-${pid}" 2>/dev/null || true; done
      exit 1
    fi
    sleep 4
  done
done
echo ">> All reward-API servers are up"
echo "REMOTE_EVAL_URLS=\"$(cat "${API_DIR}/urls.txt")\""
