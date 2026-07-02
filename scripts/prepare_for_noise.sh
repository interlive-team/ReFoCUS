#!/bin/bash
# Prefix noisy-order injection — BUILD-TIME "force index" method.
# The selector is forced to place random frames into the prefix of each candidate
# (candidate i -> i+1 forced random prefix frames) and then selects the rest around them.
# This builds a SEPARATE noise database with the force-index selector (HF_FS_refocus_noise),
# then exports indices exactly like the refocus method (`_database_to_frameidx.py`).
#
# Usage:  scripts/prepare_for_noise.sh <selector_ckpt> <benchmark> [run_name]
# Tunables:
#   CANDIDATE=0          noise level: candidate index = (#forced prefix frames - 1)
#   NUM_QUERY_FRAMES=64  NUM_FRAMES=32  MINFRAMES=128  SORT=1   (+ build_database tunables)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SELECTOR="${1:?selector checkpoint}"; BENCH="${2:?benchmark}"; RUN="${3:-refocus}_noise"
validate_task "$BENCH"

# build a noise-injected DB with the force-index selector (distinct from the refocus DB)
build_database "$SELECTOR" "$BENCH" "$RUN" HF_FS_refocus_noise

CANDIDATE="${CANDIDATE:-0}"; MINFRAMES="${MINFRAMES:-128}"
NUM_FRAMES="${NUM_FRAMES:-32}"; NUM_QUERY_FRAMES="${NUM_QUERY_FRAMES:-64}"
TAG="noise_cand${CANDIDATE}_nq${NUM_QUERY_FRAMES}_nf${NUM_FRAMES}"
SEL_SAFE="$(safe_id "$SELECTOR")"
DB="$(db_path "$SEL_SAFE" "$RUN" "$BENCH")"; FIDX="$(frameidx_path "$SEL_SAFE" "$RUN" "$BENCH" "$TAG")"
mkdir -p "$(dirname "$FIDX")"
SORT_FLAG=$([[ "${SORT:-1}" == "1" ]] && echo "--sort" || echo "")

log "Exporting prefix-noise (force-index) frame indices, candidate=${CANDIDATE} -> ${FIDX}"
python "${FRAMEIDX_DIR}/_database_to_frameidx.py" "$DB" "$FIDX" \
  --candidate "$CANDIDATE" --minframes "$MINFRAMES" \
  --numquery "$NUM_QUERY_FRAMES" --numframes "$NUM_FRAMES" $SORT_FLAG
echo "FRAMEIDX=${FIDX}"
