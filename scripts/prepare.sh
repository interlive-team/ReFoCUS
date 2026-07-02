#!/bin/bash
# Stage 1+2 for the ReFoCUS (main / "refocus baseline") method.
# Builds the selection DATABASE with the trained ReFoCUS selector, then exports the
# ReFoCUS-selected frame indices.  Runs in the SELECTOR env (env A: mamba transformers).
#
# Usage:
#   scripts/prepare.sh <selector_ckpt> <benchmark> [run_name]
# e.g.
#   scripts/prepare.sh interlive/ReFoCUS-1.3b videomme
#
# Tunables (env vars, with the reference defaults):
#   NUM_QUERY_FRAMES=64  NUM_FRAMES=32  CANDIDATE=0  MINFRAMES=128  SORT=1
#   NUM_CANDIDATES=64  TEMP=1.0  SEED=None  MAX_FPS=4.0  MAX_NUM_FRAMES=512
#   WORK_DIR=<repo>/work  NUM_PROCESSES=8
#
# Output frame-index DB path is printed on the last line (feed it to scripts/evaluate.sh).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SELECTOR="${1:?selector checkpoint, e.g. interlive/ReFoCUS-1.3b}"
BENCH="${2:?benchmark task name}"
RUN="${3:-refocus}"
validate_task "$BENCH"

build_database "$SELECTOR" "$BENCH" "$RUN"

NUM_FRAMES="${NUM_FRAMES:-32}"
NUM_QUERY_FRAMES="${NUM_QUERY_FRAMES:-64}"
CANDIDATE="${CANDIDATE:-0}"
MINFRAMES="${MINFRAMES:-128}"
SEL_SAFE="$(safe_id "$SELECTOR")"
DB="$(db_path "$SEL_SAFE" "$RUN" "$BENCH")"
FIDX="$(frameidx_path "$SEL_SAFE" "$RUN" "$BENCH" "cand${CANDIDATE}")"
mkdir -p "$(dirname "$FIDX")"

SORT_FLAG=$([[ "${SORT:-1}" == "1" ]] && echo "--sort" || echo "")
log "Exporting ReFoCUS frame indices -> ${FIDX}"
python "${FRAMEIDX_DIR}/_database_to_frameidx.py" "$DB" "$FIDX" \
  --candidate "$CANDIDATE" --minframes "$MINFRAMES" --numquery "$NUM_QUERY_FRAMES" \
  --numframes "$NUM_FRAMES" $SORT_FLAG

echo "FRAMEIDX=${FIDX}"
