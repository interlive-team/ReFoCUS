#!/bin/bash
# Uniform-sampling baseline (the default-model setting): ignores the ReFoCUS selection
# scores and samples frames uniformly over each video's range.  Still needs the per-sample
# frame range, so it reuses (or builds, idempotently) the selection DB.
#
# Usage:  scripts/prepare_for_uniform.sh <selector_ckpt> <benchmark> [run_name]
# Tunables: NUM_FRAMES=32  MODE=end|mid   (+ the build_database tunables)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SELECTOR="${1:?selector checkpoint (used only to materialize per-sample frame ranges)}"
BENCH="${2:?benchmark}"; RUN="${3:-refocus}"
validate_task "$BENCH"
build_database "$SELECTOR" "$BENCH" "$RUN"

NUM_FRAMES="${NUM_FRAMES:-32}"; MODE="${MODE:-end}"
TAG="uniform_${MODE}_nf${NUM_FRAMES}"
SEL_SAFE="$(safe_id "$SELECTOR")"
DB="$(db_path "$SEL_SAFE" "$RUN" "$BENCH")"; FIDX="$(frameidx_path "$SEL_SAFE" "$RUN" "$BENCH" "$TAG")"
mkdir -p "$(dirname "$FIDX")"

log "Exporting uniform (${MODE}) frame indices -> ${FIDX}"
python "${FRAMEIDX_DIR}/_database_to_frameidx_uniform.py" "$DB" "$FIDX" \
  --numframes "$NUM_FRAMES" --mode "$MODE"
echo "FRAMEIDX=${FIDX}"
