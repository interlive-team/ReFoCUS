#!/bin/bash
# Blur ablation: uniform-mid frames, but blur the top-k (or k random) most-selected
# temporal segments.  Produces per-sample {indices, blur[]}; the VLM applies the blur at
# eval time (evaluate.sh must set BLUR_SIGMA, which switches to the HF_FS_*_blur model).
#
# Usage:  scripts/prepare_for_blur.sh <selector_ckpt> <benchmark> [run_name]
# Tunables: NUM_FRAMES=32  BLUR_MODE=top_blur|random_blur  K=8  NUMQUERY=64  USEFIRST=64
#           BLUR_SEED=42   (+ build_database tunables)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SELECTOR="${1:?selector checkpoint}"; BENCH="${2:?benchmark}"; RUN="${3:-refocus}"
validate_task "$BENCH"
build_database "$SELECTOR" "$BENCH" "$RUN"

NUM_FRAMES="${NUM_FRAMES:-32}"; BLUR_MODE="${BLUR_MODE:-top_blur}"; K="${K:-8}"
NUMQUERY="${NUMQUERY:-64}"; USEFIRST="${USEFIRST:-64}"; BLUR_SEED="${BLUR_SEED:-42}"
TAG="blur_${BLUR_MODE}_k${K}_nf${NUM_FRAMES}_seed${BLUR_SEED}"
SEL_SAFE="$(safe_id "$SELECTOR")"
DB="$(db_path "$SEL_SAFE" "$RUN" "$BENCH")"; FIDX="$(frameidx_path "$SEL_SAFE" "$RUN" "$BENCH" "$TAG")"
mkdir -p "$(dirname "$FIDX")"

log "Exporting blur frame indices/flags (${BLUR_MODE}, k=${K}) -> ${FIDX}"
python "${FRAMEIDX_DIR}/_database_to_frameidx_for_frame_blur.py" "$DB" "$FIDX" \
  --numframes "$NUM_FRAMES" --mode "$BLUR_MODE" --k "$K" \
  --numquery "$NUMQUERY" --usefirst "$USEFIRST" --seed "$BLUR_SEED"
echo "FRAMEIDX=${FIDX}"
echo "# remember: run evaluate.sh with BLUR_SIGMA set (e.g. BLUR_SIGMA=50) for this frameidx"
