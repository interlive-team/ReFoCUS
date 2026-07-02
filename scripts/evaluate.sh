#!/bin/bash
# Stage 3: run a VLM on a precomputed frame-index DB and score the benchmark.
# Runs in the VLM-eval env (env B: modern transformers).  The VLM repo is the FIRST arg;
# it is mapped to the matching HF_FS_* frame-selection wrapper + any model-specific args.
#
# Usage:
#   scripts/evaluate.sh <selector_ckpt> <benchmark> <run_name> <vlm_repo> [extra_model_args]
# e.g.
#   scripts/evaluate.sh interlive/ReFoCUS-1.3b videomme refocus llava-hf/llava-onevision-qwen2-7b-ov-hf
#   BLUR_SIGMA=50 scripts/evaluate.sh interlive/ReFoCUS-1.3b videomme refocus OpenGVLab/InternVL3-8B-hf   # blur run
#
# <selector_ckpt> <benchmark> <run_name> match the prepare.sh call; the frame-index DB it
# produced is located automatically.
#
# Supported VLM families (see _common.sh vlm_spec): LLaVA-OneVision, InternVL3/3.5,
# Qwen3-VL, Qwen2.5-VL, VideoLLaMA3.  GPT-judge tasks (activitynetqa, videochatgpt)
# additionally require OPENAI_API_KEY and HF_HOME in the environment.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

SELECTOR="${1:?selector checkpoint used in prepare.sh, e.g. interlive/ReFoCUS-1.3b}"
BENCH="${2:?benchmark}"
RUN="${3:?run name used in prepare.sh}"
VLM="${4:?VLM repo id, e.g. llava-hf/llava-onevision-qwen2-7b-ov-hf}"
shift 4
EXTRA="${*:-}"
validate_task "$BENCH"

# Locate the frame-index DB that prepare.sh built for this <selector>/<run>/<benchmark>.
CANDIDATE="${CANDIDATE:-0}"
SEL_SAFE="$(safe_id "$SELECTOR")"
FIDX="$(frameidx_path "$SEL_SAFE" "$RUN" "$BENCH" "cand${CANDIDATE}")"
[[ -f "${FIDX}.db" || -f "${FIDX}.dir" || -f "${FIDX}" ]] || {
  echo "[ERROR] frame-index DB not found: ${FIDX}"
  echo "        build it first: scripts/prepare.sh ${SELECTOR} ${BENCH} ${RUN}"
  exit 1
}

spec="$(vlm_spec "$VLM")"; FS="${spec%%|*}"; BASEARGS="${spec#*|}"
[[ "$FS" == "ERROR" ]] && { echo "[ERROR] no HF_FS wrapper mapped for VLM '$VLM' (edit vlm_spec in _common.sh)"; exit 1; }

# blur mode: switch to the *_blur wrapper and pass blur_sigma
RUN_KIND="eval"
if [[ -n "${BLUR_SIGMA:-}" ]]; then
  case "$FS" in
    HF_FS_qwen2_5vl) echo "[ERROR] no blur variant for $FS"; exit 1 ;;
  esac
  FS="${FS}_blur"; BASEARGS="${BASEARGS:+$BASEARGS,}blur_sigma=${BLUR_SIGMA}"; RUN_KIND="blur"
fi

# assemble model_args = pretrained + (mapped base args) + frameidx/backup + user extras
TAG="$(basename "$(dirname "$FIDX")")"
SAFE_VLM="${VLM//\//_}"
BACKUP="$(backup_path "$SAFE_VLM" "$BENCH" "${TAG}_${RUN_KIND}")"
OUT="$(results_path "$SAFE_VLM" "$BENCH" "${TAG}_${RUN_KIND}")/"
mkdir -p "$(dirname "$BACKUP")" "$OUT"

MODEL_ARGS="pretrained=${VLM}"
[[ -n "$BASEARGS" ]] && MODEL_ARGS="${MODEL_ARGS},${BASEARGS}"
MODEL_ARGS="${MODEL_ARGS},frameidx_file=${FIDX},backup_file=${BACKUP}"
[[ -n "$EXTRA" ]] && MODEL_ARGS="${MODEL_ARGS},${EXTRA}"

log "Evaluating ${VLM} (${FS}) on ${BENCH}; frameidx=${FIDX}"
accelerate launch --num_processes "${NUM_PROCESSES}" \
  --main_process_port $((12340 + RANDOM % 10000)) -m lmms_eval \
  --model "$FS" \
  --model_args "$MODEL_ARGS" \
  --tasks "$BENCH" \
  --batch_size 1 --log_samples --log_samples_suffix "${SAFE_VLM}_${BENCH}_${TAG}_${RUN_KIND}" \
  --output_path "$OUT" --verbosity INFO
log "Done -> ${OUT}"
