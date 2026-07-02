#!/bin/bash
# Shared config + helpers for the ReFoCUS "frame-selection -> VLM" evaluation pipeline.
#
# Pipeline (2 stages):
#   Stage 1  build the selection DATABASE with the ReFoCUS selector  (scripts/prepare*.sh)
#   Stage 2  convert DB -> frame indices per method                  (scripts/prepare*.sh)
#   Stage 3  run a VLM on the selected frames                        (scripts/evaluate.sh)
#
# The DB is a python `shelve` keyed by sample, each value = {frame_idx, candidates, ...}
# produced by lmms-eval model `HF_FS_refocus` (the trained ReFoCUS selector).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# `refocus` (the model pkg, used by the selector) + `refocus_eval` (the lmms-eval plugin)
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/eval${PYTHONPATH:+:$PYTHONPATH}"
# registers the HF_FS_* frame-selection models into the (officially installed) lmms-eval
export LMMS_EVAL_PLUGINS="${LMMS_EVAL_PLUGINS:-refocus_eval}"
export DECORD_EOF_RETRY_MAX=102400

WORK_DIR="${WORK_DIR:-${REPO_ROOT}/work}"     # all artifacts live here
NUM_PROCESSES="${NUM_PROCESSES:-8}"            # accelerate data-parallel procs (GPUs)
FRAMEIDX_DIR="${FRAMEIDX_DIR:-${REPO_ROOT}/eval/refocus_eval/frameidx}"

# Benchmarks in scope (lmms-eval task names). activitynetqa/videochatgpt need OPENAI_API_KEY + HF_HOME (GPT judge).
VALID_TASKS="videomme longvideobench_val_v mlvu_dev nextqa_oe_val nextqa_oe_test activitynetqa videochatgpt"
validate_task(){
  for t in ${1//,/ }; do
    case " $VALID_TASKS " in *" $t "*) ;; *) echo "[ERROR] unsupported benchmark: '$t'"; echo "  valid: $VALID_TASKS"; exit 1;; esac
  done
}

safe_id(){ echo "${1//\//_}"; }                                 # repo id / local path -> path-safe segment

db_path(){       echo "${WORK_DIR}/database/$1/$2/$3/DB"; }      # <selector_safe> <run> <bench>
frameidx_path(){ echo "${WORK_DIR}/frameidx/$1/$2/$3/$4/DB"; }  # <selector_safe> <run> <bench> <tag>
backup_path(){   echo "${WORK_DIR}/backups/$1/$2/$3"; }         # <safe_vlm> <bench> <tag>
results_path(){  echo "${WORK_DIR}/results/$1/$2/$3"; }         # <safe_vlm> <bench> <tag>

log(){ echo "[$(date '+%F %T')] $*"; }

# Map a HF VLM repo id -> "<fs_model>|<extra model_args>".  Captures the per-model handling
# that the original eval scripts hard-coded (LLaVA-OV conv/model_name; VideoLLaMA3 token cap).
vlm_spec(){
  case "$1" in
    lmms-lab/llava-onevision*|llava-hf/llava-onevision*) echo "HF_FS_llavaOV|conv_template=qwen_1_5,model_name=llava_qwen" ;;
    OpenGVLab/InternVL*)      echo "HF_FS_internvl2_5|" ;;
    DAMO-NLP-SG/VideoLLaMA3*) echo "HF_FS_videollama3|max_video_tokens=12000" ;;
    Qwen/Qwen3-VL*)           echo "HF_FS_qwen3vl|" ;;
    Qwen/Qwen2.5-VL*|Qwen/Qwen2_5-VL*) echo "HF_FS_qwen2_5vl|" ;;
    *) echo "ERROR|" ;;
  esac
}

# Build the selection DATABASE for one benchmark with the ReFoCUS selector (Stage 1).
# Idempotent: skips if the DB already exists (delete it to rebuild).
# args: <selector_ckpt> <bench> <run_name> [fs_model=HF_FS_refocus]
build_database(){
  local selector="$1" bench="$2" run="$3" fs_model="${4:-HF_FS_refocus}"
  local sel; sel="$(safe_id "$selector")"
  local db; db="$(db_path "$sel" "$run" "$bench")"
  if [[ -f "${db}.dir" || -f "${db}" || -f "${db}.db" ]]; then
    log "DB exists, skip build: ${db}"; return 0
  fi
  mkdir -p "$(dirname "$db")"
  log "Building selection DB: selector=${selector} bench=${bench} -> ${db}"
  SKIP_EVAL=1 accelerate launch --num_processes "${NUM_PROCESSES}" \
    --main_process_port $((12340 + RANDOM % 10000)) -m lmms_eval \
    --model "${fs_model}" \
    --model_args "pretrained=${selector},database_file=${db},max_fps=${MAX_FPS:-4.0},max_num_frames=${MAX_NUM_FRAMES:-512},replace=False,conv_template=fpo,seed=${SEED:-None},temperature=${TEMP:-1.0},num_candidates=${NUM_CANDIDATES:-64},num_query_frames=${NUM_QUERY_FRAMES:-64}" \
    --tasks "${bench}" --batch_size 1 --verbosity INFO \
    --output_path "$(results_path "$run" "$bench" _dbbuild)/"
}
