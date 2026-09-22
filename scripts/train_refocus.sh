#!/bin/bash
# ReFoCUS training launcher (FPO / frame-preference optimization).
# One invocation = one curriculum stage; run STAGE=1..4 in order. Stage 1 starts
# from the base video-mamba LM, later stages resume from the previous stage's output.
#
# Required env:
#   STAGE              curriculum stage, 1..4
#   DATA_ROOT          dir holding train_stage{1..4}.json (scripts/prepare_train_data.py output)
#   VIDEO_FOLDER       transcoded video root (scripts/download_videos.py output)
#   REMOTE_EVAL_URLS   space-separated reward-API /vqa endpoints (scripts/launch_reward_api.sh)
# Optional env:
#   OUTPUT_ROOT        parent dir of per-stage outputs    (default: <repo>/work/train)
#   RUN_PREFIX         per-stage run name prefix          (default: refocus-1.3b)
#   BASE_MODEL         stage-1 initialization             (default: interlive/video-ma2mba-1.3b)
#   GPUS               e.g. "0,1,2,3,4,5,6,7"             (default: all visible)
#   REPORT_TO          "wandb" to log (with WANDB_API_KEY / WANDB_PROJECT)
#   HUB_MODEL_ID       with HF_TOKEN, pushes the final checkpoint
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DECORD_EOF_RETRY_MAX=102400

STAGE="${STAGE:?set STAGE to 1..4}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding train_stage1..4.json}"
VIDEO_FOLDER="${VIDEO_FOLDER:?set VIDEO_FOLDER to the transcoded video root}"
read -r -a REMOTE_EVAL_URLS <<< "${REMOTE_EVAL_URLS:?set REMOTE_EVAL_URLS to space-separated reward-API /vqa endpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/work/train}"
RUN_PREFIX="${RUN_PREFIX:-refocus-1.3b}"
BASE_MODEL="${BASE_MODEL:-interlive/video-ma2mba-1.3b}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
DEEPSPEED_CFG="${DEEPSPEED_CFG:-${REPO_ROOT}/scripts/zero1.json}"
REPORT_TO="${REPORT_TO:-none}"

# Search-space scaling curriculum: the selected-frame count T' doubles per stage
# while the candidate count N halves; the entropy bonus and the LR of the
# re-initialized parts are lowered as the space grows.
case "${STAGE}" in
  1) QUERY_FRAMES=4;  NUM_CANDIDATES=256; ENTROPY_COEF=1e-3; ADHOC_LR=6e-5 ;;
  2) QUERY_FRAMES=8;  NUM_CANDIDATES=128; ENTROPY_COEF=3e-4; ADHOC_LR=3e-5 ;;
  3) QUERY_FRAMES=16; NUM_CANDIDATES=64;  ENTROPY_COEF=2e-4; ADHOC_LR=2e-5 ;;
  4) QUERY_FRAMES=32; NUM_CANDIDATES=32;  ENTROPY_COEF=1e-4; ADHOC_LR=2e-5 ;;
  *) echo "STAGE must be 1..4, got '${STAGE}'" >&2; exit 1 ;;
esac

# Each stage trains one epoch on its own stride split of the sample pool.
DATA_PATH="${DATA_ROOT}/train_stage${STAGE}.json"
RUN_NAME="${RUN_PREFIX}-stage${STAGE}"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
if [[ "${STAGE}" == 1 ]]; then
  MODEL="${BASE_MODEL}"
  MODEL_REINIT=True    # re-initializes the last 18 backbone blocks and the selection heads
else
  MODEL="${OUTPUT_ROOT}/${RUN_PREFIX}-stage$((STAGE - 1))"
  MODEL_REINIT=False
fi

HUB_ARGS=()
if [[ -n "${HUB_MODEL_ID:-}" ]]; then
  HUB_ARGS=(--push_to_hub True --hub_model_id "${HUB_MODEL_ID}" --hub_private_repo True)
fi

# Launched as a module: running train/train_mem.py by path puts train/ first on
# sys.path, where train.py shadows the train package.
# --model_reinit_backbone_layers is passed on every stage so the value stays in
# config.json; it only acts when --model_reinit is True.
deepspeed --include "localhost:${GPUS}" --module train.train_mem \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL}" \
  --version fpo \
  --vision_tower google/siglip-so400m-patch14-384 \
  --mm_tunable_parts mm_fpo_adhoc,mm_language_model,mm_mlp_adapter \
  --model_reinit "${MODEL_REINIT}" \
  --model_reinit_backbone_layers 18 \
  --mm_vision_select_layer -2 \
  --mm_projector_type mlp2x_gelu \
  --mm_spatial_pool_mode bilinear \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --query_frames "${QUERY_FRAMES}" \
  --value_embedding_init_range 0.02 \
  --value_head_init_gain 1.0 \
  --value_head_norm rmsnorm \
  --value_head_scale 0.02 \
  --score_head_init_gain 1.0 \
  --score_head_norm rmsnorm \
  --logit_scale_fn exp \
  --logit_scale_init 1.0 \
  --logit_scale_type vector \
  --mixer_norm_rescale 0.1 \
  --frame_generation_method autoregressive \
  --attention_scaling True \
  --frame_replacement False \
  --use_rank_head False \
  --data_path "${DATA_PATH}" \
  --video_folder "${VIDEO_FOLDER}" \
  --video_folder_remote "${VIDEO_FOLDER}" \
  --lazy_preprocess True \
  --frames_upbound 512 \
  --video_fps 4 \
  --train_step fpo \
  --join_clips True \
  --shuffle_options shared \
  --num_candidates "${NUM_CANDIDATES}" \
  --group_by_varlen True \
  --remote_evaluation_urls "${REMOTE_EVAL_URLS[@]}" \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --mm_fpo_adhoc_lr "${ADHOC_LR}" \
  --adam_beta1 0.9 \
  --adam_beta2 0.99 \
  --adam_epsilon 1e-8 \
  --optim adamw_torch \
  --weight_decay 0 \
  --warmup_steps 256 \
  --lr_scheduler_type cosine_with_min_lr \
  --lr_scheduler_kwargs "{\"min_lr_rate\": 0.4}" \
  --entropy_coef "${ENTROPY_COEF}" \
  --reward_modeling probability-margin \
  --reward_normalize normalize \
  --reinforce_scale step \
  --score_epsilon 1e-4 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 256 \
  --save_total_limit 4 \
  --logging_steps 1 \
  --report_to "${REPORT_TO}" \
  "${HUB_ARGS[@]}" \
  --deepspeed "${DEEPSPEED_CFG}" \
  --ddp_timeout 120 \
  --dataloader_num_workers 4 \
  --dataloader_prefetch_factor 3 \
  --bf16 True \
  --model_max_length 1048576
