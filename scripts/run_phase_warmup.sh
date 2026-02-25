#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

MODEL_NAME=${MODEL_NAME:-YOUR_MODEL_PATH} #only required in phase warmup
PHASE=${PHASE:-phase1}
DATA_PATH=${DATA_PATH:-YOUR_DATA_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-YOUR_OUTPUT_PATH}

RESUME_PATH=${RESUME_PATH:-}

CHUNK_SIZE=${CHUNK_SIZE:-16}
TRUNCATE_LEN=${TRUNCATE_LEN:-4096}
RANDOM_GATE=${RANDOM_GATE:-1}

BATCH_SIZE=${BATCH_SIZE:-2}
STEPS=${STEPS:-220}
SAVE_STEPS=${SAVE_STEPS:-30}
LOG_STEPS=${LOG_STEPS:-1}

GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
TRAIN_ONLY_SPECIAL=${TRAIN_ONLY_SPECIAL:-true}
WARMUP_STEPS=${WARMUP_STEPS:-10}
GRAD_PROBE=${GRAD_PROBE:-true}

EVAL_DATA_PATH=${EVAL_DATA_PATH:-YOUR_EVAL_DATA_PATH}
EVAL_STEPS=${EVAL_STEPS:-30}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-0}

LR=${LR:-5e-5}
LR_PROJECTOR=${LR_PROJECTOR:-7e-5}
LR_GENERATOR=${LR_GENERATOR:-0}
LR_COMPRESSOR=${LR_COMPRESSOR:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
CLIP_GRAD_NORM=${CLIP_GRAD_NORM:-1.0}

PARALLEL=${PARALLEL:-fsdp} # none|ddp|fsdp
DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-true}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

WANDB_PROJECT=${WANDB_PROJECT:-Mark-42}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-Mark-42-phase-warmup-deepkv-group-gate}
WANDB_RUN_TAGS=${WANDB_RUN_TAGS:-phase-warmup,deepkv,javis,attn,group,gate}

JAVIS_WARMUP_SAMPLES=${JAVIS_WARMUP_SAMPLES:-20}
JAVIS_WARMUP_SAVE_PATH=${JAVIS_WARMUP_SAVE_PATH:-../javis}
JAVIS_QUERY_NUM=${JAVIS_QUERY_NUM:-2}
JAVIS_QUERY_GROUP_SIZE=${JAVIS_QUERY_GROUP_SIZE:-4}
JAVIS_Q_COS_COEFF=${JAVIS_Q_COS_COEFF:-2.0}
TEACHER_TARGETS_PATH=${TEACHER_TARGETS_PATH:-}
DISTILL_LAYERS=${DISTILL_LAYERS:-}
DISTILL_COEFF=${DISTILL_COEFF:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
args=(
  --model_name "$MODEL_NAME"
  --phase "$PHASE"
  --data_path "$DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --chunk_size "$CHUNK_SIZE"
  --batch_size "$BATCH_SIZE"
  --lr "$LR"
  --steps "$STEPS"
  --save_steps "$SAVE_STEPS"
  --log_steps "$LOG_STEPS"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --warmup_steps "$WARMUP_STEPS"
  --weight_decay "$WEIGHT_DECAY"
  --parallel "$PARALLEL"
  --ddp_find_unused_parameters "$DDP_FIND_UNUSED"
  --wandb_project "$WANDB_PROJECT"
)

if [[ -n "${RANDOM_GATE}" ]]; then args+=( --random_gate "$RANDOM_GATE" ); fi
if [[ -n "${LR_PROJECTOR}" ]]; then args+=( --lr_projector "$LR_PROJECTOR" ); fi
if [[ -n "${LR_GENERATOR}" ]]; then args+=( --lr_generator "$LR_GENERATOR" ); fi
if [[ -n "${LR_COMPRESSOR}" ]]; then args+=( --lr_compressor "$LR_COMPRESSOR" ); fi
if [[ -n "${TEACHER_TARGETS_PATH}" ]]; then args+=( --teacher_targets_path "$TEACHER_TARGETS_PATH" ); fi
if [[ -n "${DISTILL_LAYERS}" ]]; then args+=( --distill_layers "$DISTILL_LAYERS" ); fi
if [[ -n "${DISTILL_COEFF}" ]]; then args+=( --distill_coeff "$DISTILL_COEFF" ); fi
if [[ -n "${WANDB_RUN_NAME}" ]]; then args+=( --wandb_run_name "$WANDB_RUN_NAME" ); fi
if [[ -n "${WANDB_RUN_TAGS}" ]]; then args+=( --wandb_run_tags "$WANDB_RUN_TAGS" ); fi
if [[ -n "${JAVIS_WARMUP_SAMPLES}" ]]; then args+=( --javis_query_warmup_samples "$JAVIS_WARMUP_SAMPLES" ); fi
if [[ -n "${JAVIS_WARMUP_SAVE_PATH}" ]]; then args+=( --javis_query_warmup_save_path "$JAVIS_WARMUP_SAVE_PATH" ); fi
if [[ -n "${JAVIS_QUERY_NUM}" ]]; then args+=( --javis_num_queries "$JAVIS_QUERY_NUM" ); fi
if [[ -n "${JAVIS_QUERY_GROUP_SIZE}" ]]; then args+=( --javis_query_group_size "$JAVIS_QUERY_GROUP_SIZE" ); fi
if [[ -n "${JAVIS_Q_COS_COEFF}" ]]; then args+=( --javis_q_cos_coeff "$JAVIS_Q_COS_COEFF" ); fi
if [[ -n "${TRUNCATE_LEN}" ]]; then args+=( --truncate_len "$TRUNCATE_LEN" ); fi


if [[ -n "${RESUME_PATH}" ]]; then
  args+=( --resume_path "$RESUME_PATH" )
fi

if [[ "${GRAD_PROBE}" == "true" ]]; then
  args+=( --grad_probe true )
fi

if [[ "${TRAIN_ONLY_SPECIAL}" == "true" ]]; then
  args+=( --train_only_special_token_embeddings true )
fi

if [[ -n "${EVAL_DATA_PATH}" ]]; then
  args+=( --eval_data_path "$EVAL_DATA_PATH" )
  args+=( --eval_steps "$EVAL_STEPS" )
  args+=( --eval_max_batches "$EVAL_MAX_BATCHES" )
fi

if [[ "${PARALLEL}" == "ddp" || "${PARALLEL}" == "fsdp" ]]; then
  torchrun --nproc_per_node "${NPROC_PER_NODE}" -m src.train "${args[@]}"
else
  python -m src.train "${args[@]}"
fi
