#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

MODEL_NAME=${MODEL_NAME:- YOUR_MODEL_PATH }
PHASE=${PHASE:-phase1}
DATA_PATH=${DATA_PATH:-YOUR_DATA_PATH}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/phase1}
RESUME_PATH=${RESUME_PATH:-}

CHUNK_SIZE=${CHUNK_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-2}
STEPS=${STEPS:-200}
SAVE_STEPS=${SAVE_STEPS:-20}
LOG_STEPS=${LOG_STEPS:-1}

GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
TRAIN_ONLY_SPECIAL=${TRAIN_ONLY_SPECIAL:-true}
WARMUP_STEPS=${WARMUP_STEPS:-0}
EVAL_DATA_PATH=${EVAL_DATA_PATH:-}
EVAL_STEPS=${EVAL_STEPS:-0}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-0}

LR=${LR:-5e-5}
LR_PROJECTOR=${LR_PROJECTOR:-}
LR_GENERATOR=${LR_GENERATOR:-}
LR_COMPRESSOR=${LR_COMPRESSOR:-}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

PARALLEL=${PARALLEL:-ddp} # none|ddp|fsdp
DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-false}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

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
)

if [[ -n "${LR_PROJECTOR}" ]]; then args+=( --lr_projector "$LR_PROJECTOR" ); fi
if [[ -n "${LR_GENERATOR}" ]]; then args+=( --lr_generator "$LR_GENERATOR" ); fi
if [[ -n "${LR_COMPRESSOR}" ]]; then args+=( --lr_compressor "$LR_COMPRESSOR" ); fi

if [[ -n "${RESUME_PATH}" ]]; then
  args+=( --resume_path "$RESUME_PATH" )
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
