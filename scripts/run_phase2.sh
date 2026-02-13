#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

MODEL_NAME=${MODEL_NAME:-/default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B}
PHASE=${PHASE:-phase2}
DATA_PATH=${DATA_PATH:-../data/phase2_train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/phase2_debug}

RESUME_PATH=${RESUME_PATH:-checkpoints/phase1/phase1_step_20}
LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-false}
RESET_STEP_ON_RESUME=${RESET_STEP_ON_RESUME:-true}

# RESUME_PATH=${RESUME_PATH:-checkpoints/phase2/phase2_step_30}
# LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-true}
# RESET_STEP_ON_RESUME=${RESET_STEP_ON_RESUME:-false}


CHUNK_SIZE=${CHUNK_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-1}
STEPS=${STEPS:-300}
SAVE_STEPS=${SAVE_STEPS:-30}
LOG_STEPS=${LOG_STEPS:-2}

GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-8}
TRAIN_ONLY_SPECIAL=${TRAIN_ONLY_SPECIAL:-true}
WARMUP_STEPS=${WARMUP_STEPS:-30}
# LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-false}
# RESET_STEP_ON_RESUME=${RESET_STEP_ON_RESUME:-true}

export NCCL_ASYNC_ERROR_HANDLING=1

EVAL_DATA_PATH=${EVAL_DATA_PATH:-../data/phase1_eval.jsonl}
EVAL_STEPS=${EVAL_STEPS:-30}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-13}

LR=${LR:-5e-5}
LR_PROJECTOR=${LR_PROJECTOR:-5e-5} 
LR_GENERATOR=${LR_GENERATOR:-3e-5}
LR_COMPRESSOR=${LR_COMPRESSOR:-5e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}

PARALLEL=${PARALLEL:-fsdp} # none|ddp|fsdp
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

if [[ "${LOAD_OPTIMIZER}" != "true" ]]; then
  args+=( --load_weights_only true )
fi

if [[ "${RESET_STEP_ON_RESUME}" == "true" ]]; then
  args+=( --reset_step_on_resume true )
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
