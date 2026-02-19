#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

MODEL_NAME=${MODEL_NAME:-/default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B}
PHASE=${PHASE:-phase-full}
DATA_PATH=${DATA_PATH:-/default-vepfs/public/user/ga/Iron/33/data/train.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-../checkpoints/phase-full-distill}

RESUME_PATH=${RESUME_PATH:-../checkpoints/phase-cmp-residual/phase-cmp_step_90}
# RESUME_PATH=${RESUME_PATH:-../checkpoints/phase1-2query/phase1_step_40}
LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-false}
RESET_STEP_ON_RESUME=${RESET_STEP_ON_RESUME:-true}

## to load phase2 checkpoint, you need to set RESUME_PATH, LOAD_OPTIMIZER, RESET_STEP_ON_RESUME
# RESUME_PATH=${RESUME_PATH:-checkpoints/phase2/phase2_step_30}
# LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-true}
# RESET_STEP_ON_RESUME=${RESET_STEP_ON_RESUME:-false}


CHUNK_SIZE=${CHUNK_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-1}
STEPS=${STEPS:-500}
SAVE_STEPS=${SAVE_STEPS:-20}
LOG_STEPS=${LOG_STEPS:-1}

GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16}
TRAIN_ONLY_SPECIAL=${TRAIN_ONLY_SPECIAL:-true}
WARMUP_STEPS=${WARMUP_STEPS:-5}
GRAD_PROBE=${GRAD_PROBE:-true}
# LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-false}
# RESET_STEP_ON_RESUME=${RESET_STEP_ON_RESUME:-true}

GLOBAL_EPOCH=${GLOBAL_EPOCH:-3} 
export NCCL_ASYNC_ERROR_HANDLING=1

EVAL_DATA_PATH=${EVAL_DATA_PATH:-/default-vepfs/public/user/ga/Iron/33/data/eval.jsonl}
EVAL_STEPS=${EVAL_STEPS:-20}
EVAL_MAX_BATCHES=${EVAL_MAX_BATCHES:-13}

LR=${LR:-5e-5}
LR_PROJECTOR=${LR_PROJECTOR:-1e-5} 
LR_GENERATOR=${LR_GENERATOR:-5e-6}
LR_COMPRESSOR=${LR_COMPRESSOR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}

PARALLEL=${PARALLEL:-fsdp} # none|ddp|fsdp
DDP_FIND_UNUSED=${DDP_FIND_UNUSED:-false}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}


WANDB_PROJECT=${WANDB_PROJECT:-Mark-33}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-Mark-33-phase-full-distill}
WANDB_RUN_TAGS=${WANDB_RUN_TAGS:-phase3-full,2query,javis,attn,residual,distill}

JAVIS_QUERY_NUM=${JAVIS_QUERY_NUM:-2}
TEACHER_TARGETS_PATH=${TEACHER_TARGETS_PATH:-/default-vepfs/public/user/ga/Iron/33/data/precompute_teacher.pt}
DISTILL_LAYERS=${DISTILL_LAYERS:-24,26,28,30,31}
DISTILL_COEFF=${DISTILL_COEFF:-1}


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

if [[ -n "${WANDB_RUN_NAME}" ]]; then args+=( --wandb_run_name "$WANDB_RUN_NAME" ); fi
if [[ -n "${WANDB_RUN_TAGS}" ]]; then args+=( --wandb_run_tags "$WANDB_RUN_TAGS" ); fi
if [[ -n "${JAVIS_QUERY_NUM}" ]]; then args+=( --javis_num_queries "$JAVIS_QUERY_NUM" ); fi

if [[ -n "${LR_PROJECTOR}" ]]; then args+=( --lr_projector "$LR_PROJECTOR" ); fi
if [[ -n "${LR_GENERATOR}" ]]; then args+=( --lr_generator "$LR_GENERATOR" ); fi
if [[ -n "${LR_COMPRESSOR}" ]]; then args+=( --lr_compressor "$LR_COMPRESSOR" ); fi
if [[ -n "${TEACHER_TARGETS_PATH}" ]]; then args+=( --teacher_targets_path "$TEACHER_TARGETS_PATH" ); fi
if [[ -n "${DISTILL_LAYERS}" ]]; then args+=( --distill_layers "$DISTILL_LAYERS" ); fi
if [[ -n "${DISTILL_COEFF}" ]]; then args+=( --distill_coeff "$DISTILL_COEFF" ); fi

if [[ -n "${GLOBAL_EPOCH}" ]]; then args+=( --global_epoch "$GLOBAL_EPOCH" ); fi

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

if [[ "${GRAD_PROBE}" == "true" ]]; then
  args+=( --grad_probe true )
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
