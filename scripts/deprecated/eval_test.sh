#!/usr/bin/env bash
set -euo pipefail

# 1. 确保在项目根目录运行
cd "$(dirname "$0")/.."

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Current PYTHONPATH: $PYTHONPATH"
echo "Current Directory: $(pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_ASYNC_ERROR_HANDLING=1

# ==========================================
# 配置区
# ==========================================
# EVAL_MODE=${EVAL_MODE:-oracle} # oracle | amnesiac | mark42

# EVAL_MODE=${EVAL_MODE:-amnesiac} # oracle | amnesiac | mark42

# MODEL_NAME=${MODEL_NAME:-/default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B}
# RESUME_PATH=${RESUME_PATH:-}


EVAL_MODE=${EVAL_MODE:-mark42} # oracle | amnesiac | mark42
MODEL_NAME=${MODEL_NAME:-}
# RESUME_PATH=${RESUME_PATH:-/default-vepfs/public/user/ga/Iron/checkpoints/phase-full-dynamicq/phase-full_step_480}



BUFFER_SIZE=${BUFFER_SIZE:-0}



EVAL_DATA_PATH=${EVAL_DATA_PATH:-./data/eval_adaptive.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./data/eval_results}

CHUNK_SIZE=16
JAVIS_QUERY_NUM=2
# MAX_TOKENS=8192
MAX_TOKENS=4096

# ==========================================
# 运行
# ==========================================
args=(
  --eval_mode "$EVAL_MODE"
  --model_name "$MODEL_NAME"
  --resume_path "$RESUME_PATH"
  --eval_data_path "$EVAL_DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --chunk_size "$CHUNK_SIZE"
  --javis_num_queries "$JAVIS_QUERY_NUM"
  --max_tokens "$MAX_TOKENS"
  --buffer_size "$BUFFER_SIZE"
)

# python -m eval.eval "${args[@]}"
# python -m eval.pic

python -m eval.eval_with_buffer "${args[@]}"

