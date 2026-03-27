#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR%/}/../.."
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

CKPT_DIR="${CKPT_DIR:-/default-vepfs/public/user/ga/Iron/checkpoints/phase-full-dynamicq/phase-full_step_480}"
DATA_PATH="${DATA_PATH:-./data/eval_adaptive.jsonl}"
# DATA_PATH="${DATA_PATH:-./data/eval_fixed.jsonl}"

BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_BATCHES="${MAX_BATCHES:-0}"    # 0 表示不限制
TRUNCATE_LEN="${TRUNCATE_LEN:-8192}"

# TRUNCATE_LEN="${TRUNCATE_LEN:-0}"

# CHUNKING="${CHUNKING:-fixed}"   # fixed | adaptive
CHUNKING="${CHUNKING:-adaptive}"   # fixed | adaptive
CHUNK_SIZE="${CHUNK_SIZE:-16}"     # 仅 fixed 有效
DEVICE="${DEVICE:-auto}"           # auto | cuda | cpu
PARALLEL="${PARALLEL:-fsdp}"       # none | ddp | fsdp

if [[ "$PARALLEL" == "ddp" || "$PARALLEL" == "fsdp" ]]; then
  if [[ -z "${NPROC_PER_NODE:-}" ]]; then
    NPROC_PER_NODE="$(python -c 'import torch; print(torch.cuda.device_count())')"
  fi
  torchrun --nproc_per_node="$NPROC_PER_NODE" scripts/eval_ppl.py \
    --parallel "$PARALLEL" \
    --ckpt_dir "$CKPT_DIR" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --max_batches "$MAX_BATCHES" \
    --truncate_len "$TRUNCATE_LEN" \
    --chunking "$CHUNKING" \
    --chunk_size "$CHUNK_SIZE" \
    --device "$DEVICE"
else
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  python scripts/eval_ppl.py \
  --ckpt_dir "$CKPT_DIR" \
  --data_path "$DATA_PATH" \
  --batch_size "$BATCH_SIZE" \
  --max_batches "$MAX_BATCHES" \
  --truncate_len "$TRUNCATE_LEN" \
  --chunking "$CHUNKING" \
  --chunk_size "$CHUNK_SIZE" \
  --device "$DEVICE"
fi
