#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DATA_PATH=${DATA_PATH:-../data/train.jsonl}
# DATA_PATH=${DATA_PATH:-../data/train.jsonl}
OUTPUT_PATH=${OUTPUT_PATH:-../data/distill/distill_hidden_teacher_50k.pt}
MODEL_NAME=${MODEL_NAME:-/default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B}
TARGET_LAYER=${TARGET_LAYER:-15}
CHUNK_SIZE=${CHUNK_SIZE:-16}
Q_NUM=${Q_NUM:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-10240}
BATCH_SIZE=${BATCH_SIZE:-1}
DEVICE=${DEVICE:-cuda}
GPU_LIST=${GPU_LIST:-0,1,2,3,4,5,6,7}


args=(
  --model_name "$MODEL_NAME"
  --target_layer "$TARGET_LAYER"
  --chunk_size "$CHUNK_SIZE"
  --q_num "$Q_NUM"
  --max_seq_len "$MAX_SEQ_LEN"
  --batch_size "$BATCH_SIZE"
)
if [[ -z "${GPU_LIST}" ]]; then
  python src/distill_hidden.py \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --device "$DEVICE" \
    "${args[@]}"
  exit 0
fi

IFS=',' read -r -a gpu_arr <<< "$GPU_LIST"
num_shards=${#gpu_arr[@]}
if [[ "$num_shards" -le 0 ]]; then
  echo "GPU_LIST is empty"
  exit 1
fi

SHARD_DIR="${OUTPUT_PATH}.shards"
mkdir -p "$SHARD_DIR"

DATA_PATH="$DATA_PATH" SHARD_DIR="$SHARD_DIR" NUM_SHARDS="$num_shards" python - <<'PY'
import math
import os
from pathlib import Path

data_path = os.environ["DATA_PATH"]
shard_dir = Path(os.environ["SHARD_DIR"])
num_shards = int(os.environ["NUM_SHARDS"])

total = 0
with open(data_path, "r", encoding="utf-8") as f:
    for _ in f:
        total += 1

if total == 0:
    raise SystemExit("Empty dataset")

lines_per_shard = math.ceil(total / num_shards)
out_files = []
for i in range(num_shards):
    out_path = shard_dir / f"shard_{i}.jsonl"
    out_files.append(open(out_path, "w", encoding="utf-8"))

current = 0
shard_idx = 0
with open(data_path, "r", encoding="utf-8") as f:
    for line in f:
        if current >= lines_per_shard and shard_idx < num_shards - 1:
            shard_idx += 1
            current = 0
        out_files[shard_idx].write(line)
        current += 1

for fh in out_files:
    fh.close()
PY

pids=()
shard_outputs=()
for i in "${!gpu_arr[@]}"; do
  shard_path="${SHARD_DIR}/shard_${i}.jsonl"
  shard_output="${OUTPUT_PATH}.shard${i}.pt"
  shard_outputs+=( "$shard_output" )
  python src/distill_hidden.py \
    --data_path "$shard_path" \
    --output_path "$shard_output" \
    --device "cuda:${gpu_arr[$i]}" \
    "${args[@]}" &
  pids+=( "$!" )
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

python scripts/merge_precompute_teacher.py \
  --output_path "$OUTPUT_PATH" \
  --shard_paths "${shard_outputs[@]}"
