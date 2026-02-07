#!/bin/bash

# 1. 设置显卡 (只有一块卡就写 0，八块卡写 0,1,2,3,4,5,6,7)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2. 运行训练
# 使用 accelerate launch 可以自动处理多卡/混合精度，比 python train.py 更稳
# 如果没装 accelerate，直接用 python train.py 也可以

python train.py \
    --model_name "meta-llama/Meta-Llama-3-8B" \
    --phase "phase1" \
    --data_path "data/phase1_train.jsonl" \
    --output_dir "checkpoints/phase1" \
    --chunk_size 256 \
    --batch_size 2 \
    --lr 5e-5 \
    --steps 2000 \
    --save_steps 500 \
    --log_steps 10