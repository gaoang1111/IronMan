from __future__ import annotations

import os
import json
import torch
import wandb  # [Added] 引入 WandB
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import HfArgumentParser

# 引入你的项目模块
from src.configuration_iron_cell import IronCellConfig
from src.data_processor import IronCellCollator
from src.modeling_iron_cell import IronCellModel
from src.token_utils import add_iron_cell_special_tokens, resize_and_smart_init_special_tokens


@dataclass(frozen=True)
class TrainArgs:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    phase: str = "phase1"
    # [Config] 数据路径
    data_path: str = "data/phase1_train.jsonl"
    # [Config] 输出路径
    output_dir: str = "checkpoints"
    
    chunk_size: int = 256
    batch_size: int = 2  # A800 上可以尝试大一点，比如 4 或 8
    lr: float = 5e-5
    steps: int = 2000    # Phase 1 跑 2000 步即可
    save_steps: int = 500 # 每 500 步保存一次
    log_steps: int = 10


# [Added] 简单的 Dataset 类，用于读取 JSONL
class JsonlDataset(Dataset):
    def __init__(self, file_path: str):
        self.texts = []
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                self.texts.append(item["text"])
        print(f"Loaded {len(self.texts)} samples.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def set_phase(model: IronCellModel, phase: str) -> None:
    print(f"--> Setting model to {phase} mode...")
    if phase == "phase1":
        model.freeze_for_phase_1()
        # 双重保险：确保 config 状态正确
        model.config.freeze_compressor = True
        # 手动冻结 Compressor
        for p in model.compressor.parameters():
            p.requires_grad = False
    elif phase == "phase2":
        model.config.freeze_compressor = False
        # 全量解冻
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown phase: {phase}")


def save_checkpoint(model, optimizer, tokenizer, args, step):
    """
    保存模型权重 + 优化器状态 + Config
    """
    save_path = Path(args.output_dir) / f"{args.phase}_step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {save_path}...")

    # 1. 保存模型权重 (State Dict)
    # 处理 DDP 情况 (如果用了多卡)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # 2. [关键] 保存优化器状态
    torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    
    print("Checkpoint saved.")

def load_checkpoint(model, optimizer, args):
    if args.resume_path is None:
        print("No resume path specified. Starting from scratch (or base model).")
        return

    resume_path = Path(args.resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {resume_path}")

    print(f"--> Loading checkpoint from {resume_path}...")

    # 1. 加载模型权重 (State Dict)
    # 这一步是把 Phase 1 练好的 Projector (和 Llama) 读进来
    weights_path = resume_path / "pytorch_model.bin"
    if not weights_path.exists():
         # 兼容 safetensors
        weights_path = resume_path / "model.safetensors"
        
    state_dict = torch.load(weights_path, map_location=model.device)
    
    # 处理 DDP (多卡) 的 key 前缀问题
    if hasattr(model, "module"):
        # 如果当前模型是 DDP，但权重里没有 module. 前缀，或者反之，需要处理
        # 这里假设直接 load 兼容性较好的方式
        model.module.load_state_dict(state_dict)
    else:
        # 如果权重里有 module. 前缀但当前是单卡，去掉前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    
    print("Model weights loaded successfully.")

    # 2. 加载优化器状态 (仅在断点续训时需要)
    # 如果是从 Phase 1 转 Phase 2，因为训练参数变了，优化器状态必须丢弃！
    if not args.load_weights_only:
        opt_path = resume_path / "optimizer.pt"
        if opt_path.exists():
            print("Loading optimizer state...")
            optimizer.load_state_dict(torch.load(opt_path, map_location=model.device))
        else:
            print("Warning: Optimizer state not found, starting optimizer from scratch.")
    else:
        print("Skipping optimizer state (load_weights_only=True).")

def main() -> None:
    # args = TrainArgs()
    parser = HfArgumentParser((TrainArgs,))

    if len(os.sys.argv) == 2 and os.sys.argv[1].endswith(".json"):
        # 支持直接传 json 配置文件
        args = parser.parse_json_file(json_file=os.path.abspath(os.sys.argv[1]))[0]
    else:
        # 支持命令行参数
        args = parser.parse_args_into_dataclasses()[0]
    
    print(args)
    # [Added] 初始化 WandB
    wandb.init(project="soulbone", name=f"run-{args.phase}", config=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_iron_cell_special_tokens(tokenizer)

    # 2. Config & Model
    config = IronCellConfig(
        compressor_model_name=args.model_name,
        generator_model_name=args.model_name,
        freeze_compressor=(args.phase == "phase1"),
        projector_init_type="identity",
        trainable_components=["projector", "embed_tokens", "special_tokens"],
    )
    model = IronCellModel(config).to(device)

    resize_and_smart_init_special_tokens(model.generator, tokenizer)
    resize_and_smart_init_special_tokens(model.compressor, tokenizer)

    # 3. 优化设置
    model.generator.config.use_cache = False
    model.generator.gradient_checkpointing_enable()
    # Phase 2 时 Compressor 也需要开启 gradient checkpointing
    if args.phase == "phase2":
        model.compressor.gradient_checkpointing_enable()

    set_phase(model, args.phase)

    # 4. 数据加载 (替换 toy text)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}. Please run prepare_data.py first.")
        
    dataset = JsonlDataset(args.data_path)
    collator = IronCellCollator(tokenizer, chunk_size=args.chunk_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    # 5. 优化器
    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)

    if args.resume_path:
        load_checkpoint(model, optimizer, args)
        
    print("Starting training...")
    model.train()
    step = 0
    total_loss = 0.0
    
    # 6. 训练循环
    # 使用 iter(loader) 配合 while 循环可以防止 epoch 结束导致的重置，或者直接用 cycle
    data_iter = iter(loader)
    
    while step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader) # 新的 Epoch
            batch = next(data_iter)

        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        labels = batch.labels.to(device)
        attn_2d = batch.attention_mask_2d.to(device)
        position_ids = batch.position_ids.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            # A. 压缩
            memory_vectors = model.compute_compressed_vectors(
                chunk_input_ids=chunk_ids,
                chunk_attention_mask=chunk_mask,
            )
            
            # B. 注入
            inputs_embeds = model.build_inputs_embeds(
                zipper_input_ids=zipper_ids,
                memory_vectors=memory_vectors,
                memory_positions=mem_pos,
            )
            
            # C. 生成 & Loss
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_2d,
                position_ids=position_ids,
                labels=labels,
            )

            loss = out.loss
            l2_loss = torch.tensor(0.0)

            # [Phase 2 Only] L2 Regularization
            if args.phase == "phase2":
                l2_loss = memory_vectors.norm(p=2, dim=-1).mean()
                loss = loss + 1e-4 * l2_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        # [Safety] 梯度裁剪 (关键)
        grad_norm = clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()

        step += 1
        total_loss += loss.item()

        # [Logging]
        if step % args.log_steps == 0:
            avg_loss = total_loss / args.log_steps
            print(f"[{args.phase}] Step {step} | Loss: {avg_loss:.4f} | GradNorm: {grad_norm:.4f}")
            
            wandb.log({
                "loss": avg_loss,
                "l2_reg": l2_loss.item(),
                "grad_norm": grad_norm,
                "lr": args.lr
            })
            total_loss = 0.0

        # [Checkpointing]
        if step % args.save_steps == 0:
            save_checkpoint(model, optimizer, tokenizer, args, step)

    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    main()