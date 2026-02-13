from __future__ import annotations

from datetime import timedelta
import os
import torch
import torch.nn as nn
import wandb  # [Added] 引入 WandB
from dataclasses import dataclass

from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from src.data_processor import IronCellCollator
from src.modeling_iron_cell import IronCellModel
from src.train_utils import (
    JsonlDataset,
    _build_fsdp_auto_wrap_policy,
    _is_no_weight_decay_param,
    configure_special_embedding_mode,
    load_checkpoint,
    load_model,
    load_tokenizer,
    save_checkpoint,
    save_checkpoint_fsdp,
)


@dataclass(frozen=True)
class TrainArgs:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    phase: str = "phase1"
    # [Config] 数据路径
    data_path: str = "data/phase1_train.jsonl"
    # [Config] 输出路径
    output_dir: str = "checkpoints"
    
    resume_path: str | None = None
    load_weights_only: bool = False
    chunk_size: int = 16
    batch_size: int = 2  # A800 上可以尝试大一点，比如 4 或 8
    lr: float = 5e-5
    lr_projector: float | None = None
    lr_generator: float | None = None
    lr_compressor: float | None = None
    weight_decay: float = 0.0
    parallel: str = "none"  # none|ddp|fsdp
    ddp_find_unused_parameters: bool = False
    fsdp_wrap: str = "full"  # full|generator_only
    fsdp_use_orig_params: bool = True
    fsdp_cpu_offload: bool = False
    train_only_special_token_embeddings: bool = False
    grad_accum_steps: int = 1
    warmup_steps: int = 0
    reset_step_on_resume: bool = False
    eval_data_path: str | None = None
    eval_steps: int = 0
    eval_max_batches: int = 0
    steps: int = 2000    # Phase 1 跑 2000 步即可
    save_steps: int = 500 # 每 500 步保存一次
    log_steps: int = 10

class TrainStepModule(nn.Module):
    def __init__(self, iron: IronCellModel, *, phase: str, l2_coeff: float = 1e-4) -> None:
        super().__init__()
        self.iron = iron
        self.phase = str(phase)
        self.l2_coeff = float(l2_coeff)

    def forward(self, batch):  # type: ignore[override]
        device = self.iron.device

        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        attn_2d = batch.attention_mask_2d.to(device)
        position_ids = batch.position_ids.to(device)
        labels = batch.labels.to(device)

        memory_vectors = self.iron.compute_compressed_vectors(
            chunk_input_ids=chunk_ids,
            chunk_attention_mask=chunk_mask,
        )

        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors,
            memory_positions=mem_pos,
        )

        out = self.iron(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_2d,
            position_ids=position_ids,
            labels=labels,
        )

        l2_loss = torch.zeros((), device=device, dtype=out.loss.dtype)
        loss = out.loss
        if self.phase == "phase2":
            l2_loss = memory_vectors.norm(p=2, dim=-1).mean()
            loss = loss + self.l2_coeff * l2_loss

        return loss, l2_loss


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
    parallel = str(args.parallel).lower()
    use_ddp = parallel == "ddp"
    use_fsdp = parallel == "fsdp"
    use_dist = use_ddp or use_fsdp
    if use_dist:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
        device = torch.device("cuda", local_rank)
        is_rank0 = int(os.environ.get("RANK", "0")) == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_rank0 = True

    if is_rank0:
        wandb.init(project="soulbone", name=f"run-{args.phase}-", config=args)

    tokenizer, is_resume = load_tokenizer(args)
    model = load_model(args, tokenizer, device, is_resume=is_resume)

    # 3. 优化设置
    model.generator.config.use_cache = False
    model.generator.gradient_checkpointing_enable()
    # Phase 2 时 Compressor 也需要开启 gradient checkpointing
    if args.phase == "phase2":
        model.compressor.gradient_checkpointing_enable()

    set_phase(model, args.phase)
    configure_special_embedding_mode(args, model, tokenizer, is_resume=is_resume)

    # 4. 数据加载 (替换 toy text)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}. Please run prepare_data.py first.")
        
    dataset = JsonlDataset(args.data_path)
    collator = IronCellCollator(tokenizer, chunk_size=args.chunk_size)
    if use_dist:
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=0, collate_fn=collator)
    else:
        sampler = None
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    eval_loader = None
    eval_sampler = None
    if args.eval_data_path is not None and str(args.eval_data_path).strip() != "":
        if not os.path.exists(args.eval_data_path):
            raise FileNotFoundError(f"Eval file not found at {args.eval_data_path}")
        eval_dataset = JsonlDataset(args.eval_data_path)
        if use_dist:
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=eval_sampler,
                num_workers=0,
                collate_fn=collator,
            )
        else:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collator,
            )

    print("Starting training...")
    model.train()
    total_micro_loss = 0.0
    micro_count = 0
    
    step_module = TrainStepModule(model, phase=args.phase)
    if use_fsdp:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        cpu_offload = CPUOffload(offload_params=True) if bool(args.fsdp_cpu_offload) else None
        auto_wrap_policy = _build_fsdp_auto_wrap_policy(model, args.fsdp_wrap)
        step_module = FSDP(
            step_module,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp,
            cpu_offload=cpu_offload,
            use_orig_params=bool(args.fsdp_use_orig_params),
            device_id=device,
            sync_module_states=True,
        )
    elif use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        step_module = DDP(
            step_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    iron = step_module.module.iron if use_dist else step_module.iron

    lr_projector = float(args.lr_projector) if args.lr_projector is not None else float(args.lr)
    lr_generator = float(args.lr_generator) if args.lr_generator is not None else float(args.lr)
    lr_compressor = float(args.lr_compressor) if args.lr_compressor is not None else float(args.lr)

    seen: set[int] = set()

    def _append_param_groups(named_params, *, prefix: str, lr: float) -> list[dict]:
        decay = []
        no_decay = []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            full_name = f"{prefix}.{name}" if prefix else str(name)
            if _is_no_weight_decay_param(full_name):
                no_decay.append(p)
            else:
                decay.append(p)

        out = []
        wd = float(args.weight_decay)
        if decay:
            out.append({"params": decay, "lr": float(lr), "weight_decay": wd})
        if no_decay:
            out.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0})
        return out

    param_groups = []
    param_groups.extend(_append_param_groups(iron.projector.named_parameters(), prefix="projector", lr=lr_projector))
    param_groups.extend(
        _append_param_groups(iron.special_token_embeddings.named_parameters(), prefix="special_token_embeddings", lr=lr_projector)
    )
    param_groups.extend(_append_param_groups(iron.generator.named_parameters(), prefix="generator", lr=lr_generator))
    param_groups.extend(_append_param_groups(iron.compressor.named_parameters(), prefix="compressor", lr=lr_compressor))

    if not param_groups:
        raise ValueError("No trainable parameters found (all requires_grad=False).")

    optimizer = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=0.0)

    if args.resume_path:
        loaded_step = load_checkpoint(optimizer, args)
    else:
        loaded_step = 0

    if bool(args.reset_step_on_resume):
        step = 0
    else:
        step = int(loaded_step)

    base_lrs = [float(g.get("lr", float(args.lr))) for g in optimizer.param_groups]

    # 6. 训练循环
    # 使用 iter(loader) 配合 while 循环可以防止 epoch 结束导致的重置，或者直接用 cycle
    epoch = 0
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)
    
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    eval_steps = max(0, int(args.eval_steps))
    eval_max_batches = max(0, int(args.eval_max_batches))
    if eval_sampler is not None:
        eval_sampler.set_epoch(0)

    def _maybe_run_eval(*, step: int) -> float | None:
        if eval_loader is None:
            return None
        if eval_steps <= 0 or step % eval_steps != 0:
            return None

        step_module.eval()
        sum_loss = torch.zeros((), device=device, dtype=torch.float32)
        count = torch.zeros((), device=device, dtype=torch.float32)
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                if eval_max_batches > 0 and i >= eval_max_batches:
                    break
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, _ = step_module(batch)
                sum_loss += loss.detach().float()
                count += 1.0

        if use_dist:
            torch.distributed.all_reduce(sum_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

        step_module.train()

        denom = float(count.clamp(min=1.0).item())
        return float((sum_loss / denom).item())
    while step < args.steps:
        warmup_steps = max(0, int(args.warmup_steps))
        if warmup_steps > 0 and step < warmup_steps:
            warmup_factor = float(step + 1) / float(warmup_steps)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = float(base_lr) * warmup_factor
        else:
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = float(base_lr)

        optimizer.zero_grad(set_to_none=True)
        last_l2_loss = torch.zeros((), device=device)

        for micro in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)  # 新的 Epoch
                batch = next(data_iter)

            if use_dist and micro < grad_accum_steps - 1:
                sync_ctx = step_module.no_sync()
            else:
                sync_ctx = torch.enable_grad()

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, l2_loss = step_module(batch)
                    last_l2_loss = l2_loss
                    (loss / grad_accum_steps).backward()

            total_micro_loss += float(loss.item())
            micro_count += 1

        if use_fsdp:
            grad_norm = FSDP.clip_grad_norm_(step_module, 1.0)
        else:
            grad_norm = clip_grad_norm_(step_module.parameters(), 1.0)

        optimizer.step()

        step += 1

        eval_loss = _maybe_run_eval(step=step)
        if eval_loss is not None and is_rank0:
            print(f"[{args.phase}] Step {step} | EvalLoss: {eval_loss:.4f}")
            wandb.log({"eval/loss": eval_loss}, step=step)

        # [Logging]
        if step % args.log_steps == 0:
            denom = max(1, micro_count)
            loss_micro_avg = total_micro_loss / denom
            print(
                f"[{args.phase}] Step {step} | Loss: {loss_micro_avg:.4f} | GradNorm: {grad_norm:.4f}"
            )
            
            if is_rank0:
                lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
                wandb.log({
                    "loss": loss_micro_avg,
                    "l2_reg": last_l2_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": float(args.lr),
                    "lr_group0": lrs[0] if len(lrs) > 0 else 0.0,
                    "lr_group1": lrs[1] if len(lrs) > 1 else 0.0,
                    "lr_group2": lrs[2] if len(lrs) > 2 else 0.0,
                }, step=step)
            total_micro_loss = 0.0
            micro_count = 0

        # [Checkpointing]
        if step % args.save_steps == 0:
            if use_fsdp:
                save_checkpoint_fsdp(step_module, optimizer, tokenizer, args, step, is_rank0=is_rank0)
            else:
                if is_rank0:
                    to_save = step_module.module.iron if use_ddp else step_module.iron
                    save_checkpoint(to_save, optimizer, tokenizer, args, step)

    print("Training finished.")
    if is_rank0:
        wandb.finish()

if __name__ == "__main__":
    main()
