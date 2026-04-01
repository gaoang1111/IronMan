from __future__ import annotations

from datetime import timedelta
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb  # [Added] 引入 WandB
from dataclasses import dataclass, field

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
from src.models import IronCellModel
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
    warmup_init_javis_query,
)


def _parse_layers(spec: str) -> list[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


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
    buffer_num: int = 1
    truncate_len: int | None = None
    lr: float = 5e-5
    lr_projector: float | None = None
    lr_generator: float | None = None
    lr_compressor: float | None = None
    lr_javis_gate: float | None = None
    clip_grad_norm: float | None = 1.0
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
    global_epoch: int = 0
    random_gate: float = 0.0
    teacher_targets_path: str | None = None
    teacher_hidden_targets_path: str | None = None
    distill_layers: str = "24,26,28,30,31"
    distill_coeff: float = 0.0
    javis_query_warmup_samples: int | None = 100
    javis_query_warmup_save_path: str | None = None
    javis_num_queries: int = 2
    javis_query_group_size: int = 1
    javis_q_cos_coeff: float = 1.0
    javis_target_layers: list[int] = field(default_factory=lambda: [15, 23, 31])
    wandb_project: str = "soulbone" 
    wandb_run_name: str | None = None
    wandb_run_tags: str | None = None
    grad_probe: bool = False


def set_phase(model: IronCellModel, phase: str) -> None:
    print(f"--> Setting model to {phase} mode...")
    # if phase == "phase1":
    #     model.freeze_for_phase_1()
    #     # 双重保险：确保 config 状态正确
    #     model.config.freeze_compressor = True
    #     # 手动冻结 Compressor
    #     for p in model.compressor.parameters():
    #         p.requires_grad = False
    #     for name, p in model.generator.named_parameters():
    #         if "norm" in name: 
    #             p.requires_grad = True
    if phase == "phase1":
        model.freeze_for_phase_1()
        # 双重保险：确保 config 状态正确
        model.config.freeze_compressor = True
        # 手动冻结 Compressor
        for p in model.compressor.parameters():
            p.requires_grad = False
        for name, p in model.generator.named_parameters():
            if "norm" in name: 
                p.requires_grad = True
        if hasattr(model, "javis"):
            model.javis.layer_gates.requires_grad = False
            model.javis.layer_gates.data.fill_(0.07)
    elif phase == "phase2":
        model.config.freeze_compressor = False
        # 全量解冻
        for p in model.parameters():
            p.requires_grad = True

        # hack for cmp
        model.config.freeze_compressor = True
        for p in model.compressor.parameters():
            p.requires_grad = False
    elif phase == "phase3" or phase == "phase-full":
        model.config.freeze_compressor = False
        # 全量解冻
        for p in model.parameters():
            p.requires_grad = True
    elif phase == "phase-cmp":
        model.config.freeze_compressor = False
        model.freeze_for_phase_1()
        for p in model.compressor.parameters():
            p.requires_grad = True
        for name, p in model.generator.named_parameters():
            if "norm" in name: 
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
        run_name = args.wandb_run_name or f"run-{args.phase}-"
        tags = None
        if args.wandb_run_tags is not None and str(args.wandb_run_tags).strip() != "":
            tags = [t.strip() for t in str(args.wandb_run_tags).split(",") if t.strip()]
        wandb.init(project=str(args.wandb_project), name=run_name, tags=tags, config=args)
    
    tokenizer, is_resume = load_tokenizer(args)
    model = load_model(args, tokenizer, device, is_resume=is_resume)

    # 3. 优化设置
    model.generator.config.use_cache = False

    # # 3. 优化设置
    # model.generator.config.use_cache = False
    # model.generator.gradient_checkpointing_enable()
    
    # if args.phase in ["phase2", "phase3", "phase-cmp", "phase-full"]:
    #     print(f"--> Enabling Gradient Checkpointing for Compressor (Phase: {args.phase})...")
    #     model.compressor.gradient_checkpointing_enable()


    model.generator.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.phase in ["phase2", "phase3", "phase-cmp", "phase-full"]:
        print(f"--> Enabling Gradient Checkpointing for Compressor (Phase: {args.phase})...")
        model.compressor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    set_phase(model, args.phase)
    configure_special_embedding_mode(args, model, tokenizer, is_resume=is_resume)

    # 4. 数据加载 (替换 toy text)
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found at {args.data_path}. Please run prepare_data.py first.")
        
    dataset = JsonlDataset(args.data_path)
    teacher_targets = None
    if args.teacher_targets_path is not None and str(args.teacher_targets_path).strip() != "":
        if not os.path.exists(args.teacher_targets_path):
            raise FileNotFoundError(f"Teacher targets not found at {args.teacher_targets_path}")
        teacher_pack = torch.load(args.teacher_targets_path, map_location="cpu")
        if isinstance(teacher_pack, dict) and "targets" in teacher_pack:
            teacher_targets = teacher_pack["targets"]
        else:
            teacher_targets = teacher_pack

    teacher_hidden_targets = None
    teacher_hidden_valid_lens = None
    teacher_hidden_target_layer = None
    if args.teacher_hidden_targets_path is not None and str(args.teacher_hidden_targets_path).strip() != "":
        if not os.path.exists(args.teacher_hidden_targets_path):
            raise FileNotFoundError(f"Teacher hidden targets not found at {args.teacher_hidden_targets_path}")
        print(f"--> [System] Memory-Mapping massive hidden targets from {args.teacher_hidden_targets_path}...")
        hidden_pack = torch.load(
            args.teacher_hidden_targets_path, 
            map_location="cpu", 
            mmap=True  
        )

        print(f"--> [Done]  hidden targets from {args.teacher_hidden_targets_path}...")
        
        if not (isinstance(hidden_pack, dict) and "targets" in hidden_pack and "valid_v_lens" in hidden_pack):
            raise ValueError("teacher_hidden_targets_path must point to a dict with keys: targets, valid_v_lens, target_layer, chunk_size, q_num.")
        teacher_hidden_targets = hidden_pack["targets"]
        teacher_hidden_valid_lens = hidden_pack["valid_v_lens"]
        teacher_hidden_target_layer = int(hidden_pack.get("target_layer", -1))
        if teacher_hidden_target_layer < 0:
            raise ValueError("Hidden distill pack missing target_layer.")
        pack_chunk_size = int(hidden_pack.get("chunk_size", -1))
        if pack_chunk_size > 0 and int(pack_chunk_size) != int(args.chunk_size):
            raise ValueError(f"Hidden distill pack chunk_size={pack_chunk_size} mismatches args.chunk_size={args.chunk_size}.")
        pack_q_num = int(hidden_pack.get("q_num", -1))
        if pack_q_num > 0 and int(pack_q_num) != int(args.javis_num_queries):
            raise ValueError(f"Hidden distill pack q_num={pack_q_num} mismatches args.javis_num_queries={args.javis_num_queries}.")
    print(f"collator numv {args.javis_num_queries=}")
    collator = IronCellCollator(
        tokenizer,
        chunk_size=args.chunk_size,
        num_v=args.javis_num_queries,
        truncate_len=args.truncate_len,
        random_gate=args.random_gate,
        teacher_targets=teacher_targets,
        teacher_hidden_targets=teacher_hidden_targets,
        teacher_hidden_valid_lens=teacher_hidden_valid_lens,
        teacher_hidden_target_layer=teacher_hidden_target_layer,
    )

    collator_eval = IronCellCollator(
        tokenizer,
        chunk_size=args.chunk_size,
        num_v=args.javis_num_queries,
        random_gate=0,
        truncate_len=args.truncate_len,
        teacher_targets=teacher_targets,
        teacher_hidden_targets=teacher_hidden_targets,
        teacher_hidden_valid_lens=teacher_hidden_valid_lens,
        teacher_hidden_target_layer=teacher_hidden_target_layer,
    )
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
                collate_fn=collator_eval,
            )
        else:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collator_eval,
            )

    if 0:
    # if args.phase == "phase1" and not is_resume:
        warmup_init_javis_query(
            model,
            loader,
            num_samples=getattr(model.config, "javis_query_warmup_samples", None),
            save_path=getattr(model.config, "javis_query_warmup_save_path", None),
            use_dist=use_dist,
        )

    print("Starting training...")
    model.train()
    total_micro_loss = 0.0
    total_micro_q_cos = 0.0
    total_micro_distill = 0.0
    total_micro_student_attn = 0.0
    total_micro_teacher_attn = 0.0

    micro_count = 0
    
    from src.attention import TrainStepModule
    step_module = TrainStepModule(
        model,
        phase=args.phase,
        javis_q_cos_coeff=args.javis_q_cos_coeff,
        chunk_size=args.chunk_size,
    )
    step_module.grad_probe = bool(args.grad_probe)
    step_module.to(torch.bfloat16)
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

    step_impl = step_module.module if use_dist else step_module
    iron = step_impl.iron

    lr_projector = float(args.lr_projector) if args.lr_projector is not None else float(args.lr)
    lr_generator = float(args.lr_generator) if args.lr_generator is not None else float(args.lr)
    lr_compressor = float(args.lr_compressor) if args.lr_compressor is not None else float(args.lr)
    lr_javis_gate = float(args.lr_javis_gate) if args.lr_javis_gate is not None else float(args.lr)
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
    # param_groups.extend(_append_param_groups(iron.javis.named_parameters(), prefix="javis", lr=lr_projector))
    
    javis_gate_params = []
    javis_base_params = []
    for n, p in iron.javis.named_parameters():
        if "layer_gates" in n:
            javis_gate_params.append((n, p))
        else:
            javis_base_params.append((n, p))

    # 常规矩阵参数保持 1e-5 (lr_projector)
    param_groups.extend(_append_param_groups(javis_base_params, prefix="javis", lr=lr_projector))
    # Gate 标量参数单独给 1e-2 (放大 1000 倍)
    param_groups.extend(_append_param_groups(javis_gate_params, prefix="javis_gates", lr=lr_javis_gate))


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
    epoch = args.global_epoch
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
                    loss, _, _ = step_module(batch, return_metrics=False)
                sum_loss += loss.detach().float()
                count += 1.0

        if use_dist:
            torch.distributed.all_reduce(sum_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

        step_module.train()

        denom = float(count.clamp(min=1.0).item())
        return float((sum_loss / denom).item())

    if is_rank0:
        print(f"[DEBUG] LM Head requires_grad: {step_module.iron.generator.lm_head.weight.requires_grad}")
        print(f"[DEBUG] Layer 10 requires_grad: {step_module.iron.generator.model.layers[10].self_attn.q_proj.weight.requires_grad}")
        print(f"[DEBUG] Javis Gate requires_grad: {step_module.iron.javis.layer_gates.requires_grad}")
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
        if step_impl.grad_probe:
            step_impl.reset_grad_probe()

        javis_metrics = None
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
                    want_metrics = step % args.log_steps == 0 and micro == grad_accum_steps - 1
                    if want_metrics:
                        loss, l2_loss,  javis_metrics, current_out_cos = step_module(batch, return_metrics=True)
                    else:
                        loss, l2_loss,  current_out_cos = step_module(batch, return_metrics=False)
                    last_l2_loss = l2_loss
                    (loss / grad_accum_steps).backward()

                    # step_impl.manual_backward_sync(grad_accum_steps)

            total_micro_loss += float(loss.item())
            total_micro_q_cos += float(current_out_cos.item())
            # if want_metrics:
            #     total_micro_student_attn += float(student_attn_mean.item())
            #     total_micro_teacher_attn += float(teacher_attn_mean.item())
            micro_count += 1

        if step % args.log_steps == 0:
            gen_grad_norm = torch.zeros((), device=device)
            for name, p in iron.generator.named_parameters():
                if "layers.0.self_attn.q_proj" in name and p.grad is not None:
                    gen_grad_norm = (p.grad.float() ** 2).sum()
                    break

            javis_grad_norm = torch.zeros((), device=device)
            if iron.javis.q_base.grad is not None:
                javis_grad_norm = (iron.javis.q_base.grad.float() ** 2).sum()

            cmp_grad_norm = torch.zeros((), device=device)
            for name, p in iron.compressor.named_parameters():
                if "layers.31.self_attn.o_proj" in name and p.grad is not None:
                    cmp_grad_norm = (p.grad.float() ** 2).sum()
                    break

            if use_dist and torch.distributed.is_initialized():
                torch.distributed.all_reduce(gen_grad_norm, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(javis_grad_norm, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cmp_grad_norm, op=torch.distributed.ReduceOp.SUM)

            # if is_rank0:
            #     gen_grad = gen_grad_norm.sqrt().item()
            #     javis_grad = javis_grad_norm.sqrt().item()
            #     cmp_grad = cmp_grad_norm.sqrt().item()
            #     print(
            #         f"  --> [Gradient X-Ray] GEN_L0_q: {gen_grad:.4f} | JAVIS_q: {javis_grad:.4f} | CMP_L31_o: {cmp_grad:.4f}"
            #     )
            if step_impl.grad_probe:
                names = ["memory_vectors", "inputs_embeds", "logits"]
                probe_vals = [step_impl.grad_probe_sums.get(n, torch.zeros((), device=device)) for n in names]
                if use_dist and torch.distributed.is_initialized():
                    for i in range(len(probe_vals)):
                        torch.distributed.all_reduce(probe_vals[i], op=torch.distributed.ReduceOp.SUM)
                if is_rank0:
                    out_vals = [v.sqrt().item() for v in probe_vals]
                    msg = " | ".join(f"{n}: {v:.4f}" for n, v in zip(names, out_vals))
                    print(f"  --> [Grad Probe] {msg}")

        if use_fsdp:
            grad_norm = FSDP.clip_grad_norm_(step_module, args.clip_grad_norm)
        else:
            grad_norm = clip_grad_norm_(step_module.parameters(), args.clip_grad_norm)

        optimizer.step()

        

        eval_loss = _maybe_run_eval(step=step)
        if eval_loss is not None and is_rank0:
            print(f"[{args.phase}] Step {step} | EvalLoss: {eval_loss:.4f}")
            wandb.log({"eval/loss": eval_loss}, step=step)

        # [Logging]
        if step % args.log_steps == 0:
            denom = max(1, micro_count)
            loss_micro_avg = total_micro_loss / denom
            javis_q_cos_micro_avg = total_micro_q_cos / denom
            distill_micro_avg = total_micro_distill / denom
            # student_attn_avg = total_micro_student_attn / denom
            # teacher_attn_avg = total_micro_teacher_attn / denom
            print(
                f"[{args.phase}] Step {step} | Loss: {loss_micro_avg:.4f} | Distill: {distill_micro_avg:.4f} | JavisQCos: {javis_q_cos_micro_avg:.4f} | GradNorm: {grad_norm:.4f}"
            )
            if javis_metrics is not None and is_rank0:
                metrics_msg = " | ".join(f"{k}: {v:.4f}" for k, v in javis_metrics.items())
                if metrics_msg:
                    print(f"  --> [Javis Metrics] {metrics_msg}")
            
            if is_rank0:
                lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
                log_dict = {
                    "loss": loss_micro_avg,
                    "distill_loss": distill_micro_avg,
                    # "attn_student_mean": student_attn_avg,
                    # "attn_teacher_mean": teacher_attn_avg,
                    "javis_q_cos": javis_q_cos_micro_avg,
                    "l2_reg": last_l2_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": float(args.lr),
                    "lr_group0": lrs[0] if len(lrs) > 0 else 0.0,
                    "lr_group1": lrs[1] if len(lrs) > 1 else 0.0,
                    "lr_group2": lrs[2] if len(lrs) > 2 else 0.0,
                }
                if javis_metrics is not None:
                    log_dict.update(javis_metrics)
                javis_grad_cos = getattr(iron.javis, "current_q_grad_cos", None)
                if javis_grad_cos is not None:
                    log_dict["javis_q_grad_cos"] = float(javis_grad_cos)
                wandb.log(log_dict, step=step)
            total_micro_loss = 0.0
            total_micro_q_cos = 0.0
            total_micro_distill = 0.0
            total_micro_student_attn = 0.0
            total_micro_teacher_attn = 0.0
            micro_count = 0

        step += 1
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
