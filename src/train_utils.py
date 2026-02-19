from __future__ import annotations

import json
import os
import re
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.configuration_iron_cell import IronCellConfig
from src.modeling_iron_cell import IronCellModel
from src.token_utils import IronCellSpecialTokens, add_iron_cell_special_tokens, resize_and_smart_init_special_tokens


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
        return self.texts[idx], int(idx)


def save_checkpoint(model, optimizer, tokenizer, args, step):
    """
    保存模型权重 + 优化器状态 + Config
    """
    save_path = Path(args.output_dir) / f"{args.phase}_step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {save_path}...")

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    _save_trainer_state(save_path, step=step)

    print("Checkpoint saved.")


def save_checkpoint_fsdp(step_module: FSDP, optimizer, tokenizer, args, step: int, *, is_rank0: bool) -> None:
    save_path = Path(args.output_dir) / f"{args.phase}_step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {save_path}...")

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(step_module, StateDictType.FULL_STATE_DICT, cfg):
        full_state = step_module.state_dict()
    if is_rank0:
        print(f"Rank 0 is writing model to {save_path}...")
        iron_state = _extract_iron_state_dict(full_state)
        step_module.module.iron.save_pretrained(save_path, state_dict=iron_state)
        tokenizer.save_pretrained(save_path)
        _save_trainer_state(save_path, step=step)
        print("Model and config saved by Rank 0.")

    rank = int(os.environ.get("RANK", "0"))
    torch.save(optimizer.state_dict(), save_path / f"optimizer_rank{rank}.pt")

    print("Checkpoint saved.")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    if is_rank0:
        print(f"Step {step} checkpoint sync complete.")


def _save_trainer_state(save_path: Path, *, step: int) -> None:
    state = {"step": int(step)}
    (save_path / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_trainer_state(resume_path: Path) -> dict | None:
    p = resume_path / "trainer_state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_step_from_dirname(resume_path: Path) -> int | None:
    m = re.search(r"_step_(\d+)$", resume_path.name)
    if m is None:
        return None
    return int(m.group(1))


def load_checkpoint(optimizer, args) -> int:
    if args.resume_path is None:
        return 0

    resume_path = Path(args.resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {resume_path}")

    if not args.load_weights_only:
        parallel = str(getattr(args, "parallel", "none")).lower()
        if parallel == "fsdp":
            rank = int(os.environ.get("RANK", "0"))
            opt_path = resume_path / f"optimizer_rank{rank}.pt"
            if opt_path.exists():
                print("Loading optimizer state...")
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            else:
                print("Warning: Optimizer state not found, starting optimizer from scratch.")
        else:
            opt_path = resume_path / "optimizer.pt"
            if opt_path.exists():
                print("Loading optimizer state...")
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
            else:
                print("Warning: Optimizer state not found, starting optimizer from scratch.")
    else:
        print("Skipping optimizer state (load_weights_only=True).")

    state = _load_trainer_state(resume_path)
    if state is not None and "step" in state:
        return int(state["step"])

    parsed = _parse_step_from_dirname(resume_path)
    if parsed is not None:
        return parsed

    return 0


def _is_no_weight_decay_param(name: str) -> bool:
    n = str(name).lower()
    return ("bias" in n) or ("layer_norm" in n) or ("layernorm" in n) or ("ln_" in n)


def _get_transformer_layer_cls(module: nn.Module) -> type[nn.Module] | None:
    for attr_path in ("model.layers", "model.decoder.layers", "transformer.h", "layers"):
        cur: object = module
        ok = True
        for part in attr_path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if not ok:
            continue
        if isinstance(cur, (list, tuple)) and len(cur) > 0 and isinstance(cur[0], nn.Module):
            return type(cur[0])
        if isinstance(cur, nn.ModuleList) and len(cur) > 0:
            return type(cur[0])
    return None


def _build_fsdp_auto_wrap_policy(model: IronCellModel, fsdp_wrap: str):
    layer_classes: set[type[nn.Module]] = set()
    fsdp_wrap = str(fsdp_wrap).lower()
    if fsdp_wrap in {"generator_only", "full"}:
        gen_layer_cls = _get_transformer_layer_cls(model.generator)
        if gen_layer_cls is not None:
            layer_classes.add(gen_layer_cls)
    if fsdp_wrap == "full":
        comp_layer_cls = _get_transformer_layer_cls(model.compressor)
        if comp_layer_cls is not None:
            layer_classes.add(comp_layer_cls)
    if not layer_classes:
        return None
    return partial(transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)


def _extract_iron_state_dict(full_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "iron."
    out: dict[str, torch.Tensor] = {}
    for k, v in full_state.items():
        if k.startswith(prefix):
            out[k[len(prefix) :]] = v
    return out


def _ensure_pad_and_bos(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id to set pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id to set bos_token.")
        tokenizer.bos_token = tokenizer.eos_token


def _get_default_special_token_ids(tokenizer) -> list[int]:
    tokens = IronCellSpecialTokens()
    return [
        int(tokenizer.convert_tokens_to_ids(tokens.soc_token)),
        int(tokenizer.convert_tokens_to_ids(tokens.eoc_token)),
        int(tokenizer.convert_tokens_to_ids(tokens.v_none_token)),
    ]


def _validate_required_special_tokens(tokenizer) -> None:
    ids = _get_default_special_token_ids(tokenizer)
    if any(i < 0 for i in ids):
        raise ValueError(
            "Tokenizer missing required special tokens (<soc>, <eoc>, <v_none>). "
            "For resume: tokenizer must come from the checkpoint directory."
        )


def load_tokenizer(args) -> tuple[object, bool]:
    is_resume = args.resume_path is not None
    tok_src = args.resume_path if is_resume else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    _ensure_pad_and_bos(tokenizer)
    if is_resume:
        _validate_required_special_tokens(tokenizer)
    else:
        add_iron_cell_special_tokens(tokenizer)
        _validate_required_special_tokens(tokenizer)
    return tokenizer, is_resume


def load_model(args, tokenizer, device: torch.device, *, is_resume: bool) -> IronCellModel:
    if is_resume:
        config = IronCellConfig.from_pretrained(args.resume_path)
        setattr(config, "tokenizer_vocab_size", int(len(tokenizer)))
        model = IronCellModel.from_pretrained(args.resume_path, config=config).to(device)
        gen_vocab = int(model.generator.get_input_embeddings().weight.size(0))
        comp_vocab = int(model.compressor.get_input_embeddings().weight.size(0))
        if gen_vocab != len(tokenizer) or comp_vocab != len(tokenizer):
            raise ValueError(
                f"Tokenizer/model vocab mismatch: tokenizer={len(tokenizer)}, "
                f"generator={gen_vocab}, compressor={comp_vocab}. "
                "Load tokenizer from the same checkpoint directory."
            )
        return model

    config = IronCellConfig(
        compressor_model_name=args.model_name,
        generator_model_name=args.model_name,
        freeze_compressor=(args.phase == "phase1"),
        projector_init_type="identity",
        trainable_components=["javis", "embed_tokens", "special_tokens"],
        javis_query_warmup_samples=getattr(args, "javis_query_warmup_samples", None),
        javis_query_warmup_save_path=getattr(args, "javis_query_warmup_save_path", None),
        javis_num_queries=getattr(args, "javis_num_queries", 1),
    )
    model = IronCellModel(config).to(device)
    resize_and_smart_init_special_tokens(model.generator, tokenizer)
    resize_and_smart_init_special_tokens(model.compressor, tokenizer)
    return model


def warmup_init_javis_query(
    model: IronCellModel,
    loader,
    *,
    num_samples: int | None,
    save_path: str | None,
    use_dist: bool,
) -> None:
    if num_samples is None or int(num_samples) <= 0:
        return

    save_path = None if save_path is None or str(save_path).strip() == "" else str(save_path)

    device = model.device
    model_was_training = model.training
    model.eval()

    hidden_size = int(model.javis.hidden_size)
    num_queries = int(model.javis.num_queries)
    sum_vec = torch.zeros((num_queries, hidden_size), device=device, dtype=torch.float32)
    sumsq_vec = torch.zeros((num_queries, hidden_size), device=device, dtype=torch.float32)
    count = torch.zeros((num_queries,), device=device, dtype=torch.float32)

    saved: list[torch.Tensor] = []
    seen_samples = 0

    data_iter = iter(loader)
    while seen_samples < int(num_samples):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        bsz, num_chunks, chunk_len = chunk_ids.shape
        seen_samples += int(bsz)

        flat_ids = chunk_ids.view(bsz * num_chunks, chunk_len)
        flat_mask = chunk_mask.view(bsz * num_chunks, chunk_len)

        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                outputs = model.compressor(input_ids=flat_ids, attention_mask=flat_mask)
                hidden = outputs.last_hidden_state
                x = model.javis.pre_kv_hidden(hidden)

        segment_len = chunk_len // num_queries
        for q_i in range(num_queries):
            seg_start = q_i * segment_len
            seg_end = (q_i + 1) * segment_len if q_i < num_queries - 1 else chunk_len
            seg_mask = flat_mask[:, seg_start:seg_end].to(dtype=torch.bool)
            seg_x = x[:, seg_start:seg_end, :]
            seg_mask_flat = seg_mask.reshape(-1)
            seg_x_flat = seg_x.reshape(-1, seg_x.size(-1))
            seg_x_valid = seg_x_flat[seg_mask_flat]
            if seg_x_valid.numel() > 0:
                seg_x_float = seg_x_valid.float()
                sum_vec[q_i] += seg_x_float.sum(dim=0)
                sumsq_vec[q_i] += (seg_x_float * seg_x_float).sum(dim=0)
                count[q_i] += float(seg_x_valid.size(0))
                if save_path is not None:
                    saved.append(seg_x_valid.detach().to("cpu"))

    if use_dist and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sum_vec, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sumsq_vec, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

    denom = count.clamp(min=1.0).unsqueeze(-1)
    mean = sum_vec / denom
    var = (sumsq_vec / denom) - (mean * mean)
    var = var.clamp(min=0.0)
    std = torch.sqrt(var + 1e-6)

    with torch.no_grad():
        q = torch.zeros((num_queries, hidden_size), device=device, dtype=mean.dtype)
        basis = []
        for q_i in range(num_queries):
            vec = mean[q_i]
            for prev in basis:
                denom_proj = torch.dot(prev, prev) + 1e-8
                vec = vec - (torch.dot(vec, prev) / denom_proj) * prev
            scale = mean[q_i].norm() / (vec.norm() + 1e-8)
            vec = vec * scale
            vec = vec + std[q_i] * 1e-3 * torch.randn_like(vec)
            q[q_i] = vec
            basis.append(vec)

        if use_dist and torch.distributed.is_initialized():
            torch.distributed.broadcast(q, src=0)
        model.javis.q.copy_(q.to(dtype=model.javis.q.dtype))

    print(f"init q now {num_samples} {num_queries=}  {torch.distributed.is_initialized()=}  {torch.distributed.get_rank()=} \n" + "="*50)
    
    if not use_dist or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0):
        import sys
        with torch.no_grad():
            q_final = model.javis.q.detach().float()
            if num_queries > 1:
                # 计算余弦相似度矩阵
                norm_q = q_final / (q_final.norm(dim=1, keepdim=True) + 1e-8)
                similarity_matrix = torch.matmul(norm_q, norm_q.T)
                
                print("\n" + "="*50)
                print(f"🚀 Javis Query Initialization Diagnostic (num_queries={num_queries})")
                print(f"📊 Cosine Similarity Matrix:\n{similarity_matrix}")
                
                # 提取 Q0 和 Q1 的相似度（针对 16:2）
                if num_queries == 2:
                    cos_sim = similarity_matrix[0, 1].item()
                    print(f"🔍 Q0 <-> Q1 Similarity: {cos_sim:.6f} (Expect near 0.0)")
                    print(f"🔍 Q0 <-> Q1 Similarity: {cos_sim:.6f}")
                sys.stdout.flush()
                print("="*50 + "\n")

    if save_path is not None:
        is_rank0 = True
        if use_dist and torch.distributed.is_initialized():
            is_rank0 = torch.distributed.get_rank() == 0
        if is_rank0:
            data = torch.cat(saved, dim=0) if saved else torch.empty((0, hidden_size))
            out_path = Path(save_path)
            if not out_path.is_absolute():
                out_path = Path(os.getcwd()) / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"hidden": data, "mean": mean.cpu(), "std": std.cpu(), "q_init": model.javis.q.detach().cpu()}, out_path)

    if model_was_training:
        model.train()


def configure_special_embedding_mode(args, model: IronCellModel, tokenizer, *, is_resume: bool) -> None:
    if not bool(args.train_only_special_token_embeddings):
        return

    token_ids = _get_default_special_token_ids(tokenizer)

    if is_resume and int(getattr(model.special_token_embeddings, "num_embeddings", 0)) > 0:
        model.special_token_embeddings.weight.requires_grad = True
    else:
        model.enable_special_token_training(token_ids, init_from_generator=True)

    base_embed = model.generator.get_input_embeddings()
    base_embed.weight.requires_grad = False
