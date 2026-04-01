"""Javis query initialization and special embedding configuration."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from src.models import IronCellModel
from src.token_utils import IronCellSpecialTokens


def warmup_init_javis_query(
    model: IronCellModel,
    loader,
    *,
    num_samples: int | None,
    save_path: str | None,
    use_dist: bool,
) -> None:
    """Initialize Javis query vectors from compressor hidden states."""
    if num_samples is None or int(num_samples) <= 0:
        return

    save_path = None if save_path is None or str(save_path).strip() == "" else str(save_path)
    device = model.device
    model_was_training = model.training
    model.eval()

    hidden_size = int(model.javis.hidden_size)
    num_queries = int(model.javis.num_queries)
    num_query_group = int(model.javis.num_query_group)

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
        base_q = torch.zeros((num_queries, hidden_size), device=device, dtype=mean.dtype)
        basis = []
        for q_i in range(num_queries):
            vec = mean[q_i]
            for prev in basis:
                denom_proj = torch.dot(prev, prev) + 1e-8
                vec = vec - (torch.dot(vec, prev) / denom_proj) * prev
            scale = mean[q_i].norm() / (vec.norm() + 1e-8)
            vec = vec * scale
            vec = vec + std[q_i] * 1e-3 * torch.randn_like(vec)
            base_q[q_i] = vec
            basis.append(vec)

        q_groups = torch.zeros((num_query_group, num_queries, hidden_size), device=device, dtype=mean.dtype)
        for g in range(num_query_group):
            group_noise = torch.randn_like(base_q) * 1e-4
            q_groups[g] = base_q + group_noise

        if use_dist and torch.distributed.is_initialized():
            torch.distributed.broadcast(q_groups, src=0)

        model.javis.q_base.copy_(q_groups.to(dtype=model.javis.q_base.dtype))

    print(f"init q: {num_samples=} {num_query_group=} {num_queries=}")

    if save_path is not None:
        is_rank0 = not use_dist or (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0)
        if is_rank0:
            data = torch.cat(saved, dim=0) if saved else torch.empty((0, hidden_size))
            out_path = Path(save_path)
            if not out_path.is_absolute():
                out_path = Path(os.getcwd()) / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "hidden": data,
                "mean": mean.cpu(),
                "std": std.cpu(),
                "q_init": model.javis.q_base.detach().cpu()
            }, out_path)

    if model_was_training:
        model.train()


def configure_special_embedding_mode(
    args, model: IronCellModel, tokenizer, *, is_resume: bool
) -> None:
    """Configure special token embedding training mode."""
    if not bool(args.train_only_special_token_embeddings):
        return

    tokens = IronCellSpecialTokens()
    token_ids = [
        int(tokenizer.convert_tokens_to_ids(tokens.soc_token)),
        int(tokenizer.convert_tokens_to_ids(tokens.eoc_token)),
        int(tokenizer.convert_tokens_to_ids(tokens.v_none_token)),
    ]

    if is_resume and int(getattr(model.special_token_embeddings, "num_embeddings", 0)) > 0:
        model.special_token_embeddings.weight.requires_grad = True
    else:
        model.enable_special_token_training(token_ids, init_from_generator=True)

    base_embed = model.generator.get_input_embeddings()
    base_embed.weight.requires_grad = False
