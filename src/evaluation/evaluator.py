"""Unified evaluation utilities for IronCell."""
from __future__ import annotations

import math
import torch
from torch.utils.data import DataLoader


def compute_ppl(
    sum_loss_times_tokens: torch.Tensor,
    sum_tokens: torch.Tensor,
    distributed: bool = False,
) -> tuple[float, float]:
    """Compute eval loss and perplexity from accumulated values."""
    if distributed and torch.distributed.is_initialized():
        torch.distributed.all_reduce(sum_loss_times_tokens, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sum_tokens, op=torch.distributed.ReduceOp.SUM)

    denom = float(sum_tokens.clamp(min=1.0).item())
    eval_loss = float((sum_loss_times_tokens / denom).item())
    ppl = math.exp(min(eval_loss, 80.0))
    return eval_loss, ppl


def run_eval_loop(
    step_module: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 0,
    distributed: bool = False,
    use_amp: bool = True,
) -> tuple[float, float, int]:
    """Run evaluation loop over a DataLoader. Returns (eval_loss, ppl, total_tokens)."""
    step_module.eval()
    sum_loss_times_tokens = torch.zeros((), device=device, dtype=torch.float32)
    sum_tokens = torch.zeros((), device=device, dtype=torch.float32)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches > 0 and i >= max_batches:
                break
            n_tokens = (batch.labels != -100).sum().to(dtype=torch.float32)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, *_ = step_module(batch)
            sum_loss_times_tokens += loss.detach().float() * n_tokens.to(device=device)
            sum_tokens += n_tokens.to(device=device)

    eval_loss, ppl = compute_ppl(sum_loss_times_tokens, sum_tokens, distributed=distributed)
    total_tokens = int(sum_tokens.item())
    step_module.train()
    return eval_loss, ppl, total_tokens
