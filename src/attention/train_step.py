from __future__ import annotations

"""
Unified TrainStepModule for both DDP and FSDP training.

This module wraps IronCellModel and handles:
1. Deep KV injection via DEEP_KV_CONTEXT
2. Gradient isolation for stable training
3. Loss computation (LM loss + L2 regularization + orthogonality penalty)
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import IronCellModel
from .kv_context import DEEP_KV_CONTEXT, set_kv_context, clear_kv_context
from .patched_attention import smart_hybrid_attention_forward


class TrainStepModule(nn.Module):
    """
    Unified training step module for IronCell.
    
    Handles:
    - Compression via Javis
    - Deep KV injection to all 32 layers
    - Gradient isolation (detach + surrogate loss for DDP compatibility)
    - Loss computation with regularization
    
    Works with both DDP and FSDP distributed training.
    """

    def __init__(
        self,
        iron: IronCellModel,
        *,
        phase: str,
        l2_coeff: float = 1e-4,
        javis_q_cos_coeff: float = 1.0,
        distill_coeff: float = 0.0,
        distill_layers: list[int] | None = None,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.iron = iron
        self.phase = str(phase)
        self.l2_coeff = float(l2_coeff)
        self.javis_q_cos_coeff = float(javis_q_cos_coeff)
        self.distill_coeff = float(distill_coeff)
        self.distill_layers = distill_layers or []
        self.chunk_size = int(chunk_size)
        self.grad_probe = False
        self.grad_probe_sums: dict[str, torch.Tensor] = {}

        # Monkey-patch attention layers for deep KV injection
        for layer in self.iron.generator.model.layers:
            layer.self_attn.forward = types.MethodType(smart_hybrid_attention_forward, layer.self_attn)

    def reset_grad_probe(self) -> None:
        """Reset gradient probe accumulators."""
        self.grad_probe_sums = {}

    def _register_grad_probe(self, name: str, tensor: torch.Tensor) -> None:
        """Register a gradient probe hook on a tensor."""
        if not self.grad_probe or not tensor.requires_grad:
            return
        
        def _hook(grad: torch.Tensor) -> None:
            val = (grad.float() ** 2).sum()
            if name in self.grad_probe_sums:
                self.grad_probe_sums[name] = self.grad_probe_sums[name] + val
            else:
                self.grad_probe_sums[name] = val
        
        tensor.register_hook(_hook)

    def forward(self, batch, *, return_metrics: bool = False):
        """
        Forward pass with deep KV injection.
        
        Args:
            batch: ZipperBatch containing all inputs
            return_metrics: Whether to return detailed metrics
            
        Returns:
            If return_metrics=False:
                (total_loss, l2_loss, q_cos_similarity)
            If return_metrics=True:
                (total_loss, l2_loss, javis_metrics, q_cos_similarity)
        """
        device = self.iron.device
        
        # Move batch tensors to device
        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        attn_2d = batch.attention_mask_2d.to(device)
        position_ids = batch.position_ids.to(device)
        labels = batch.labels.to(device)

        # =========================================================
        # 1. Compute compressed vectors and deep KV
        # =========================================================
        memory_out = self.iron.compute_compressed_vectors(
            chunk_input_ids=chunk_ids,
            chunk_attention_mask=chunk_mask,
            return_metrics=return_metrics,
        )
        
        javis_metrics = None
        if return_metrics:
            memory_hook, memory_vectors, deep_layer_kvs, javis_metrics, current_out_cos = memory_out
        else:
            memory_hook, memory_vectors, deep_layer_kvs, current_out_cos = memory_out
            
        self._register_grad_probe("memory_vectors", memory_hook)

        # =========================================================
        # 2. Set global KV context for attention layers
        # =========================================================
        set_kv_context(deep_layer_kvs, mem_pos, self.iron.javis.num_queries)

        # =========================================================
        # 3. Build inputs_embeds with memory vectors
        # =========================================================
        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors, 
            memory_positions=mem_pos,
        )
        self._register_grad_probe("inputs_embeds", inputs_embeds)

        # =========================================================
        # 4. Generator forward
        # =========================================================
        try:
            out = self.iron(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_2d,
                position_ids=position_ids, 
                labels=labels,
                use_cache=False,
            )
        finally:
            # Always clear context to free memory
            clear_kv_context()
        
        if hasattr(out, "logits") and out.logits.requires_grad:
            self._register_grad_probe("logits", out.logits)

        gen_loss = out.loss

        # =========================================================
        # 5. Compute regularization losses
        # =========================================================
        # L2 regularization on memory vectors (only in phase2)
        l2_loss = (
            memory_vectors.norm(p=2, dim=-1).mean() 
            if self.phase == "phase2" 
            else torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
        )
        
        # Orthogonality penalty on query vectors
        q_params = self.iron.javis.q_base  # [G, Q, H]
        Q = q_params.size(1)
        
        if Q >= 2:
            # Compute cosine similarity matrix for all query pairs
            q_norm = F.normalize(q_params, p=2, dim=-1)
            sim_matrix = torch.bmm(q_norm, q_norm.transpose(1, 2))  # [G, Q, Q]
            
            # Mask out diagonal (self-similarity)
            eye_mask = torch.eye(Q, device=gen_loss.device, dtype=torch.bool).unsqueeze(0)
            off_diag_sim = sim_matrix.masked_select(~eye_mask)
            
            # Penalize absolute similarity > 0.1
            mean_q_cos = off_diag_sim.abs().mean()
            ortho_penalty = torch.relu(mean_q_cos.to(gen_loss.dtype) - 0.1)
        else:
            ortho_penalty = torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
            mean_q_cos = torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
            
        # Total loss
        total_loss = gen_loss + (self.l2_coeff * l2_loss) + (self.javis_q_cos_coeff * ortho_penalty)

        # =========================================================
        # 6. Return
        # =========================================================
        if return_metrics:
            clean_metrics = None
            if javis_metrics is not None:
                clean_metrics = {
                    k: (v.detach().item() if isinstance(v, torch.Tensor) else v) 
                    for k, v in javis_metrics.items()
                }
            return total_loss, l2_loss.detach(), clean_metrics, mean_q_cos.detach()
        else:
            return total_loss, l2_loss.detach(), mean_q_cos.detach()
