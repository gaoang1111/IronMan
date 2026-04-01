from __future__ import annotations

"""
Patched LlamaAttention forward with deep KV injection support.

This module provides a monkey-patched attention forward function that:
1. Reads KV values from DEEP_KV_CONTEXT
2. Performs residual injection at specified memory positions
3. Maintains full compatibility with the original LlamaAttention interface
"""

import math

import torch
import torch.nn as nn

from .kv_context import DEEP_KV_CONTEXT


def smart_hybrid_attention_forward(
    self,
    hidden_states: torch.Tensor,
    *args,
    **kwargs,
):
    """
    Patched LlamaAttention.forward with deep KV residual injection.
    
    This function is designed to be monkey-patched onto LlamaAttention instances.
    It reads KV values from DEEP_KV_CONTEXT and performs residual injection
    at the positions specified by memory_positions.
    
    Injection method: Residual addition (orig_k + k_javis, orig_v + v_javis)
    """
    # =========================================================
    # 1. Parse native arguments
    # =========================================================
    attention_mask = kwargs.get("attention_mask", args[0] if len(args) > 0 else None)
    position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
    past_key_value = kwargs.get("past_key_value", args[2] if len(args) > 2 else None)
    output_attentions = kwargs.get("output_attentions", args[3] if len(args) > 3 else False)

    bsz, q_len, _ = hidden_states.size()
    layer_idx = getattr(self, "layer_idx", None)
    
    # =========================================================
    # 2. Native QKV projection
    # =========================================================
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    # =========================================================
    # 3. Deep KV Residual Injection
    # =========================================================
    if layer_idx is not None and DEEP_KV_CONTEXT.get("layer_kvs") is not None:
        k_javis_raw, v_javis_raw = DEEP_KV_CONTEXT["layer_kvs"][layer_idx]
        mem_pos = DEEP_KV_CONTEXT["memory_positions"]
        num_q = DEEP_KV_CONTEXT["num_queries"]

        # Must clone to avoid in-place modification errors
        key_states = key_states.clone()
        value_states = value_states.clone()

        # Residual injection at memory positions
        for b in range(bsz):
            for c in range(mem_pos.size(1)):
                start_idx = int(mem_pos[b, c].item())
                if start_idx >= 0 and start_idx + num_q <= q_len:
                    # Get original KV at memory positions
                    orig_k = key_states[b, :, start_idx : start_idx + num_q, :].clone()
                    orig_v = value_states[b, :, start_idx : start_idx + num_q, :].clone()
                    
                    # Residual addition: orig + javis
                    key_states[b, :, start_idx : start_idx + num_q, :] = orig_k + k_javis_raw[b, c].to(key_states.dtype)
                    value_states[b, :, start_idx : start_idx + num_q, :] = orig_v + v_javis_raw[b, c].to(value_states.dtype)

    # =========================================================
    # 4. RoPE rotation and GQA expansion
    # =========================================================
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is None:
        raise ValueError("position_embeddings must be provided in kwargs")

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(
            bsz, self.config.num_key_value_heads, num_key_value_groups, q_len, self.head_dim
        ).reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            bsz, self.config.num_key_value_heads, num_key_value_groups, q_len, self.head_dim
        ).reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)
    
    # Transpose for matmul: [B, H, L, D] -> [B, H, D, L]
    key_states = key_states.transpose(2, 3)

    # =========================================================
    # 5. Chunked attention computation (memory efficient)
    # =========================================================
    attn_output = torch.zeros_like(query_states)
    
    CHUNK_SIZE = 1024
    for i in range(0, q_len, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, q_len)
        q_chunk = query_states[:, :, i:end, :]
        
        attn_weights = torch.matmul(q_chunk, key_states) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, i:end, :]
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output[:, :, i:end, :] = torch.matmul(attn_weights, value_states)
        
        del attn_weights, q_chunk

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    
    del query_states, key_states, value_states
    
    # =========================================================
    # 6. Return with correct signature
    # =========================================================
    if output_attentions:
        return attn_output, None, past_key_value
    else:
        return attn_output, past_key_value
