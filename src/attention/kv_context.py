from __future__ import annotations

"""
Global context for deep KV injection.

This module provides a thread-local-like global context that allows
the patched attention forward to access Javis-computed KV values
without modifying the HuggingFace model's forward signature.
"""

import torch

# Global context for deep KV injection
# This is set by TrainStepModule before calling generator.forward()
# and read by smart_hybrid_attention_forward in each attention layer.
DEEP_KV_CONTEXT: dict = {
    "layer_kvs": None,           # List[Tuple[K, V]], 32 layers, each [B, C, num_kv_heads, Q, head_dim]
    "memory_positions": None,    # [B, C] positions where V slots sit in the sequence
    "num_queries": 2,            # Number of query vectors per chunk
}


def set_kv_context(
    layer_kvs: list[tuple[torch.Tensor, torch.Tensor]] | None,
    memory_positions: torch.Tensor | None,
    num_queries: int,
) -> None:
    """
    Set the global KV context for deep injection.
    
    Args:
        layer_kvs: List of 32 (K, V) tuples, each with shape [B, C, num_kv_heads, Q, head_dim]
        memory_positions: [B, C] positions where to inject KV in the sequence
        num_queries: Number of query vectors per chunk
    """
    DEEP_KV_CONTEXT["layer_kvs"] = layer_kvs
    DEEP_KV_CONTEXT["memory_positions"] = memory_positions
    DEEP_KV_CONTEXT["num_queries"] = int(num_queries)


def clear_kv_context() -> None:
    """
    Clear the global KV context to free memory.
    Should be called after each forward pass.
    """
    DEEP_KV_CONTEXT["layer_kvs"] = None
    DEEP_KV_CONTEXT["memory_positions"] = None
