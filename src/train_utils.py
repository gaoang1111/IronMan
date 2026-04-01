"""
Backward compatibility shim - imports from src.utils.

New code should import directly from src.utils:
    from src.utils import load_model, load_tokenizer, JsonlDataset, ...
"""
from __future__ import annotations

# Re-export all from src.utils for backward compatibility
from src.utils.checkpoint import (
    save_checkpoint,
    save_checkpoint_fsdp,
    load_checkpoint,
)
from src.utils.data import JsonlDataset
from src.utils.distributed import (
    build_fsdp_auto_wrap_policy as _build_fsdp_auto_wrap_policy,
    is_no_weight_decay_param as _is_no_weight_decay_param,
)
from src.utils.model_loader import load_tokenizer, load_model
from src.utils.javis_init import warmup_init_javis_query, configure_special_embedding_mode

__all__ = [
    "save_checkpoint",
    "save_checkpoint_fsdp",
    "load_checkpoint",
    "JsonlDataset",
    "_build_fsdp_auto_wrap_policy",
    "_is_no_weight_decay_param",
    "load_tokenizer",
    "load_model",
    "warmup_init_javis_query",
    "configure_special_embedding_mode",
]
