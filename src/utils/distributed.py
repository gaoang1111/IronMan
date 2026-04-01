"""Distributed training utilities."""
from __future__ import annotations

from functools import partial

import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from src.models import IronCellModel


def build_fsdp_auto_wrap_policy(model: IronCellModel, fsdp_wrap: str):
    """Build FSDP auto wrap policy based on transformer layers."""
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


def _get_transformer_layer_cls(module: nn.Module) -> type[nn.Module] | None:
    """Auto-detect transformer layer class from a model."""
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


def is_no_weight_decay_param(name: str) -> bool:
    """Check if parameter should skip weight decay."""
    n = str(name).lower()
    return ("bias" in n) or ("layer_norm" in n) or ("layernorm" in n) or ("ln_" in n)
