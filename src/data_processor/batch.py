"""ZipperBatch dataclass - shared by fixed and adaptive chunking."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ZipperBatch:
    """
    A batch that represents the Iron-Cell zipper layout.

    Tensors are padded to the batch max sequence length.

    Fields:
        zipper_input_ids: [B, S] token ids used to build base embeddings
            (compressed vector slots are placeholder ids; their embeddings will
            be overwritten later).
        labels: [B, S] language modeling labels; zipper prefix is -100.
        attention_mask_2d: [B, S, S] boolean attention mask where True means
            "allowed to attend".
        position_ids: [B, S] manual RoPE position ids (padded with 0).
        chunk_input_ids: [B, C, Lc] token ids for compressor chunks.
        chunk_attention_mask: [B, C, Lc] attention mask for compressor chunks.
        memory_positions: [B, C] positions in zipper sequence where V slots sit.
        prefix_lens: [B] prefix length (SOC + (V,EOC)*C).
        valid_lens: [B] unpadded zipper length.
        teacher_attn_targets: [B, T] teacher attention targets (optional).
        teacher_hidden_targets: [B, V, H] teacher hidden targets for V slots (optional).
        valid_v_lens: [B] number of valid V slots in teacher_hidden_targets (optional).
        teacher_hidden_target_layer: scalar int tensor; teacher target layer (optional).
    """

    zipper_input_ids: torch.LongTensor
    labels: torch.LongTensor
    attention_mask_2d: torch.BoolTensor
    position_ids: torch.LongTensor
    chunk_input_ids: torch.LongTensor
    chunk_attention_mask: torch.LongTensor
    memory_positions: torch.LongTensor
    prefix_lens: torch.LongTensor
    valid_lens: torch.LongTensor
    teacher_attn_targets: torch.FloatTensor
    teacher_hidden_targets: torch.FloatTensor
    valid_v_lens: torch.LongTensor
    teacher_hidden_target_layer: torch.LongTensor
