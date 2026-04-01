"""Collators for batching zipper layout data."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase

from .batch import ZipperBatch
from .builder import ZipperBuilderBase, ZipperBuilder, AdaptiveZipperBuilder


class IronCellCollatorBase(ABC):
    """
    Base class for collating zipper batches.
    
    Subclasses implement _parse_item() and _create_builder() for their chunking strategy.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        pad_to_multiple_of: int | None = 8,
        buffer_size: int = 0,
        num_v: int = 1,
        random_gate: float = 0,
        truncate_len: int | None = None,
        teacher_targets: torch.Tensor | None = None,
        teacher_hidden_targets: torch.Tensor | None = None,
        teacher_hidden_valid_lens: torch.Tensor | None = None,
        teacher_hidden_target_layer: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.buffer_size = int(buffer_size)
        self.num_v = int(num_v)
        self.random_gate = float(random_gate)
        self.truncate_len = truncate_len
        self.teacher_targets = teacher_targets
        self.teacher_hidden_targets = teacher_hidden_targets
        self.teacher_hidden_valid_lens = teacher_hidden_valid_lens
        self.teacher_hidden_target_layer = teacher_hidden_target_layer

    @abstractmethod
    def _parse_item(self, item: object) -> tuple[str, int | None, dict]:
        """
        Parse a dataset item.
        
        Returns:
            (text, idx, extra_kwargs) where extra_kwargs are passed to _create_builder.
        """
        pass

    @abstractmethod
    def _create_builder(self, text: str, extra_kwargs: dict) -> ZipperBuilderBase:
        """Create the appropriate builder for this chunking strategy."""
        pass

    def __call__(self, items: Sequence[object]) -> ZipperBatch:
        """Collate items into a ZipperBatch."""
        if len(items) == 0:
            raise ValueError("Empty batch")

        device = torch.device("cpu")
        pad_id = (
            int(self.tokenizer.pad_token_id)
            if self.tokenizer.pad_token_id is not None
            else int(self.tokenizer.eos_token_id)
        )

        # Parse all items
        texts: list[str] = []
        idxs: list[int] = []
        extras: list[dict] = []
        for item in items:
            text, idx, extra = self._parse_item(item)
            texts.append(text)
            if idx is not None:
                idxs.append(int(idx))
            extras.append(extra)

        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device) if len(idxs) == len(texts) else None

        # Create builders
        builders: list[ZipperBuilderBase] = [
            self._create_builder(text, extra) for text, extra in zip(texts, extras)
        ]

        # Compute max dimensions
        max_zip_len = max(b.valid_len for b in builders)
        max_num_cmp_chunks = max(b.num_cmp_chunks for b in builders)
        max_cmp_chunk_len = 1
        for b in builders:
            for c in b.cmp_wrapped_chunks:
                max_cmp_chunk_len = max(max_cmp_chunk_len, len(c))

        # Pad to multiple
        if self.pad_to_multiple_of is not None and max_zip_len % self.pad_to_multiple_of != 0:
            m = self.pad_to_multiple_of
            max_zip_len = ((max_zip_len + m - 1) // m) * m

        # Initialize tensors
        batch_size = len(texts)
        zipper_input_ids = torch.full((batch_size, max_zip_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((batch_size, max_zip_len), -100, dtype=torch.long, device=device)
        attention_mask_2d = torch.zeros((batch_size, max_zip_len, max_zip_len), dtype=torch.bool, device=device)
        position_ids = torch.zeros((batch_size, max_zip_len), dtype=torch.long, device=device)
        prefix_lens = torch.zeros((batch_size,), dtype=torch.long, device=device)
        valid_lens = torch.zeros((batch_size,), dtype=torch.long, device=device)

        chunk_input_ids = torch.full(
            (batch_size, max_num_cmp_chunks, max_cmp_chunk_len),
            pad_id, dtype=torch.long, device=device,
        )
        chunk_attention_mask = torch.zeros(
            (batch_size, max_num_cmp_chunks, max_cmp_chunk_len),
            dtype=torch.long, device=device,
        )
        memory_positions = torch.full(
            (batch_size, max_num_cmp_chunks),
            -1, dtype=torch.long, device=device,
        )

        # Teacher targets
        if self.teacher_targets is not None:
            if idxs_t is None:
                raise ValueError("teacher_targets requires dataset indices.")
            teacher_attn_targets = self.teacher_targets.index_select(0, idxs_t).to(dtype=torch.float32)
        else:
            teacher_attn_targets = torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

        if self.teacher_hidden_targets is not None:
            if idxs_t is None:
                raise ValueError("teacher_hidden_targets requires dataset indices.")
            if self.teacher_hidden_valid_lens is None:
                raise ValueError("teacher_hidden_valid_lens is required when teacher_hidden_targets is set.")
            teacher_hidden_targets = self.teacher_hidden_targets.index_select(0, idxs_t).to(dtype=torch.float32)
            valid_v_lens = self.teacher_hidden_valid_lens.index_select(0, idxs_t).to(dtype=torch.long)
        else:
            teacher_hidden_targets = torch.zeros((batch_size, 0, 0), dtype=torch.float32, device=device)
            valid_v_lens = torch.zeros((batch_size,), dtype=torch.long, device=device)

        teacher_hidden_target_layer = torch.tensor(
            -1 if self.teacher_hidden_target_layer is None else int(self.teacher_hidden_target_layer),
            dtype=torch.long, device=device,
        )

        # Fill tensors for each sample
        for b_idx, builder in enumerate(builders):
            valid_len = builder.valid_len
            prefix_len = builder.prefix_len

            zipper_input_ids[b_idx, :valid_len] = torch.tensor(
                builder.gen_input_ids, dtype=torch.long, device=device
            )

            labels_b = builder.build_gen_labels(device=device)
            labels[b_idx, :valid_len] = labels_b
            labels[b_idx, valid_len:] = -100

            prefix_lens[b_idx] = prefix_len
            valid_lens[b_idx] = valid_len

            memory_positions[b_idx, : builder.num_cmp_chunks] = torch.tensor(
                builder.memory_positions, dtype=torch.long, device=device
            )

            for i, c in enumerate(builder.cmp_wrapped_chunks):
                ln = len(c)
                chunk_input_ids[b_idx, i, :ln] = torch.tensor(c, dtype=torch.long, device=device)
                chunk_attention_mask[b_idx, i, :ln] = 1

            attn, pos_ids = builder.build_gen_attention_and_pos(seq_len=max_zip_len, device=device)
            attention_mask_2d[b_idx] = attn
            position_ids[b_idx] = pos_ids

        return ZipperBatch(
            zipper_input_ids=zipper_input_ids,
            labels=labels,
            attention_mask_2d=attention_mask_2d,
            position_ids=position_ids,
            chunk_input_ids=chunk_input_ids,
            chunk_attention_mask=chunk_attention_mask,
            memory_positions=memory_positions,
            prefix_lens=prefix_lens,
            valid_lens=valid_lens,
            teacher_attn_targets=teacher_attn_targets,
            teacher_hidden_targets=teacher_hidden_targets,
            valid_v_lens=valid_v_lens,
            teacher_hidden_target_layer=teacher_hidden_target_layer,
        )


class IronCellCollator(IronCellCollatorBase):
    """Collator for fixed-size chunking."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        chunk_size: int = 16,
        pad_to_multiple_of: int | None = 8,
        buffer_size: int = 0,
        num_v: int = 1,
        random_gate: float = 0,
        truncate_len: int | None = None,
        teacher_targets: torch.Tensor | None = None,
        teacher_hidden_targets: torch.Tensor | None = None,
        teacher_hidden_valid_lens: torch.Tensor | None = None,
        teacher_hidden_target_layer: int | None = None,
    ) -> None:
        self.chunk_size = int(chunk_size)
        super().__init__(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            buffer_size=buffer_size,
            num_v=num_v,
            random_gate=random_gate,
            truncate_len=truncate_len,
            teacher_targets=teacher_targets,
            teacher_hidden_targets=teacher_hidden_targets,
            teacher_hidden_valid_lens=teacher_hidden_valid_lens,
            teacher_hidden_target_layer=teacher_hidden_target_layer,
        )

    def _parse_item(self, item: object) -> tuple[str, int | None, dict]:
        """Parse item: expects (text, idx) tuple or just text."""
        if isinstance(item, tuple):
            text, idx = item
            return str(text), int(idx), {}
        return str(item), None, {}

    def _create_builder(self, text: str, extra_kwargs: dict) -> ZipperBuilder:
        return ZipperBuilder(
            self.tokenizer,
            text,
            chunk_size=self.chunk_size,
            buffer_size=self.buffer_size,
            num_v=self.num_v,
            random_gate=self.random_gate,
            truncate_len=self.truncate_len,
        )


class AdaptiveIronCellCollator(IronCellCollatorBase):
    """Collator for adaptive (variable-length) chunking."""

    def _parse_item(self, item: object) -> tuple[str, int | None, dict]:
        """Parse item: expects (text, idx, chunk_lens) or dict with those fields."""
        if isinstance(item, dict):
            text = item.get("text")
            if not isinstance(text, str):
                raise ValueError("Adaptive collator expects item['text'] as str.")
            idx = item.get("idx")
            idx_i = int(idx) if idx is not None else None
            lens = item.get("chunk_lens", item.get("raw_chunk_lens"))
            if lens is None:
                raise ValueError("Adaptive collator expects 'chunk_lens' or 'raw_chunk_lens'.")
            return text, idx_i, {"raw_chunk_lens": lens}

        if isinstance(item, tuple):
            if len(item) == 3:
                text, idx, lens = item
                return str(text), int(idx), {"raw_chunk_lens": lens}
            if len(item) == 2:
                raise ValueError("Adaptive collator requires chunk_lens in dataset items.")

        raise ValueError(f"Unsupported item type for adaptive collator: {type(item)}")

    def _create_builder(self, text: str, extra_kwargs: dict) -> AdaptiveZipperBuilder:
        return AdaptiveZipperBuilder(
            self.tokenizer,
            text,
            raw_chunk_lens=extra_kwargs["raw_chunk_lens"],
            buffer_size=self.buffer_size,
            num_v=self.num_v,
            random_gate=self.random_gate,
            truncate_len=self.truncate_len,
        )
