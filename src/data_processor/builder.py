"""Zipper layout builders with inheritance for fixed/adaptive chunking."""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase


class ZipperBuilderBase(ABC):
    """
    Base class for building zipper layout sequences.
    
    Subclasses implement _compute_segments() to define chunking strategy.
    All other logic (prefix, mask, labels) is shared.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        *,
        buffer_size: int = 0,
        num_v: int = 1,
        random_gate: float = 0,
        truncate_len: int | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.buffer_size = int(buffer_size)
        self.num_v = int(num_v)
        self.random_gate = float(random_gate)
        self.truncate_len = truncate_len

        # Initialize special tokens
        self._init_special_tokens(tokenizer)

        # Tokenize prompt
        self.raw_core_ids = tokenizer.encode(
            prompt, add_special_tokens=False, max_length=truncate_len, truncation=True
        )
        self.raw_len = len(self.raw_core_ids)

        # Subclass computes segments
        self._compute_segments()

        # Validate
        if self.num_cmp_chunks < 1:
            raise ValueError(
                f"Invalid sample for compression: num_cmp_chunks must be >= 1. "
                f"raw_len={self.raw_len}, num_segments={self.num_segments}"
            )

        # Build prefix and generator sequence
        self._build_prefix()
        self._build_gen_sequence()

    def _init_special_tokens(self, tokenizer: PreTrainedTokenizerBase) -> None:
        """Initialize special token IDs."""
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("Tokenizer must define both bos_token_id and eos_token_id.")
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.soc_id = int(tokenizer.convert_tokens_to_ids("<soc>"))
        self.eoc_id = int(tokenizer.convert_tokens_to_ids("<eoc>"))
        self.v_none_id = int(tokenizer.convert_tokens_to_ids("<v_none>"))
        if self.soc_id < 0 or self.eoc_id < 0 or self.v_none_id < 0:
            raise ValueError("Special tokens not found in tokenizer; call add_iron_cell_special_tokens first.")

    @abstractmethod
    def _compute_segments(self) -> None:
        """
        Compute chunking segments. Must set:
        - self._segment_lens: list[int] - length of each segment
        - self.num_cmp_chunks: int - number of chunks for compressor
        - self.cmp_wrapped_chunks: list[list[int]] - token ids for each compressor chunk
        """
        pass

    @property
    def segment_lens(self) -> list[int]:
        """Length of each segment in the raw content."""
        return self._segment_lens

    @property
    def num_segments(self) -> int:
        """Number of segments (control groups)."""
        return len(self._segment_lens)

    def _build_prefix(self) -> None:
        """Build prefix: [BOS, SOC, (V*num_v, EOC) * num_segments]."""
        prefix_ids: list[int] = [self.bos_id, self.soc_id]
        for _ in range(self.num_segments):
            prefix_ids.extend([self.v_none_id] * self.num_v)
            prefix_ids.append(self.eoc_id)
        self.prefix_ids = prefix_ids
        self.prefix_len = len(prefix_ids)

    def _build_gen_sequence(self) -> None:
        """Build generator input sequence and memory positions."""
        self.gen_input_ids = self.prefix_ids + self.raw_core_ids
        self.valid_len = len(self.gen_input_ids)
        
        # Memory positions: where V slots sit for each compressor chunk
        self.memory_positions = [
            2 + (self.num_v + 1) * k for k in range(1, self.num_cmp_chunks + 1)
        ]

    def _get_segment_starts(self) -> list[int]:
        """Compute cumulative start positions for each segment."""
        starts = [0]
        for ln in self._segment_lens[:-1]:
            starts.append(starts[-1] + ln)
        return starts

    def build_gen_labels(self, device: torch.device) -> torch.LongTensor:
        """Build labels tensor with zipper layout."""
        input_ids_t = torch.tensor(self.gen_input_ids, dtype=torch.long, device=device)
        ignore_index = -100
        labels = torch.full_like(input_ids_t, ignore_index)

        raw_phys_start = self.prefix_len
        segment_starts = self._get_segment_starts()

        # Random gate for visible length
        visible_cap = 14 if random.random() < self.random_gate else None

        # Label raw content (shifted by 1 for next-token prediction)
        for k, (seg_start, seg_len) in enumerate(zip(segment_starts, self._segment_lens)):
            curr_start = raw_phys_start + seg_start
            curr_end = curr_start + seg_len
            if visible_cap is not None:
                curr_end = min(curr_end, curr_start + visible_cap)

            if curr_end - curr_start > 1:
                labels[curr_start + 1 : curr_end] = input_ids_t[curr_start + 1 : curr_end]

        # Label first token of each segment after EOC
        for k, seg_start in enumerate(segment_starts):
            eoc_idx = 2 + (self.num_v + 1) * k + self.num_v
            curr_raw_start = raw_phys_start + seg_start
            if curr_raw_start < self.valid_len and eoc_idx + 1 < labels.size(-1):
                labels[eoc_idx + 1] = input_ids_t[curr_raw_start]

        return labels

    def build_gen_attention_and_pos(
        self, *, seq_len: int, device: torch.device
    ) -> tuple[torch.BoolTensor, torch.LongTensor]:
        """Build attention mask and position IDs."""
        num_segments = self.num_segments
        segment_starts = self._get_segment_starts()
        segment_lens = self._segment_lens

        raw_phys_start = self.prefix_len

        # Initialize mask
        mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

        # BOS can see itself
        mask[0, 0] = True
        # SOC can see BOS and itself
        mask[1, :2] = True

        # Control groups (V tokens and EOC)
        all_v_indices: list[int] = []
        for k in range(num_segments):
            group_base = 2 + (self.num_v + 1) * k
            v_indices = list(range(group_base, group_base + self.num_v))
            eoc_idx = group_base + self.num_v

            for v_idx in v_indices:
                mask[v_idx, :2] = True  # See BOS, SOC
                if all_v_indices:
                    mask[v_idx, all_v_indices] = True  # See previous V tokens
                mask[v_idx, v_idx] = True  # See itself
                all_v_indices.append(v_idx)

            mask[eoc_idx, :2] = True
            if all_v_indices:
                mask[eoc_idx, all_v_indices] = True
            mask[eoc_idx, eoc_idx] = True

        # Raw content segments
        for i in range(num_segments):
            c_start = raw_phys_start + segment_starts[i]
            c_end = min(c_start + segment_lens[i], self.valid_len)
            c_len = c_end - c_start
            if c_len <= 0:
                continue

            # Causal within segment
            chunk_causal = torch.tril(torch.ones((c_len, c_len), dtype=torch.bool, device=device))
            mask[c_start:c_end, c_start:c_end] = chunk_causal

            # See BOS, SOC
            mask[c_start:c_end, :2] = True
            # See first V group
            mask[c_start:c_end, 2 : 2 + self.num_v] = True

            # Buffer logic
            split_idx = i - self.buffer_size

            if split_idx > 0:
                # See V tokens up to split_idx
                limit_idx = split_idx
                for j in range(limit_idx + 1):
                    group_base = 2 + (self.num_v + 1) * j
                    for v_off in range(self.num_v):
                        mask[c_start:c_end, group_base + v_off] = True
                # See last EOC
                last_eoc_idx = 2 + (self.num_v + 1) * limit_idx + self.num_v
                mask[c_start:c_end, last_eoc_idx] = True
            else:
                # See first EOC
                mask[c_start:c_end, 2 + self.num_v] = True

            # See previous buffer segments (raw tokens)
            start_buffer_idx = max(0, split_idx)
            for j in range(start_buffer_idx, i):
                prev_c_start = raw_phys_start + segment_starts[j]
                prev_c_end = min(prev_c_start + segment_lens[j], self.valid_len)
                if prev_c_end > prev_c_start:
                    mask[c_start:c_end, prev_c_start:prev_c_end] = True

        # Position IDs from mask
        pos_ids = mask.long().sum(dim=-1) - 1
        pos_ids = pos_ids.clamp(min=0)

        return mask, pos_ids


class ZipperBuilder(ZipperBuilderBase):
    """Fixed chunking builder - splits by constant chunk_size."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        *,
        chunk_size: int,
        buffer_size: int = 0,
        num_v: int = 1,
        random_gate: float = 0,
        truncate_len: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
        self.chunk_size = int(chunk_size)
        super().__init__(
            tokenizer, prompt,
            buffer_size=buffer_size, num_v=num_v,
            random_gate=random_gate, truncate_len=truncate_len,
        )

    def _compute_segments(self) -> None:
        """Compute fixed-size segments."""
        num_chunks = self.raw_len // self.chunk_size
        left_over = self.raw_len % self.chunk_size

        # Segment lengths
        self._segment_lens = [self.chunk_size] * num_chunks
        if left_over > 0:
            self._segment_lens.append(left_over)

        # Compressor chunks (all except last segment)
        if left_over == 0:
            self.num_cmp_chunks = num_chunks - 1
        else:
            self.num_cmp_chunks = num_chunks

        full_part_len = num_chunks * self.chunk_size
        full_part = self.raw_core_ids[:full_part_len]
        raw_chunks = [full_part[i : i + self.chunk_size] for i in range(0, full_part_len, self.chunk_size)]
        self.cmp_wrapped_chunks = [list(c) for c in raw_chunks[: self.num_cmp_chunks]]


class AdaptiveZipperBuilder(ZipperBuilderBase):
    """Adaptive chunking builder - splits by variable raw_chunk_lens."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        *,
        raw_chunk_lens: Sequence[int],
        buffer_size: int = 0,
        num_v: int = 1,
        random_gate: float = 0,
        truncate_len: int | None = None,
    ) -> None:
        self.raw_chunk_lens = [int(x) for x in raw_chunk_lens]
        super().__init__(
            tokenizer, prompt,
            buffer_size=buffer_size, num_v=num_v,
            random_gate=random_gate, truncate_len=truncate_len,
        )

    def _compute_segments(self) -> None:
        """Compute variable-size segments from raw_chunk_lens."""
        # Validate chunk_lens sum
        if sum(self.raw_chunk_lens) != self.raw_len:
            raise ValueError(
                f"raw_chunk_lens sum mismatch: sum={sum(self.raw_chunk_lens)} != raw_len={self.raw_len}"
            )

        # Segment lengths directly from input
        self._segment_lens = self.raw_chunk_lens

        # Split raw tokens by chunk_lens
        raw_chunks = self._split_by_lens(self.raw_core_ids, self.raw_chunk_lens)
        
        # Compressor chunks (all except last segment)
        self.num_cmp_chunks = len(raw_chunks) - 1
        self.cmp_wrapped_chunks = [list(c) for c in raw_chunks[: self.num_cmp_chunks]]

    @staticmethod
    def _split_by_lens(token_ids: list[int], chunk_lens: Sequence[int]) -> list[list[int]]:
        """Split token_ids by variable lengths."""
        out: list[list[int]] = []
        cur = 0
        for ln in chunk_lens:
            ln_i = int(ln)
            if ln_i <= 0:
                raise ValueError(f"chunk_lens must be > 0, got {chunk_lens}")
            out.append(token_ids[cur : cur + ln_i])
            cur += ln_i
        return out
