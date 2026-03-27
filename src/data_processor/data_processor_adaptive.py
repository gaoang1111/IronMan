from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Iterable

import torch
from transformers import PreTrainedTokenizerBase

from .fixed import ZipperBatch


def _split_by_lens(token_ids: list[int], chunk_lens: Sequence[int]) -> list[list[int]]:
    out: list[list[int]] = []
    cur = 0
    n = len(token_ids)
    for ln in chunk_lens:
        ln_i = int(ln)
        if ln_i <= 0:
            raise ValueError(f"chunk_lens must be > 0, got {chunk_lens}")
        nxt = cur + ln_i
        if nxt > n:
            raise ValueError(f"chunk_lens exceeds token length: sum={nxt} > n={n}")
        out.append(token_ids[cur:nxt])
        cur = nxt
    if cur != n:
        raise ValueError(f"chunk_lens sum mismatch: sum={cur} != n={n}")
    return out


def _cumsum_starts(chunk_lens: Sequence[int], *, device: torch.device) -> torch.LongTensor:
    lens = torch.tensor([int(x) for x in chunk_lens], dtype=torch.long, device=device)
    if int(lens.numel()) == 0:
        return torch.zeros((0,), dtype=torch.long, device=device)
    if (lens <= 0).any():
        raise ValueError(f"chunk_lens must be > 0, got {chunk_lens}")
    starts = torch.zeros_like(lens)
    if int(lens.numel()) > 1:
        starts[1:] = torch.cumsum(lens[:-1], dim=0)
    return starts


def build_zipper_mask_posid_adaptive(
    *,
    seq_len: int,
    valid_len: int,
    raw_chunk_lens: Sequence[int],
    device: torch.device,
    buffer_size: int = 0,
    num_v: int = 1,
) -> tuple[torch.BoolTensor, torch.LongTensor]:
    num_raw_segments = int(len(raw_chunk_lens))
    if num_raw_segments <= 0:
        raise ValueError("raw_chunk_lens must be non-empty.")

    prefix_len = 1 + 1 + (num_v + 1) * num_raw_segments
    raw_phys_start = prefix_len
    raw_total = int(sum(int(x) for x in raw_chunk_lens))
    expected_valid_len = prefix_len + raw_total
    assert int(valid_len) == int(expected_valid_len), (
        f"Geometry mismatch! valid_len={valid_len} vs expected={expected_valid_len}."
    )

    mask = torch.zeros((int(seq_len), int(seq_len)), dtype=torch.bool, device=device)

    mask[0, 0] = True
    mask[1, :2] = True

    all_v_indices: list[int] = []
    for k in range(num_raw_segments):
        group_base = 2 + (num_v + 1) * k
        v_indices = group_base + torch.arange(num_v, device=device)
        eoc_idx = group_base + num_v

        for v_idx in v_indices.tolist():
            mask[v_idx, :2] = True
            if all_v_indices:
                mask[v_idx, torch.tensor(all_v_indices, device=device)] = True
            mask[v_idx, v_idx] = True
            all_v_indices.append(v_idx)

        mask[eoc_idx, :2] = True
        if all_v_indices:
            mask[eoc_idx, torch.tensor(all_v_indices, device=device)] = True
        mask[eoc_idx, eoc_idx] = True

    starts = _cumsum_starts(raw_chunk_lens, device=device)
    lens = torch.tensor([int(x) for x in raw_chunk_lens], dtype=torch.long, device=device)

    for i in range(num_raw_segments):
        c_start = int(raw_phys_start + int(starts[i].item()))
        c_end = int(c_start + int(lens[i].item()))

        if c_end <= c_start:
            continue

        c_len = c_end - c_start
        chunk_causal = torch.tril(torch.ones((c_len, c_len), dtype=torch.bool, device=device))
        mask[c_start:c_end, c_start:c_end] = chunk_causal

        mask[c_start:c_end, :2] = True
        mask[c_start:c_end, 2 : 2 + num_v] = True

        split_idx = int(i - int(buffer_size))

        if split_idx > 0:
            limit_idx = split_idx
            group_bases = 2 + (num_v + 1) * torch.arange(limit_idx + 1, device=device)
            v_offsets = torch.arange(num_v, device=device)
            v_indices = (group_bases.unsqueeze(1) + v_offsets.unsqueeze(0)).reshape(-1)
            mask[c_start:c_end, v_indices] = True

            last_eoc_idx = 2 + (num_v + 1) * limit_idx + num_v
            mask[c_start:c_end, int(last_eoc_idx)] = True
        else:
            mask[c_start:c_end, 2 + num_v] = True

        start_buffer_idx = max(0, split_idx)
        for j in range(int(start_buffer_idx), int(i)):
            prev_c_start = int(raw_phys_start + int(starts[j].item()))
            prev_c_end = int(prev_c_start + int(lens[j].item()))
            if prev_c_end > prev_c_start:
                mask[c_start:c_end, prev_c_start:prev_c_end] = True

    pos_ids = mask.long().sum(dim=-1) - 1
    pos_ids = pos_ids.clamp(min=0)
    return mask, pos_ids


def build_zipper_labels_adaptive(
    *,
    input_ids: torch.Tensor,
    valid_len: int,
    raw_chunk_lens: Sequence[int],
    num_v: int = 1,
    ignore_index: int = -100,
    random_gate: float = 0,
) -> torch.Tensor:
    num_raw_segments = int(len(raw_chunk_lens))
    if num_raw_segments <= 0:
        raise ValueError("raw_chunk_lens must be non-empty.")

    prefix_len = 2 + (num_v + 1) * num_raw_segments
    raw_total = int(sum(int(x) for x in raw_chunk_lens))
    expected_valid_len = int(prefix_len + raw_total)
    assert int(valid_len) == int(expected_valid_len), (
        f"Geometry Error! valid_len={valid_len} vs expected={expected_valid_len}."
    )

    labels = torch.full_like(input_ids, ignore_index)
    raw_phys_start = int(prefix_len)

    import random

    visible_cap = 14 if random.random() < float(random_gate) else None

    starts = _cumsum_starts(raw_chunk_lens, device=input_ids.device)
    lens = torch.tensor([int(x) for x in raw_chunk_lens], dtype=torch.long, device=input_ids.device)

    for k in range(num_raw_segments):
        curr_start = int(raw_phys_start + int(starts[k].item()))
        curr_end = int(curr_start + int(lens[k].item()))
        if visible_cap is not None:
            curr_end = int(min(curr_end, curr_start + int(visible_cap)))

        chunk_len = curr_end - curr_start
        if chunk_len > 1:
            labels[..., curr_start + 1 : curr_end] = input_ids[..., curr_start + 1 : curr_end]

    for k in range(num_raw_segments):
        eoc_idx = 2 + (num_v + 1) * k + num_v
        curr_raw_start = int(raw_phys_start + int(starts[k].item()))
        if curr_raw_start < int(valid_len):
            if int(eoc_idx + 1) < int(labels.size(-1)):
                labels[..., eoc_idx + 1] = input_ids[..., curr_raw_start]

    return labels


@dataclass
class AdaptiveZipperBuilder:
    tokenizer: PreTrainedTokenizerBase
    prompt: str
    raw_chunk_lens: Sequence[int]
    buffer_size: int = 0
    num_v: int = 1
    random_gate: float = 0
    truncate_len: int | None = None

    def __post_init__(self) -> None:
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        if bos_id is None or eos_id is None:
            raise ValueError("Tokenizer must define both bos_token_id and eos_token_id.")
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.soc_id = int(self.tokenizer.convert_tokens_to_ids("<soc>"))
        self.eoc_id = int(self.tokenizer.convert_tokens_to_ids("<eoc>"))
        self.v_none_id = int(self.tokenizer.convert_tokens_to_ids("<v_none>"))
        if self.soc_id < 0 or self.eoc_id < 0 or self.v_none_id < 0:
            raise ValueError("Special tokens not found in tokenizer; call add_iron_cell_special_tokens first.")

        raw_core_ids = self.tokenizer.encode(
            self.prompt, add_special_tokens=False, max_length=self.truncate_len, truncation=True
        )
        self.raw_core_ids = raw_core_ids
        self.raw_len = int(len(raw_core_ids))

        if self.raw_len <= 0:
            raise ValueError("Empty tokenized prompt.")

        chunk_lens = [int(x) for x in self.raw_chunk_lens]
        if int(sum(chunk_lens)) != int(self.raw_len):
            raise ValueError(
                f"raw_chunk_lens sum mismatch: sum={sum(chunk_lens)} != raw_len={self.raw_len}"
            )

        self.raw_chunks = _split_by_lens(self.raw_core_ids, chunk_lens)
        self.num_raw_segments = int(len(self.raw_chunks))
        if self.num_raw_segments < 2:
            raise ValueError("Invalid sample for compression: need at least 2 segments.")

        self.num_cmp_chunks = int(self.num_raw_segments - 1)
        self.cmp_wrapped_chunks = [list(c) for c in self.raw_chunks[: self.num_cmp_chunks]]

        prefix_ids: list[int] = [self.bos_id, self.soc_id]
        for _ in range(self.num_raw_segments):
            for _ in range(int(self.num_v)):
                prefix_ids.append(self.v_none_id)
            prefix_ids.append(self.eoc_id)
        self.prefix_ids = prefix_ids
        self.prefix_len = int(len(prefix_ids))

        self.gen_input_ids = self.prefix_ids + self.raw_core_ids
        self.valid_len = int(len(self.gen_input_ids))

        self.memory_positions = [2 + (int(self.num_v) + 1) * k for k in range(1, self.num_cmp_chunks + 1)]

    def build_gen_labels(self, *, device: torch.device) -> torch.LongTensor:
        input_ids_t = torch.tensor(self.gen_input_ids, dtype=torch.long, device=device)
        return build_zipper_labels_adaptive(
            input_ids=input_ids_t,
            valid_len=self.valid_len,
            raw_chunk_lens=self.raw_chunk_lens,
            num_v=int(self.num_v),
            random_gate=float(self.random_gate),
        )

    def build_gen_attention_and_pos(
        self, *, seq_len: int, device: torch.device
    ) -> tuple[torch.BoolTensor, torch.LongTensor]:
        return build_zipper_mask_posid_adaptive(
            seq_len=int(seq_len),
            valid_len=self.valid_len,
            raw_chunk_lens=self.raw_chunk_lens,
            device=device,
            buffer_size=int(self.buffer_size),
            num_v=int(self.num_v),
        )


def _extract_text_idx_lens(item: object) -> tuple[str, int | None, Sequence[int]]:
    if isinstance(item, dict):
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("Adaptive collator expects item['text'] as str.")
        idx = item.get("idx")
        idx_i = int(idx) if idx is not None else None
        lens = item.get("chunk_lens", item.get("raw_chunk_lens"))
        if lens is None:
            raise ValueError("Adaptive collator expects 'chunk_lens' or 'raw_chunk_lens'.")
        return text, idx_i, lens

    if isinstance(item, tuple):
        if len(item) == 3:
            text, idx, lens = item
            return str(text), int(idx), lens
        if len(item) == 2:
            text, idx = item
            raise ValueError("Adaptive collator requires chunk_lens in dataset items.")

    raise ValueError("Unsupported item type for adaptive collator.")


class AdaptiveIronCellCollator:
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

    def __call__(self, items: Sequence[object]) -> ZipperBatch:
        if len(items) == 0:
            raise ValueError("Empty batch")

        device = torch.device("cpu")
        pad_id = (
            int(self.tokenizer.pad_token_id)
            if self.tokenizer.pad_token_id is not None
            else int(self.tokenizer.eos_token_id)
        )

        texts: list[str] = []
        idxs: list[int] = []
        lens_list: list[Sequence[int]] = []
        for it in items:
            text, idx, lens = _extract_text_idx_lens(it)
            texts.append(text)
            if idx is not None:
                idxs.append(int(idx))
            lens_list.append(lens)

        idxs_t = torch.tensor(idxs, dtype=torch.long, device=device) if len(idxs) == len(texts) else None

        builders: list[AdaptiveZipperBuilder] = [
            AdaptiveZipperBuilder(
                self.tokenizer,
                text,
                raw_chunk_lens=lens,
                buffer_size=self.buffer_size,
                num_v=self.num_v,
                random_gate=self.random_gate,
                truncate_len=self.truncate_len,
            )
            for text, lens in zip(texts, lens_list)
        ]

        max_zip_len = max(b.valid_len for b in builders)
        max_num_cmp_chunks = max(b.num_cmp_chunks for b in builders)
        max_cmp_chunk_len = 1
        for b in builders:
            for c in b.cmp_wrapped_chunks:
                max_cmp_chunk_len = max(max_cmp_chunk_len, int(len(c)))

        if self.pad_to_multiple_of is not None and max_zip_len % int(self.pad_to_multiple_of) != 0:
            m = int(self.pad_to_multiple_of)
            max_zip_len = ((max_zip_len + m - 1) // m) * m

        zipper_input_ids = torch.full((len(texts), max_zip_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(texts), max_zip_len), -100, dtype=torch.long, device=device)
        attention_mask_2d = torch.zeros((len(texts), max_zip_len, max_zip_len), dtype=torch.bool, device=device)
        position_ids = torch.zeros((len(texts), max_zip_len), dtype=torch.long, device=device)
        prefix_lens = torch.zeros((len(texts),), dtype=torch.long, device=device)
        valid_lens = torch.zeros((len(texts),), dtype=torch.long, device=device)

        chunk_input_ids = torch.full(
            (len(texts), max_num_cmp_chunks, max_cmp_chunk_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        chunk_attention_mask = torch.zeros(
            (len(texts), max_num_cmp_chunks, max_cmp_chunk_len),
            dtype=torch.long,
            device=device,
        )
        memory_positions = torch.full(
            (len(texts), max_num_cmp_chunks),
            -1,
            dtype=torch.long,
            device=device,
        )

        if self.teacher_targets is not None:
            if idxs_t is None:
                raise ValueError("teacher_targets requires dataset indices.")
            teacher_attn_targets = self.teacher_targets.index_select(0, idxs_t).to(dtype=torch.float32)
        else:
            teacher_attn_targets = torch.zeros((len(texts), 0), dtype=torch.float32, device=device)

        if self.teacher_hidden_targets is not None:
            if idxs_t is None:
                raise ValueError("teacher_hidden_targets requires dataset indices.")
            if self.teacher_hidden_valid_lens is None:
                raise ValueError("teacher_hidden_valid_lens is required when teacher_hidden_targets is set.")
            teacher_hidden_targets = self.teacher_hidden_targets.index_select(0, idxs_t).to(dtype=torch.float32)
            valid_v_lens = self.teacher_hidden_valid_lens.index_select(0, idxs_t).to(dtype=torch.long)
        else:
            teacher_hidden_targets = torch.zeros((len(texts), 0, 0), dtype=torch.float32, device=device)
            valid_v_lens = torch.zeros((len(texts),), dtype=torch.long, device=device)

        teacher_hidden_target_layer = torch.tensor(
            -1 if self.teacher_hidden_target_layer is None else int(self.teacher_hidden_target_layer),
            dtype=torch.long,
            device=device,
        )

        for b_idx, builder in enumerate(builders):
            valid_len = int(builder.valid_len)
            prefix_len = int(builder.prefix_len)

            zipper_input_ids[b_idx, :valid_len] = torch.tensor(builder.gen_input_ids, dtype=torch.long, device=device)

            labels_b = builder.build_gen_labels(device=device)
            labels[b_idx, :valid_len] = labels_b
            labels[b_idx, valid_len:] = -100

            prefix_lens[b_idx] = prefix_len
            valid_lens[b_idx] = valid_len

            memory_positions[b_idx, : builder.num_cmp_chunks] = torch.tensor(
                builder.memory_positions, dtype=torch.long, device=device
            )

            for i, c in enumerate(builder.cmp_wrapped_chunks):
                ln = int(len(c))
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
