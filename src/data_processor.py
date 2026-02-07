from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import PreTrainedTokenizerBase


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


def _chunk_list(token_ids: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def _build_prefix_ids(
    soc_id: int,
    eoc_id: int,
    num_chunks: int,
    placeholder_id: int,
) -> tuple[list[int], list[int]]:
    prefix_ids: list[int] = [soc_id]
    memory_positions: list[int] = []
    for _ in range(num_chunks):
        memory_positions.append(len(prefix_ids))
        prefix_ids.append(placeholder_id)
        prefix_ids.append(eoc_id)
    return prefix_ids, memory_positions


def build_zipper_geometry(
    *,
    seq_len: int,
    valid_len: int,
    num_chunks: int,
    chunk_size: int,
    device: torch.device,
) -> tuple[torch.BoolTensor, torch.LongTensor]:
    """
    Build Iron-Cell zipper geometry (Sieve Mask + Manual RoPE position ids) for one sample.

    Physical layout:
        [SOC] [V1] [EOC1] [V2] [EOC2] ... [VN] [EOCN] || [Raw_Chunk_1] ... [Raw_Chunk_N]

    Sieve mask rules (code.md spec):
        - Prefix region (SOC/V/EOC): standard causal mask.
        - Raw_Chunk_k:
            * must see: SOC, V1..Vk, and ONLY its own EOC_k
            * must NOT see: any other EOC (past or future), any future V, any other raw chunk
            * must see: its own chunk internal causal history

    Manual RoPE position_ids:
        pos(SOC)=0
        pos(V_k)=2k-1, pos(EOC_k)=2k
        pos(Raw_k_start)=pos(EOC_k)+1=2k+1, then increments within chunk.
    """
    if valid_len > seq_len:
        raise ValueError(f"valid_len ({valid_len}) cannot exceed seq_len ({seq_len})")
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    prefix_len = 1 + 2 * num_chunks
    if prefix_len > valid_len:
        raise ValueError(f"prefix_len ({prefix_len}) cannot exceed valid_len ({valid_len})")

    q = torch.arange(seq_len, device=device).view(-1, 1)
    k = torch.arange(seq_len, device=device).view(1, -1)

    valid_q = q < valid_len
    valid_k = k < valid_len
    valid = valid_q & valid_k

    is_prefix_q = q < prefix_len
    is_raw_q = (~is_prefix_q) & valid_q

    causal = k <= q
    prefix_allowed = valid & is_prefix_q & causal

    raw_qpos = (q - prefix_len).clamp(min=0)
    q_chunk = (raw_qpos // chunk_size) + 1
    q_chunk = torch.where(is_raw_q, q_chunk, torch.zeros_like(q_chunk))

    is_prefix_k = (k < prefix_len) & valid_k
    is_raw_k = (k >= prefix_len) & valid_k

    raw_kpos = (k - prefix_len).clamp(min=0)
    k_chunk = (raw_kpos // chunk_size) + 1
    k_chunk = torch.where(is_raw_k, k_chunk, torch.zeros_like(k_chunk))

    same_raw_chunk = is_raw_k & (k_chunk == q_chunk)
    raw_internal_causal = same_raw_chunk & causal

    soc_key = k == 0
    is_v_key = is_prefix_k & (k % 2 == 1)
    v_index = (k + 1) // 2
    v_allowed = is_v_key & (v_index <= q_chunk)

    is_eoc_key = is_prefix_k & (k % 2 == 0) & (k > 0)
    eoc_index = k // 2
    eoc_allowed = is_eoc_key & (eoc_index == q_chunk)

    prefix_allowed_for_raw = is_prefix_k & (soc_key | v_allowed | eoc_allowed)
    raw_allowed = valid & is_raw_q & (raw_internal_causal | prefix_allowed_for_raw)

    attention_mask_2d = prefix_allowed | raw_allowed

    position_ids = torch.zeros((seq_len,), dtype=torch.long, device=device)
    valid_pos = torch.arange(seq_len, device=device) < valid_len
    pos = torch.arange(seq_len, device=device, dtype=torch.long)

    prefix_pos = pos < prefix_len
    position_ids = torch.where(valid_pos & prefix_pos, pos, position_ids)

    raw_pos = valid_pos & (~prefix_pos)
    raw_offset = (pos - prefix_len).clamp(min=0)
    raw_chunk_idx = (raw_offset // chunk_size) + 1
    raw_within = raw_offset - (raw_chunk_idx - 1) * chunk_size
    raw_base = 2 * raw_chunk_idx + 1
    raw_position_ids = raw_base + raw_within
    position_ids = torch.where(raw_pos, raw_position_ids, position_ids)

    return attention_mask_2d, position_ids


def build_staircase_mask(
    *,
    seq_len: int,
    valid_len: int,
    prefix_len: int,
    num_chunks: int,
    chunk_size: int,
    device: torch.device,
) -> torch.BoolTensor:
    """
    Build the 2D staircase attention mask for a single sample.

    Mask rule (MVP):
        - For all positions: causal within the whole sequence (k <= q).
        - For raw tokens: additionally block attending to future compression anchors:
          raw tokens in chunk N can only attend to {SOC, V1..VN, EOC1..EOCN}.

    Args:
        seq_len: padded sequence length (S).
        valid_len: unpadded sequence length.
        prefix_len: SOC + (V,EOC)*C.
        num_chunks: number of chunks (C).
        chunk_size: tokens per chunk for the raw text split.
        device: torch device.
    """
    if valid_len > seq_len:
        raise ValueError(f"valid_len ({valid_len}) cannot exceed seq_len ({seq_len})")
    if prefix_len > valid_len:
        raise ValueError(f"prefix_len ({prefix_len}) cannot exceed valid_len ({valid_len})")
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")

    q = torch.arange(seq_len, device=device).view(-1, 1)
    k = torch.arange(seq_len, device=device).view(1, -1)

    valid_q = q < valid_len
    valid_k = k < valid_len
    causal = (k <= q) & valid_q & valid_k

    query_is_raw = (q >= prefix_len) & valid_q
    key_is_prefix = (k < prefix_len) & valid_k

    raw_pos = (q - prefix_len).clamp(min=0)
    query_chunk_index = (raw_pos // chunk_size) + 1
    query_chunk_index = torch.where(query_is_raw, query_chunk_index, torch.zeros_like(query_chunk_index))

    key_pos = k
    key_mem_index = torch.where(
        key_pos == 0,
        torch.zeros_like(key_pos),
        (key_pos + 1) // 2,
    )
    key_mem_index = torch.where(key_is_prefix, key_mem_index, torch.zeros_like(key_mem_index))

    block_future_mem = query_is_raw & key_is_prefix & (key_mem_index > query_chunk_index)
    allowed = causal & (~block_future_mem)
    return allowed


class IronCellCollator:
    """
    Build zipper layout + staircase mask for Iron-Cell masked parallel training.

    This collator is intentionally "heavy": it performs tokenization, chunking,
    zipper construction, attention-mask construction, and label construction.
    The model itself only receives embeddings + attention mask.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        chunk_size: int = 256,
        pad_to_multiple_of: int | None = 8,
        soc_token: str = "<soc>",
        eoc_token: str = "<eoc>",
    ) -> None:
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.soc_id = int(tokenizer.convert_tokens_to_ids(soc_token))
        self.eoc_id = int(tokenizer.convert_tokens_to_ids(eoc_token))
        if self.soc_id < 0 or self.eoc_id < 0:
            raise ValueError("Special tokens not found in tokenizer; call add_iron_cell_special_tokens first.")

    def __call__(self, texts: Sequence[str]) -> ZipperBatch:
        if len(texts) == 0:
            raise ValueError("Empty batch")

        device = torch.device("cpu")
        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else 0

        per_sample = []
        max_zip_len = 0
        max_num_chunks = 0
        max_chunk_len = 0

        for text in texts:
            raw_ids = self.tokenizer.encode(text, add_special_tokens=False)
            if len(raw_ids) == 0:
                raw_ids = [pad_id]

            chunks = _chunk_list(raw_ids, self.chunk_size)
            num_chunks = len(chunks)

            prefix_ids, mem_pos = _build_prefix_ids(
                soc_id=self.soc_id,
                eoc_id=self.eoc_id,
                num_chunks=num_chunks,
                placeholder_id=self.soc_id,
            )
            zipper_ids = prefix_ids + raw_ids
            prefix_len = len(prefix_ids)
            valid_len = len(zipper_ids)

            max_zip_len = max(max_zip_len, valid_len)
            max_num_chunks = max(max_num_chunks, num_chunks)
            max_chunk_len = max(max_chunk_len, max(len(c) for c in chunks))

            per_sample.append(
                {
                    "zipper_ids": zipper_ids,
                    "prefix_len": prefix_len,
                    "valid_len": valid_len,
                    "mem_pos": mem_pos,
                    "chunks": chunks,
                }
            )

        if self.pad_to_multiple_of is not None and max_zip_len % self.pad_to_multiple_of != 0:
            m = self.pad_to_multiple_of
            max_zip_len = ((max_zip_len + m - 1) // m) * m

        zipper_input_ids = torch.full((len(texts), max_zip_len), pad_id, dtype=torch.long, device=device)
        labels = torch.full((len(texts), max_zip_len), -100, dtype=torch.long, device=device)
        attention_mask_2d = torch.zeros((len(texts), max_zip_len, max_zip_len), dtype=torch.bool, device=device)
        position_ids = torch.zeros((len(texts), max_zip_len), dtype=torch.long, device=device)
        prefix_lens = torch.zeros((len(texts),), dtype=torch.long, device=device)
        valid_lens = torch.zeros((len(texts),), dtype=torch.long, device=device)

        chunk_input_ids = torch.full(
            (len(texts), max_num_chunks, max_chunk_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        chunk_attention_mask = torch.zeros(
            (len(texts), max_num_chunks, max_chunk_len),
            dtype=torch.long,
            device=device,
        )
        memory_positions = torch.full(
            (len(texts), max_num_chunks),
            -1,
            dtype=torch.long,
            device=device,
        )

        for b, item in enumerate(per_sample):
            ids = item["zipper_ids"]
            prefix_len = int(item["prefix_len"])
            valid_len = int(item["valid_len"])
            mem_pos = item["mem_pos"]
            chunks = item["chunks"]

            zipper_input_ids[b, :valid_len] = torch.tensor(ids, dtype=torch.long)

            labels[b, :valid_len] = zipper_input_ids[b, :valid_len]
            labels[b, :prefix_len] = -100
            labels[b, valid_len:] = -100

            prefix_lens[b] = prefix_len
            valid_lens[b] = valid_len

            memory_positions[b, : len(mem_pos)] = torch.tensor(mem_pos, dtype=torch.long)

            for i, c in enumerate(chunks):
                c_len = len(c)
                chunk_input_ids[b, i, :c_len] = torch.tensor(c, dtype=torch.long)
                chunk_attention_mask[b, i, :c_len] = 1

            attn, pos_ids = build_zipper_geometry(
                seq_len=max_zip_len,
                valid_len=valid_len,
                num_chunks=len(chunks),
                chunk_size=self.chunk_size,
                device=device,
            )
            attention_mask_2d[b] = attn
            position_ids[b] = pos_ids

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
        )

