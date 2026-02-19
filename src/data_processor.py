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
        teacher_attn_targets: [B, T] teacher attention targets (optional).
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


def _chunk_list(token_ids: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    return [token_ids[i : i + chunk_size] for i in range(0, len(token_ids), chunk_size)]


def build_zipper_mask_posid(
    *,
    seq_len: int,
    valid_len: int,
    num_chunks: int,
    chunk_size: int,
    left_over: int = 0,
    device: torch.device,
    buffer_size: int = 0,
    num_v: int = 1,
) -> tuple[torch.BoolTensor, torch.LongTensor]:
    
    # ==========================================
    # 1. 统一逻辑：Raw 段数 = Control 组数
    # ==========================================
    # 如果有 leftover，它就是第 N+1 个段
    num_raw_segments = num_chunks + (1 if left_over > 0 else 0)
    num_control_groups = num_raw_segments
    
    # 计算 Prefix 长度
    # Prefix: BOS + SOC + (v + EOC) * num_control_groups
    # 这里的 num_control_groups 包含了初始组 (-1 组) 到 倒数第二组
    # 刚好对应每一个 Raw Segment
    prefix_len = 1 + 1 + (num_v + 1) * num_control_groups
    
    raw_phys_start = prefix_len
    
    # 校验长度 (非常重要，防止 Input 拼错了但 Mask 没报错)
    expected_valid_len = prefix_len + num_chunks * chunk_size + left_over
    assert valid_len == expected_valid_len, \
        f"Geometry mismatch! Valid:{valid_len} vs Calc:{expected_valid_len}. Check Input Assembly."

    mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    
    # ==========================================
    # 2. Prefix 内部逻辑 (Control Chain)
    # ==========================================
    
    # 2.1 BOS & SOC
    mask[0, 0] = True      # BOS 看 BOS
    mask[1, :2] = True     # SOC 看 BOS, SOC
    
    # 2.2 v 和 EOC (因果链)
    all_v_indices: list[int] = []
    for k in range(num_control_groups):
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
    
    # ==========================================
    # 3. Raw Chunk 逻辑 (Hybrid Attention)
    # ==========================================
    
    for i in range(num_raw_segments):
        # 计算当前 Raw Chunk 的物理范围
        c_start = raw_phys_start + i * chunk_size
        c_end = min(c_start + chunk_size, valid_len)
        
        # 实际长度 (最后一段可能是 leftover)
        c_len = c_end - c_start
        if c_len <= 0: continue # 防御性代码

        # A. Chunk 内部自回归 (Autoregressive)
        chunk_causal = torch.tril(torch.ones((c_len, c_len), dtype=torch.bool, device=device))
        mask[c_start:c_end, c_start:c_end] = chunk_causal
        
        # B. 历史判定 (Split Logic)
        # split_idx 表示从哪里开始是 Buffer (保留高清原图)
        # 比 split_idx 小的都是 Compressed (只看 v)
        split_idx = i - buffer_size
        
        # --- 压缩部分 (Compressed History) ---
        if split_idx >= 0:
            # 1. BOS, SOC
            mask[c_start:c_end, :2] = True
            
            # 2. v 序列 (直到 split_idx)
            # 注意：如果 Raw_i 是 Compressed，那么 Raw_i 对应的 v_i 就是 summary。
            # 当前 chunk 应该看之前的 chunks 对应的 v。
            # 第 k 个 chunk 对应的 v 是 Group k-1 (如果 Group -1 算 0 号的话...)
            # 让我们理一下索引：
            # Raw 0 看 v_-1 (Group 0)
            # Raw 1 看 v_-1, v_0 (Group 0, 1)
            # 所以 Raw i 应该看 Group 0 到 Group i (inclusive, 因为 v_i 还没生成呢? 不，v_i 是 Raw i 的总结)
            # 正确逻辑: Raw i 只能看 Raw 0..i-1 的总结。
            # Raw 0 对应的总结是 v_0 (在 Group 1, Idx 4)。
            # 初始状态是 v_-1 (在 Group 0, Idx 2)。
            
            # 这里的 split_idx 是 chunk index。
            # 我们需要看到 Group 0 到 Group split_idx 的 v。
            limit_idx = split_idx
            group_bases = 2 + (num_v + 1) * torch.arange(limit_idx + 1, device=device)
            v_offsets = torch.arange(num_v, device=device)
            v_indices = (group_bases.unsqueeze(1) + v_offsets.unsqueeze(0)).reshape(-1)
            mask[c_start:c_end, v_indices] = True
            
            # 3. Trigger EOC (仅最后一个被压缩块的 EOC，作为桥梁)
            # 也就是 Group split_idx 的 EOC
            last_eoc_idx = 2 + (num_v + 1) * limit_idx + num_v
            mask[c_start:c_end, last_eoc_idx] = True
            
        # --- 缓冲部分 (Buffered History) ---
        start_buffer_idx = max(0, split_idx)
        for j in range(start_buffer_idx, i):
            prev_c_start = raw_phys_start + j * chunk_size
            prev_c_end = min(prev_c_start + chunk_size, valid_len)
            
            if prev_c_end > prev_c_start:
                mask[c_start:c_end, prev_c_start:prev_c_end] = True

    # ==========================
    # 4. RoPE IDs
    # ==========================
    pos_ids = mask.long().sum(dim=-1) - 1
    pos_ids = pos_ids.clamp(min=0)

    return mask, pos_ids


def build_zipper_labels(
    input_ids: torch.Tensor,
    valid_len: int,
    num_chunks: int,
    chunk_size: int,
    left_over: int,
    num_v: int = 1,
    ignore_index: int = -100,
) -> torch.Tensor:
    num_raw_segments = num_chunks + (1 if left_over > 0 else 0)
    num_control_groups = num_raw_segments

    prefix_len = 2 + (num_v + 1) * num_control_groups
    raw_content_len = (num_chunks * chunk_size) + left_over
    expected_valid_len = prefix_len + raw_content_len

    assert valid_len == expected_valid_len, (
        f"Geometry Error! Valid:{valid_len} vs Calc:{expected_valid_len}. "
        f"(Chunks:{num_chunks}, Size:{chunk_size}, Left:{left_over} -> "
        f"Segments:{num_raw_segments}, Prefix:{prefix_len})"
    )

    raw_phys_start = prefix_len
    labels = torch.full_like(input_ids, ignore_index)

    for k in range(num_raw_segments):
        curr_start = raw_phys_start + k * chunk_size
        curr_end = min(curr_start + chunk_size, valid_len)
        chunk_len = curr_end - curr_start

        if chunk_len > 1:
            labels[..., curr_start + 1 : curr_end] = input_ids[..., curr_start + 1 : curr_end]

    for k in range(num_control_groups):
        eoc_idx = 2 + (num_v + 1) * k + num_v
        curr_raw_start = raw_phys_start + k * chunk_size

        if curr_raw_start < valid_len:
            if eoc_idx + 1 < labels.size(-1):
                labels[..., eoc_idx + 1] = input_ids[..., curr_raw_start]

    return labels


build_zipper_attn_mask_and_pos_ids = build_zipper_mask_posid


class ZipperBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        *,
        chunk_size: int,
        buffer_size: int = 0,
        num_v: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.chunk_size = int(chunk_size)
        self.buffer_size = int(buffer_size)
        self.num_v = int(num_v)

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")

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

        self.raw_core_ids = tokenizer.encode(prompt, add_special_tokens=False)
        self.raw_len = len(self.raw_core_ids)

        self.num_chunks = self.raw_len // self.chunk_size
        self.left_over = self.raw_len % self.chunk_size

        if self.left_over == 0:
            self.num_cmp_chunks = self.num_chunks - 1
        else:
            self.num_cmp_chunks = self.num_chunks

        if self.num_cmp_chunks < 1:
            raise ValueError(
                "Invalid sample for compression: num_cmp_chunks must be >= 1. "
                f"raw_len={self.raw_len}, chunk_size={self.chunk_size}, "
                f"num_chunks={self.num_chunks}, left_over={self.left_over}"
            )

        self.cmp_chunk_len = self.chunk_size + 2
        full_part_len = self.num_chunks * self.chunk_size
        full_part = self.raw_core_ids[:full_part_len]
        raw_chunks = _chunk_list(full_part, self.chunk_size)
        raw_chunks = raw_chunks[: self.num_cmp_chunks]
        self.cmp_wrapped_chunks = [[*c] for c in raw_chunks]

        num_control_groups = self.num_chunks + (1 if self.left_over > 0 else 0)
        prefix_ids: list[int] = [self.bos_id, self.soc_id]
        for _ in range(num_control_groups):
            for _ in range(self.num_v):
                prefix_ids.append(self.v_none_id)
            prefix_ids.append(self.eoc_id)
        self.prefix_ids = prefix_ids
        self.prefix_len = len(prefix_ids)

        self.gen_input_ids = self.prefix_ids + self.raw_core_ids
        self.valid_len = len(self.gen_input_ids)
        
        # WARNING: This assumes prefix is exactly [BOS, SOC] (len 2). 
        # k starts from 1 to skip v_-1 (at index 2), aligning with CMP outputs (v_0, v_1...).
        self.memory_positions = [
            2 + (self.num_v + 1) * k for k in range(1, self.num_cmp_chunks + 1)
        ]

    def build_gen_labels(self, device: torch.device) -> torch.LongTensor:
        input_ids_t = torch.tensor(self.gen_input_ids, dtype=torch.long, device=device)
        return build_zipper_labels(
            input_ids=input_ids_t,
            valid_len=self.valid_len,
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size,
            left_over=self.left_over,
            num_v=self.num_v,
        )

    def build_gen_attention_and_pos(
        self, *, seq_len: int, device: torch.device
    ) -> tuple[torch.BoolTensor, torch.LongTensor]:
        return build_zipper_mask_posid(
            seq_len=seq_len,
            valid_len=self.valid_len,
            num_chunks=self.num_chunks,
            chunk_size=self.chunk_size,
            left_over=self.left_over,
            device=device,
            buffer_size=self.buffer_size,
            num_v=self.num_v,
        )

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
        chunk_size: int = 16,
        pad_to_multiple_of: int | None = 8,
        buffer_size: int = 0,
        num_v: int = 1,
        teacher_targets: torch.Tensor | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.pad_to_multiple_of = pad_to_multiple_of
        self.buffer_size = int(buffer_size)
        self.num_v = int(num_v)
        self.teacher_targets = teacher_targets

    def __call__(self, items: Sequence[object]) -> ZipperBatch:
        if len(items) == 0:
            raise ValueError("Empty batch")

        device = torch.device("cpu")
        pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if isinstance(items[0], tuple):
            texts, idxs = zip(*items)
            idxs_t = torch.tensor(idxs, dtype=torch.long, device=device)
        else:
            texts = items
            idxs_t = None

        builders: list[ZipperBuilder] = [
            ZipperBuilder(
                self.tokenizer,
                text,
                chunk_size=self.chunk_size,
                buffer_size=self.buffer_size,
                num_v=self.num_v,
            )
            for text in texts
        ]

        max_zip_len = max(b.valid_len for b in builders)
        max_num_cmp_chunks = max(b.num_cmp_chunks for b in builders)
        cmp_chunk_len = self.chunk_size

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
            (len(texts), max_num_cmp_chunks, cmp_chunk_len),
            pad_id,
            dtype=torch.long,
            device=device,
        )
        chunk_attention_mask = torch.zeros(
            (len(texts), max_num_cmp_chunks, cmp_chunk_len),
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

        for b, builder in enumerate(builders):
            valid_len = builder.valid_len
            prefix_len = builder.prefix_len

            zipper_input_ids[b, :valid_len] = torch.tensor(builder.gen_input_ids, dtype=torch.long, device=device)

            labels_b = builder.build_gen_labels(device=device)
            labels[b, :valid_len] = labels_b
            labels[b, valid_len:] = -100

            prefix_lens[b] = prefix_len
            valid_lens[b] = valid_len

            memory_positions[b, : builder.num_cmp_chunks] = torch.tensor(
                builder.memory_positions, dtype=torch.long, device=device
            )

            for i, c in enumerate(builder.cmp_wrapped_chunks):
                chunk_input_ids[b, i, :] = torch.tensor(c, dtype=torch.long, device=device)
                chunk_attention_mask[b, i, :] = 1

            attn, pos_ids = builder.build_gen_attention_and_pos(seq_len=max_zip_len, device=device)
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
            teacher_attn_targets=teacher_attn_targets,
        )
