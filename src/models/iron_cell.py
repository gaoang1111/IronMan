from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)

from .config import IronCellConfig
from .javis import Javis


def _to_4d_additive_mask(attn_2d: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a boolean [B,S,S] mask (True=allowed) into an additive [B,1,S,S] mask
    (0 for allowed, -inf for blocked).
    """
    if attn_2d.dim() != 3:
        raise ValueError(f"Expected [B,S,S] mask, got shape {tuple(attn_2d.shape)}")
    neg_inf = torch.finfo(dtype).min
    additive = torch.where(
        attn_2d,
        torch.zeros((), device=attn_2d.device, dtype=dtype),
        torch.tensor(neg_inf, device=attn_2d.device, dtype=dtype),
    )
    return additive.unsqueeze(1)


class IronCellModel(PreTrainedModel):
    """
    Iron-Cell core model (Compressor + Javis + Generator).

    Architecture:
        - Compressor: Frozen LLaMA model that extracts hidden states from input chunks
        - Javis: Cross-attention module that compresses chunk representations
        - Generator: LLaMA model that generates output using compressed memory

    Design:
        - Heavy preprocessing (zipper layout, staircase mask) happens outside the model
        - Model accepts pre-built inputs_embeds and custom attention masks
        - Convenience helpers provided for compression and embedding injection
    """

    config_class = IronCellConfig

    def __init__(self, config: IronCellConfig) -> None:
        super().__init__(config)

        # Load compressor and generator from pretrained
        self.compressor = AutoModel.from_pretrained(
            config.compressor_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )
        self.generator = AutoModelForCausalLM.from_pretrained(
            config.generator_model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=False,
        )

        # Resize embeddings if needed
        target_vocab = int(getattr(config, "tokenizer_vocab_size", 0))
        if target_vocab > 0:
            gen_vocab = int(self.generator.get_input_embeddings().weight.size(0))
            comp_vocab = int(self.compressor.get_input_embeddings().weight.size(0))
            if target_vocab > gen_vocab:
                self.generator.resize_token_embeddings(target_vocab, mean_resizing=False)
            if target_vocab > comp_vocab:
                self.compressor.resize_token_embeddings(target_vocab, mean_resizing=False)

        # Initialize Javis
        comp_h = int(getattr(self.compressor.config, "hidden_size"))
        gen_h = int(getattr(self.generator.config, "hidden_size"))
        gen_dtype = self.generator.get_input_embeddings().weight.dtype

        num_heads = int(getattr(config, "javis_num_heads", 16))
        num_queries = int(getattr(config, "javis_num_queries", 1))
        query_group_size = int(getattr(config, "javis_query_group_size", 1))

        ln_in_enabled = bool(getattr(config, "javis_ln_in", True))
        ln_out_enabled = bool(getattr(config, "javis_ln_out", True))
        init_noise_std = float(getattr(config, "javis_init_noise_std", 0.01))
        
        self.javis = Javis(
            input_dim=comp_h,
            hidden_size=gen_h,
            num_heads=num_heads,
            num_queries=num_queries,
            query_group_size=query_group_size,
            ln_in_enabled=ln_in_enabled,
            ln_out_enabled=ln_out_enabled,
            init_noise_std=init_noise_std,
            dtype=gen_dtype,
        )

        # Special token embeddings (trainable)
        self.register_buffer("special_token_ids", torch.empty((0,), dtype=torch.long), persistent=True)
        cfg_special_ids = getattr(config, "special_token_ids", None)
        if isinstance(cfg_special_ids, list) and len(cfg_special_ids) > 0:
            self.special_token_ids = torch.tensor([int(i) for i in cfg_special_ids], dtype=torch.long)
            self.special_token_embeddings = nn.Embedding(len(cfg_special_ids), gen_h).to(dtype=gen_dtype)
        else:
            self.special_token_embeddings = nn.Embedding(0, gen_h).to(dtype=gen_dtype)
        self.special_token_embeddings.weight.requires_grad = False

        # Freeze compressor if configured
        if config.freeze_compressor:
            for p in self.compressor.parameters():
                p.requires_grad = False

    def _load_from_state_dict(  # type: ignore[override]
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Handle legacy checkpoint compatibility (projector -> javis rename)."""
        projector_prefix = f"{prefix}projector."
        if any(k.startswith(projector_prefix) for k in state_dict.keys()):
            to_add: dict[str, torch.Tensor] = {}
            to_del: list[str] = []
            for k, v in state_dict.items():
                if k.startswith(projector_prefix):
                    to_del.append(k)
                    new_k = f"{prefix}javis.{k[len(projector_prefix):]}"
                    if new_k not in state_dict:
                        to_add[new_k] = v
            for k in to_del:
                state_dict.pop(k, None)
            state_dict.update(to_add)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return next(self.parameters()).device

    def freeze_for_phase_1(self) -> None:
        """
        Freeze strategy for Phase-1 (warmup):
        - Freeze compressor fully (already handled by config.freeze_compressor)
        - Freeze generator backbone (transformer blocks)
        - Unfreeze Javis
        - Optionally unfreeze generator embeddings
        """
        for p in self.javis.parameters():
            p.requires_grad = True

        for name, p in self.generator.named_parameters():
            p.requires_grad = False
            if ".embed_tokens." in name or name.endswith("embed_tokens.weight"):
                if "embed_tokens" in self.config.trainable_components:
                    p.requires_grad = True

    def enable_special_token_training(self, token_ids: list[int], *, init_from_generator: bool = True) -> None:
        """
        Enable training for special tokens (<soc>, <eoc>, <v_none>).
        
        Args:
            token_ids: List of special token IDs
            init_from_generator: Initialize from generator embeddings
        """
        unique_ids = []
        seen = set()
        for tid in token_ids:
            tid_i = int(tid)
            if tid_i < 0 or tid_i in seen:
                continue
            seen.add(tid_i)
            unique_ids.append(tid_i)

        gen_h = int(getattr(self.generator.config, "hidden_size"))
        base_embed = self.generator.get_input_embeddings()
        base_vocab = int(base_embed.weight.size(0))

        filtered_ids = [tid for tid in unique_ids if 0 <= tid < base_vocab]
        if len(filtered_ids) == 0:
            raise ValueError(
                "No valid special token ids found for generator embedding table. "
                f"Got token_ids={unique_ids}, generator_vocab={base_vocab}."
            )

        ids_t = torch.tensor(filtered_ids, dtype=torch.long, device=self.device)
        self.special_token_ids = ids_t

        gen_dtype = base_embed.weight.dtype
        self.special_token_embeddings = nn.Embedding(len(filtered_ids), gen_h).to(device=self.device, dtype=gen_dtype)
        self.special_token_embeddings.weight.requires_grad = True

        if init_from_generator:
            with torch.no_grad():
                self.special_token_embeddings.weight.copy_(base_embed.weight.index_select(0, ids_t))

        self.config.special_token_ids = filtered_ids

    def compute_compressed_vectors(
        self,
        *,
        chunk_input_ids: torch.LongTensor,
        chunk_attention_mask: torch.LongTensor,
        return_metrics: bool = False,
    ):
        """
        Run compressor and Javis to obtain compressed vectors and deep KV.

        Args:
            chunk_input_ids: [B, C, L] - Token IDs for each chunk
            chunk_attention_mask: [B, C, L] - Attention mask for chunks
            return_metrics: Whether to return diagnostic metrics

        Returns:
            If return_metrics=False:
                (javis_out_full, memory_vectors, deep_layer_kvs, cos_similarity)
            If return_metrics=True:
                (javis_out_full, memory_vectors, deep_layer_kvs, metrics, cos_similarity)
                
            Where:
                - javis_out_full: [B*C, G, Q, H] - Full Javis output
                - memory_vectors: [B, C, Q, H] - Vectors for embedding injection (Group 0)
                - deep_layer_kvs: List of 32 (K, V) pairs for deep injection
        """
        bsz, num_chunks, chunk_len = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(bsz * num_chunks, chunk_len).to(self.device)
        flat_mask = chunk_attention_mask.view(bsz * num_chunks, chunk_len).to(self.device)

        # Run compressor (frozen or not based on config)
        do_no_grad = bool(self.config.freeze_compressor)
        with torch.no_grad() if do_no_grad else torch.enable_grad():
            outputs = self.compressor(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = outputs.last_hidden_state  # [B*C, L, Hc]

        # Run Javis
        if return_metrics:
            javis_out_flat, javis_metrics, current_out_cos = self.javis(hidden, attention_mask=flat_mask, return_metrics=True)
        else:
            javis_out_flat, current_out_cos = self.javis(hidden, attention_mask=flat_mask, return_metrics=False)
            javis_metrics = None

        # Get deep KV for all 32 layers
        layer_kvs = self.javis.get_all_layer_kv(javis_out_flat)
        
        # Reshape layer KVs: [B*C, ...] -> [B, C, ...]
        reshaped_layer_kvs = []
        for l in range(self.javis.num_layers):
            k_flat, v_flat = layer_kvs[l]
            k_res = k_flat.view(bsz, num_chunks, self.javis.num_kv_heads, self.javis.num_queries, self.javis.head_dim)
            v_res = v_flat.view(bsz, num_chunks, self.javis.num_kv_heads, self.javis.num_queries, self.javis.head_dim)
            reshaped_layer_kvs.append((k_res, v_res))

        # Extract Group 0 for inputs_embeds injection
        javis_out_group0 = javis_out_flat[:, 0, :, :]  # [B*C, Q, H]
        javis_vecs = javis_out_group0.view(bsz, num_chunks, self.javis.num_queries, -1)

        if return_metrics:
            return javis_out_flat, javis_vecs, reshaped_layer_kvs, javis_metrics, current_out_cos
        return javis_out_flat, javis_vecs, reshaped_layer_kvs, current_out_cos

    def build_inputs_embeds(
        self,
        *,
        zipper_input_ids: torch.LongTensor,  # [B, S]
        memory_vectors: torch.Tensor,         # [B, C, Q, H]
        memory_positions: torch.LongTensor,   # [B, C] (-1 for padding)
    ) -> torch.Tensor:
        """
        Build inputs_embeds for the generator by injecting memory vectors.

        Args:
            zipper_input_ids: [B, S] - Token IDs in zipper layout
            memory_vectors: [B, C, Q, H] - Compressed vectors to inject
            memory_positions: [B, C] - Positions where to inject (-1 = skip)

        Returns:
            inputs_embeds: [B, S, H] - Embeddings with injected memory vectors
        """
        zipper_input_ids = zipper_input_ids.to(self.device)
        memory_vectors = memory_vectors.to(self.device)
        memory_positions = memory_positions.to(self.device)

        embed = self.generator.get_input_embeddings()
        inputs_embeds = embed(zipper_input_ids)

        # Replace special token embeddings with trainable versions
        if self.special_token_ids.numel() > 0:
            special_ids = self.special_token_ids.to(device=self.device)
            for i in range(int(special_ids.numel())):
                tid = int(special_ids[i].item())
                mask = zipper_input_ids == tid
                if mask.any():
                    inputs_embeds[mask] = self.special_token_embeddings.weight[i].to(inputs_embeds.dtype)

        # Inject memory vectors at specified positions
        valid = memory_positions >= 0
        if valid.any():
            b_idx, c_idx = torch.where(valid)
            start_pos = memory_positions[b_idx, c_idx]
            assert (start_pos + self.javis.num_queries <= inputs_embeds.size(1)).all()
            for q_i in range(self.javis.num_queries):
                inputs_embeds[b_idx, start_pos + q_i] = memory_vectors[b_idx, c_idx, q_i].to(inputs_embeds.dtype)

        return inputs_embeds

    def build_student_attn_mask(
        self,
        *,
        seq_len: int,
        memory_positions: torch.LongTensor,
        prefix_lens: torch.LongTensor,
        valid_lens: torch.LongTensor,
        chunk_size: int,
    ) -> torch.BoolTensor:
        """Build attention mask for student (raw tokens attending to memory slots)."""
        memory_positions = memory_positions.to(self.device)
        prefix_lens = prefix_lens.to(self.device)
        valid_lens = valid_lens.to(self.device)
        bsz = int(memory_positions.size(0))
        seq_len = int(seq_len)
        num_v = int(self.javis.num_queries)

        p_idx = torch.arange(seq_len, device=self.device).view(1, -1).expand(bsz, -1)
        raw_pos = p_idx - prefix_lens.view(-1, 1)
        raw_mask = (p_idx >= prefix_lens.view(-1, 1)) & (p_idx < valid_lens.view(-1, 1))
        chunk_idx = torch.where(raw_pos >= 0, raw_pos // int(chunk_size), torch.zeros_like(raw_pos))

        base = torch.full_like(chunk_idx, -1)
        base = torch.where((chunk_idx == 0) & raw_mask, torch.full_like(base, 2), base)
        if memory_positions.numel() > 0:
            mem_idx = (chunk_idx - 1).clamp(min=0)
            mem_base = memory_positions.gather(1, mem_idx)
            base = torch.where((chunk_idx > 0) & raw_mask, mem_base, base)

        mask = torch.zeros((bsz, seq_len, seq_len), dtype=torch.bool, device=self.device)
        v_offsets = torch.arange(num_v, device=self.device).view(1, 1, -1)
        col_pos = base.unsqueeze(-1) + v_offsets
        valid_col = raw_mask.unsqueeze(-1) & (col_pos >= 0) & (col_pos < seq_len)
        b_idx = torch.arange(bsz, device=self.device).view(bsz, 1, 1).expand_as(col_pos)
        row_idx = torch.arange(seq_len, device=self.device).view(1, seq_len, 1).expand_as(col_pos)
        mask[b_idx[valid_col], row_idx[valid_col], col_pos[valid_col]] = True

        eoc_pos = base + num_v
        valid_eoc = raw_mask & (eoc_pos >= 0) & (eoc_pos < seq_len)
        b_idx2 = torch.arange(bsz, device=self.device).view(bsz, 1).expand_as(eoc_pos)
        row_idx2 = torch.arange(seq_len, device=self.device).view(1, seq_len).expand_as(eoc_pos)
        mask[b_idx2[valid_eoc], row_idx2[valid_eoc], eoc_pos[valid_eoc]] = True

        return mask

    def forward(  # type: ignore[override]
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        **kwargs: Any,
    ):
        """
        Forward pass for training/inference.

        Args:
            inputs_embeds: [B, S, H] - Pre-built embedding sequence (zipper layout)
            attention_mask: Custom mask, supports:
                - [B, S, S] boolean (True=allowed)
                - [B, 1, S, S] additive float mask (0 / -inf)
                - [B, S] padding mask (fallback to model's causal mask)
            position_ids: [B, S] - Manual RoPE position IDs for zipper geometry
            labels: [B, S] - LM labels (prefix positions should be -100)

        Returns:
            CausalLMOutputWithPast from generator
        """
        if attention_mask is not None and attention_mask.dim() == 3 and attention_mask.dtype == torch.bool:
            attention_mask = _to_4d_additive_mask(attention_mask, dtype=inputs_embeds.dtype)

        return self.generator(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )
