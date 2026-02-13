from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    PreTrainedModel,
)

from .configuration_iron_cell import IronCellConfig


@dataclass
class CompressedMemory:
    """Compressed vectors V and their zipper positions."""

    vectors: torch.Tensor  # [B, C, H]
    positions: torch.LongTensor  # [B, C] positions in zipper sequence


def _init_projector_(linear: nn.Linear, init_type: str, std_noise: float = 1e-3) -> None:
    if init_type == "identity":
        if linear.in_features != linear.out_features:
            raise ValueError(
                f"identity init requires in_features == out_features, got {linear.in_features} vs {linear.out_features}"
            )
        nn.init.eye_(linear.weight)
        with torch.no_grad():
            linear.weight.add_(torch.randn_like(linear.weight) * std_noise)
    elif init_type == "gaussian":
        nn.init.normal_(linear.weight, mean=0.0, std=0.02)
    else:
        raise ValueError(f"Unknown projector_init_type: {init_type}")


def _to_4d_additive_mask(attn_2d: torch.BoolTensor, dtype: torch.dtype) -> torch.Tensor:
    """
    Convert a boolean [B,S,S] mask (True=allowed) into an additive [B,1,S,S] mask
    (0 for allowed, -inf for blocked).
    """
    if attn_2d.dim() != 3:
        raise ValueError(f"Expected [B,S,S] mask, got shape {tuple(attn_2d.shape)}")
    neg_inf = torch.finfo(dtype).min
    additive = torch.where(attn_2d, torch.zeros((), device=attn_2d.device, dtype=dtype), torch.tensor(neg_inf, device=attn_2d.device, dtype=dtype))
    return additive.unsqueeze(1)


class IronCellModel(PreTrainedModel):
    """
    Iron-Cell core model (Compressor + Projector + Generator).

    Design goal:
        - The heavy preprocessing happens outside the model.
        - The model can accept:
            * pre-built `inputs_embeds`
            * a custom attention mask (2D/4D)
        - Convenience helpers are provided to compute compressor vectors and
          inject them into a zipper-layout embedding sequence.
    """

    config_class = IronCellConfig

    def __init__(self, config: IronCellConfig) -> None:
        super().__init__(config)

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

        target_vocab = int(getattr(config, "tokenizer_vocab_size", 0))
        if target_vocab > 0:
            gen_vocab = int(self.generator.get_input_embeddings().weight.size(0))
            comp_vocab = int(self.compressor.get_input_embeddings().weight.size(0))
            if target_vocab > gen_vocab:
                self.generator.resize_token_embeddings(target_vocab, mean_resizing=False)
            if target_vocab > comp_vocab:
                self.compressor.resize_token_embeddings(target_vocab, mean_resizing=False)

        comp_h = int(getattr(self.compressor.config, "hidden_size"))
        gen_h = int(getattr(self.generator.config, "hidden_size"))
        gen_dtype = self.generator.get_input_embeddings().weight.dtype

        self.projector = nn.Linear(comp_h, gen_h, bias=False).to(dtype=gen_dtype)
        _init_projector_(self.projector, config.projector_init_type)

        self.register_buffer("special_token_ids", torch.empty((0,), dtype=torch.long), persistent=True)
        cfg_special_ids = getattr(config, "special_token_ids", None)
        if isinstance(cfg_special_ids, list) and len(cfg_special_ids) > 0:
            self.special_token_ids = torch.tensor([int(i) for i in cfg_special_ids], dtype=torch.long)
            self.special_token_embeddings = nn.Embedding(len(cfg_special_ids), gen_h).to(dtype=gen_dtype)
        else:
            self.special_token_embeddings = nn.Embedding(0, gen_h).to(dtype=gen_dtype)
        self.special_token_embeddings.weight.requires_grad = False

        if config.freeze_compressor:
            for p in self.compressor.parameters():
                p.requires_grad = False

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        return next(self.parameters()).device

    def freeze_for_phase_1(self) -> None:
        """
        Freeze strategy (Phase-1 MVP):

        - Freeze compressor fully (already handled by config.freeze_compressor).
        - Freeze generator backbone (transformer blocks).
        - Unfreeze projector.
        - Optionally unfreeze generator embeddings (default for MVP).
        """
        for p in self.projector.parameters():
            p.requires_grad = True

        for name, p in self.generator.named_parameters():
            p.requires_grad = False
            if ".embed_tokens." in name or name.endswith("embed_tokens.weight"):
                if "embed_tokens" in self.config.trainable_components:
                    p.requires_grad = True

    def enable_special_token_training(self, token_ids: list[int], *, init_from_generator: bool = True) -> None:
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
        chunk_input_ids: torch.LongTensor,  # [B,C,L]
        chunk_attention_mask: torch.LongTensor,  # [B,C,L]
    ) -> torch.Tensor:
        """
        Run compressor and projector to obtain compressed vectors V.

        Returns:
            vectors: [B, C, H_generator]
        """
        bsz, num_chunks, chunk_len = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(bsz * num_chunks, chunk_len).to(self.device)
        flat_mask = chunk_attention_mask.view(bsz * num_chunks, chunk_len).to(self.device)

        do_no_grad = bool(self.config.freeze_compressor)
        with torch.no_grad() if do_no_grad else torch.enable_grad():
            outputs = self.compressor(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = outputs.last_hidden_state  # [B*C, L, Hc]

        last_index = (flat_mask.sum(dim=1) - 1).clamp(min=0)  # [B*C]
        gather_index = last_index.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        pooled = hidden.gather(dim=1, index=gather_index).squeeze(1)  # [B*C, Hc]

        projected = self.projector(pooled)  # [B*C, Hg]
        return projected.view(bsz, num_chunks, -1)

    def build_inputs_embeds(
        self,
        *,
        zipper_input_ids: torch.LongTensor,  # [B,S]
        memory_vectors: torch.Tensor,  # [B,C,H]
        memory_positions: torch.LongTensor,  # [B,C] (-1 padded)
    ) -> torch.Tensor:
        """
        Build `inputs_embeds` for the generator by injecting V into zipper positions.
        """
        zipper_input_ids = zipper_input_ids.to(self.device)
        memory_vectors = memory_vectors.to(self.device)
        memory_positions = memory_positions.to(self.device)

        embed = self.generator.get_input_embeddings()
        inputs_embeds = embed(zipper_input_ids)

        if self.special_token_ids.numel() > 0:
            special_ids = self.special_token_ids.to(device=self.device)
            for i in range(int(special_ids.numel())):
                tid = int(special_ids[i].item())
                mask = zipper_input_ids == tid
                if mask.any():
                    inputs_embeds[mask] = self.special_token_embeddings.weight[i].to(inputs_embeds.dtype)

        valid = memory_positions >= 0
        if valid.any():
            b_idx, c_idx = torch.where(valid)
            pos = memory_positions[b_idx, c_idx]
            inputs_embeds[b_idx, pos] = memory_vectors[b_idx, c_idx]

        return inputs_embeds

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
        Forward for training/inference.

        Args:
            inputs_embeds: pre-built embedding sequence (zipper layout).
            attention_mask: custom mask; supports:
                - [B,S,S] boolean (True=allowed)
                - [B,1,S,S] additive float mask (0 / -inf)
                - [B,S] padding mask (fallback to model's causal mask)
            position_ids: manual RoPE position ids (recommended for zipper geometry).
            labels: language modeling labels; typically prefix positions are -100.
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
