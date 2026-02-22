from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def _init_eye_plus_noise_(linear: nn.Linear, *, std_noise: float) -> None:
    if linear.in_features != linear.out_features:
        raise ValueError(
            f"eye+noise init requires in_features == out_features, got {linear.in_features} vs {linear.out_features}"
        )
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype))
        linear.weight.add_(torch.randn_like(linear.weight) * float(std_noise))


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


class Javis(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_heads: int,
        num_queries: int,
        num_layers: int = 32,      # LLaMA-3 default
        num_kv_heads: int = 8,     # LLaMA-3 8B GQA default
        head_dim: int = 128,       # LLaMA-3 default
        ln_in_enabled: bool,
        ln_out_enabled: bool,
        init_noise_std: float,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_queries = int(num_queries)
        
        self.num_layers = int(num_layers)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        
        self.ln_in_enabled = bool(ln_in_enabled)
        self.ln_out_enabled = bool(ln_out_enabled)
        self.init_noise_std = float(init_noise_std)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads, got {self.hidden_size} % {self.num_heads}")
        if self.num_queries <= 0:
            raise ValueError(f"num_queries must be >= 1, got {self.num_queries}")

        self.in_proj: nn.Module
        if int(input_dim) != self.hidden_size:
            self.in_proj = nn.Linear(int(input_dim), self.hidden_size, bias=False).to(dtype=dtype)
            nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        else:
            self.in_proj = nn.Identity()

        self.ln_in = nn.LayerNorm(self.hidden_size, elementwise_affine=True).to(dtype=dtype)
        self.ln_out = nn.LayerNorm(self.hidden_size, elementwise_affine=True).to(dtype=dtype)

        self.q = nn.Parameter(torch.empty((self.num_queries, self.hidden_size), dtype=dtype))
        nn.init.normal_(self.q, mean=0.0, std=1.0)

        self.wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype=dtype)
        self.wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype=dtype)
        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype=dtype)

        _init_eye_plus_noise_(self.wk, std_noise=self.init_noise_std)
        _init_eye_plus_noise_(self.wv, std_noise=self.init_noise_std)
        _init_eye_plus_noise_(self.wo, std_noise=self.init_noise_std)
        self.current_q_grad_cos = None


        # ==========================================
        # Deep KV Injection Protocol (Giant Matrix)
        # Target Dim: 32 layers * 2 (K,V) * 8 heads * 128 dim = 65536
        # ==========================================
        self.kv_dim_per_layer = self.num_kv_heads * self.head_dim
        self.total_kv_proj_dim = self.num_layers * 2 * self.kv_dim_per_layer
        
        self.kv_proj = nn.Linear(self.hidden_size, self.total_kv_proj_dim, bias=False).to(dtype=dtype)
        
        std_dev = 1.0 / math.sqrt(self.hidden_size)  # 约 0.015625
        nn.init.normal_(self.kv_proj.weight, mean=0.0, std=std_dev)

    def get_all_layer_kv(self, v_out: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Transforms the Javis 2-token output into LLaMA-3.1 strict past_key_values format.
        Args:
            v_out: Tensor of shape [B, num_queries, hidden_size]
        Returns:
            Tuple of length `num_layers`.
            Each element is a tuple (K_tensor, V_tensor).
            K/V shape: [B, num_kv_heads, num_queries, head_dim]
        """
        B, Q, _ = v_out.shape
        
        # [B, Q, 65536]
        flat_kv = self.kv_proj(v_out)
        
        # Reshape: [B, Q, layers, 2(K/V), kv_heads, head_dim]
        flat_kv = flat_kv.view(B, Q, self.num_layers, 2, self.num_kv_heads, self.head_dim)
        
        # Permute to isolate layers: [layers, 2, B, kv_heads, Q, head_dim]
        flat_kv = flat_kv.permute(2, 3, 0, 4, 1, 5)
        
        past_key_values = []
        for l in range(self.num_layers):
            k = flat_kv[l, 0] # [B, num_kv_heads, Q, head_dim]
            v = flat_kv[l, 1] # [B, num_kv_heads, Q, head_dim]
            past_key_values.append((k, v))
            
        return tuple(past_key_values)
    
    def pre_kv_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(hidden)
        if self.ln_in_enabled:
            x = self.ln_in(x)
        return x

    def forward(
        self,
        hidden: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        return_metrics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, float], torch.Tensor]:
        x = self.pre_kv_hidden(hidden)

        k = self.wk(x)
        v = self.wv(x)

        n = int(x.size(0))
        q_tensor = self.q.unsqueeze(0).expand(n, -1, -1)
        if self.training and return_metrics and q_tensor.requires_grad:
            def _q_grad_hook(grad: torch.Tensor) -> None:
                grad_q0 = grad[:, 0, :].float()
                grad_q1 = grad[:, 1, :].float()
                cos_sim = F.cosine_similarity(grad_q0, grad_q1, dim=-1, eps=1e-8).mean()
                self.current_q_grad_cos = float(cos_sim.item())
            q_tensor.register_hook(_q_grad_hook)
        q = q_tensor

        q_len = int(q.size(1))
        seq_len = int(k.size(1))
        h = self.num_heads
        d_k = self.hidden_size // h

        q = q.view(n, q_len, h, d_k).transpose(1, 2)  # [N,h,Q,d]
        k = k.view(n, seq_len, h, d_k).transpose(1, 2)  # [N,h,L,d]
        v = v.view(n, seq_len, h, d_k).transpose(1, 2)  # [N,h,L,d]

        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [N,h,Q,L]
        if attention_mask is not None:
            m = attention_mask.to(dtype=torch.bool, device=logits.device).view(n, 1, 1, seq_len)
            logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        ctx = torch.matmul(attn, v)  # [N,h,Q,d]
        ctx = ctx.transpose(1, 2).contiguous().view(n, q_len, self.hidden_size)  # [N,Q,H]
        
        out = self.wo(ctx)
        if self.ln_out_enabled:
            out = self.ln_out(out)
        assert out.dim() == 3 and out.size(1) == self.num_queries
        current_out_cos = torch.zeros((), device=out.device, dtype=out.dtype)
        if out.size(1) >= 2:
            current_out_cos = F.cosine_similarity(out[:, 0, :], out[:, 1, :], dim=-1, eps=1e-8).mean()

        # B, S, D = hidden.shape
        # shortcut = hidden.view(B, self.num_queries, S // self.num_queries, D).mean(dim=2)
        
        # return out + shortcut

        global_mean = hidden.mean(dim=1, keepdim=True) # [B, 1, D]
        shortcut = global_mean.expand(-1, self.num_queries, -1) # [B, 2, D]
        metrics: dict[str, float] | None = None
        if return_metrics:
            metrics = {}
            with torch.no_grad():
                out_detached = out.detach().float()
                mean_detached = global_mean.detach().float()
                out_pair = out_detached[:, :2, :]
                norm_out = out_pair.norm(p=2, dim=-1).mean()
                norm_mean = mean_detached.norm(p=2, dim=-1).mean()
                metrics["javis_norm_ratio"] = float((norm_out / (norm_mean + 1e-9)).item())
                metrics["javis_out_cos"] = float(current_out_cos.detach().item())
                token_len = min(16, seq_len)
                if token_len > 0 and out_pair.size(1) >= 2:
                    attn_detached = attn.detach().float()
                    attn_q0 = attn_detached[:, :, 0, :token_len].mean(dim=1)
                    attn_q1 = attn_detached[:, :, 1, :token_len].mean(dim=1)
                    kl_div = F.kl_div((attn_q1 + 1e-9).log(), attn_q0, reduction="batchmean")
                    metrics["javis_attn_kl"] = float(kl_div.item())

        final_out = out + shortcut*0.5
        if return_metrics:
            return final_out, metrics, current_out_cos
        return final_out, current_out_cos


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
            # attn_implementation="eager",
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

        num_heads = int(getattr(config, "javis_num_heads", 16))
        num_queries = int(getattr(config, "javis_num_queries", 1))
        ln_in_enabled = bool(getattr(config, "javis_ln_in", True))
        ln_out_enabled = bool(getattr(config, "javis_ln_out", True))
        init_noise_std = float(getattr(config, "javis_init_noise_std", 0.01))
        self.javis = Javis(
            input_dim=comp_h,
            hidden_size=gen_h,
            num_heads=num_heads,
            num_queries=num_queries,
            ln_in_enabled=ln_in_enabled,
            ln_out_enabled=ln_out_enabled,
            init_noise_std=init_noise_std,
            dtype=gen_dtype,
        )

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
        Freeze strategy (Phase-1 MVP):

        - Freeze compressor fully (already handled by config.freeze_compressor).
        - Freeze generator backbone (transformer blocks).
        - Unfreeze Javis.
        - Optionally unfreeze generator embeddings (default for MVP).
        """
        for p in self.javis.parameters():
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

    # def compute_compressed_vectors(
    #     self,
    #     *,
    #     chunk_input_ids: torch.LongTensor,  # [B,C,L]
    #     chunk_attention_mask: torch.LongTensor,  # [B,C,L]
    #     return_metrics: bool = False,
    # ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    #     """
    #     Run compressor and projector to obtain compressed vectors V.

    #     Returns:
    #         vectors: [B, C, H_generator]
    #     """
    #     bsz, num_chunks, chunk_len = chunk_input_ids.shape
    #     flat_ids = chunk_input_ids.view(bsz * num_chunks, chunk_len).to(self.device)
    #     flat_mask = chunk_attention_mask.view(bsz * num_chunks, chunk_len).to(self.device)

    #     do_no_grad = bool(self.config.freeze_compressor)
    #     with torch.no_grad() if do_no_grad else torch.enable_grad():
    #         outputs = self.compressor(input_ids=flat_ids, attention_mask=flat_mask)
    #         hidden = outputs.last_hidden_state  # [B*C, L, Hc]

    #     javis_out = self.javis(hidden, attention_mask=flat_mask, return_metrics=return_metrics)  # [B*C, Q, Hg]
    #     if return_metrics:
    #         javis_vecs, javis_metrics, current_out_cos = javis_out
    #         return javis_vecs.view(bsz, num_chunks, self.javis.num_queries, -1), javis_metrics, current_out_cos
    #     javis_vecs, current_out_cos = javis_out
    #     return javis_vecs.view(bsz, num_chunks, self.javis.num_queries, -1), current_out_cos

    def compute_compressed_vectors(
        self,
        *,
        chunk_input_ids: torch.LongTensor,
        chunk_attention_mask: torch.LongTensor,
        return_metrics: bool = False,
    ):
        bsz, num_chunks, chunk_len = chunk_input_ids.shape
        flat_ids = chunk_input_ids.view(bsz * num_chunks, chunk_len).to(self.device)
        flat_mask = chunk_attention_mask.view(bsz * num_chunks, chunk_len).to(self.device)

        do_no_grad = bool(self.config.freeze_compressor)
        with torch.no_grad() if do_no_grad else torch.enable_grad():
            outputs = self.compressor(input_ids=flat_ids, attention_mask=flat_mask)
            hidden = outputs.last_hidden_state  # [B*C, L, Hc]

        # javis_out_flat: [B*C, Q, Hg]
        if return_metrics:
            javis_out_flat, javis_metrics, current_out_cos = self.javis(hidden, attention_mask=flat_mask, return_metrics=True)
        else:
            javis_out_flat, current_out_cos = self.javis(hidden, attention_mask=flat_mask, return_metrics=False)
            javis_metrics = None

        # ==========================================
        # 🔥 获取 32 层 Deep KV (核心改动)
        # ==========================================
        layer_kvs = self.javis.get_all_layer_kv(javis_out_flat) 
        # layer_kvs: 32 elements of (K, V)
        # K/V shape: [B*C, num_kv_heads, Q, head_dim]
        
        # 将 flat 的 K/V reshape 回 [B, C, ... ] 以便后续切片覆写
        reshaped_layer_kvs = []
        for l in range(self.javis.num_layers):
            k_flat, v_flat = layer_kvs[l]
            k_res = k_flat.view(bsz, num_chunks, self.javis.num_kv_heads, self.javis.num_queries, self.javis.head_dim)
            v_res = v_flat.view(bsz, num_chunks, self.javis.num_kv_heads, self.javis.num_queries, self.javis.head_dim)
            reshaped_layer_kvs.append((k_res, v_res))

        javis_vecs = javis_out_flat.view(bsz, num_chunks, self.javis.num_queries, -1)

        # 统一返回 Deep KV
        if return_metrics:
            return javis_out_flat, javis_vecs, reshaped_layer_kvs, javis_metrics, current_out_cos
        return javis_out_flat, javis_vecs, reshaped_layer_kvs, current_out_cos
        
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
            start_pos = memory_positions[b_idx, c_idx]
            assert (start_pos + self.javis.num_queries <= inputs_embeds.size(1)).all()
            for q_i in range(self.javis.num_queries):
                inputs_embeds[b_idx, start_pos + q_i] = memory_vectors[b_idx, c_idx, q_i].to(
                    inputs_embeds.dtype
                )

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
