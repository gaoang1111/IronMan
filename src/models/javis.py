from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CompressedMemory:
    """Compressed vectors V and their zipper positions."""

    vectors: torch.Tensor  # [B, C, H]
    positions: torch.LongTensor  # [B, C] positions in zipper sequence


def _init_eye_plus_noise_(linear: nn.Linear, *, std_noise: float) -> None:
    """Initialize a linear layer as identity + small noise."""
    if linear.in_features != linear.out_features:
        raise ValueError(
            f"eye+noise init requires in_features == out_features, got {linear.in_features} vs {linear.out_features}"
        )
    with torch.no_grad():
        linear.weight.copy_(torch.eye(linear.in_features, device=linear.weight.device, dtype=linear.weight.dtype))
        linear.weight.add_(torch.randn_like(linear.weight) * float(std_noise))


class Javis(nn.Module):
    """
    Cross-attention module for compressing chunk representations into memory vectors.
    
    Architecture:
        - Input: Hidden states from compressor [B, L, H]
        - Learnable queries: [G, Q, H] where G = num_layers / query_group_size
        - Output: Compressed vectors [B, G, Q, H]
        - Deep KV: Per-layer KV projections for all 32 transformer layers
    
    Key features:
        - Query groups: Different query sets for different layer groups
        - Delta-q: Adaptive query bias based on chunk content
        - Deep KV injection: Residual KV for each transformer layer
        - Layer gates: Learnable scaling for each layer's KV contribution
    """

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_size: int,
        num_heads: int,
        num_queries: int,
        query_group_size: int,
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
        assert num_layers % query_group_size == 0, f"num_layers must be divisible by query_group_size, got {num_layers} % {query_group_size}"
        self.query_group_size = int(query_group_size)
        self.num_query_group = int(num_layers // query_group_size)

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

        # Input projection (if dimensions differ)
        self.in_proj: nn.Module
        if int(input_dim) != self.hidden_size:
            self.in_proj = nn.Linear(int(input_dim), self.hidden_size, bias=False).to(dtype=dtype)
            nn.init.normal_(self.in_proj.weight, mean=0.0, std=0.02)
        else:
            self.in_proj = nn.Identity()

        # Layer norms
        self.ln_in = nn.LayerNorm(self.hidden_size, elementwise_affine=True).to(dtype=dtype)
        self.group_ln_out = nn.ModuleList([
            nn.LayerNorm(self.hidden_size, dtype=dtype) for _ in range(self.num_query_group)
        ])
        
        # Learnable queries: [G, Q, H]
        self.q_base = nn.Parameter(torch.empty((self.num_query_group, num_queries, hidden_size), dtype=dtype))
        self.q_proj = nn.Linear(hidden_size, num_queries * hidden_size).to(dtype=dtype)
        self.alpha = 1.0
        self.q_ln = nn.LayerNorm(hidden_size, dtype=dtype)
        nn.init.normal_(self.q_base, mean=0.0, std=1.0)

        # KV projections
        self.wk = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype=dtype)
        self.wv = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype=dtype)
        self.wo = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(dtype=dtype)

        _init_eye_plus_noise_(self.wk, std_noise=self.init_noise_std)
        _init_eye_plus_noise_(self.wv, std_noise=self.init_noise_std)
        _init_eye_plus_noise_(self.wo, std_noise=self.init_noise_std)
        self.current_q_grad_cos = None

        # ==========================================
        # Deep KV Injection Protocol
        # Each layer needs: num_kv_heads * head_dim * 2 (K + V)
        # ==========================================
        self.kv_dim_per_layer = self.num_kv_heads * self.head_dim * 2

        # Per-layer projection weights: [L, H, kv_dim_per_layer]
        self.kv_proj_weights = nn.Parameter(torch.empty(self.num_layers, self.hidden_size, self.kv_dim_per_layer))
        std_dev = 1.0 / math.sqrt(self.hidden_size)
        nn.init.normal_(self.kv_proj_weights, mean=0.0, std=std_dev)

        # Per-layer gates for residual injection strength
        self.layer_gates = nn.Parameter(torch.full((self.num_layers,), 0.07))

    def get_all_layer_kv(self, v_out_blocks: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Transform Javis output into per-layer KV pairs for deep injection.
        
        Args:
            v_out_blocks: [B, G, Q, H] - Javis output organized by query groups
            
        Returns:
            Tuple of 32 (K, V) pairs, each with shape [B, num_kv_heads, Q, head_dim]
        """
        B, G, Q, H = v_out_blocks.shape
        L = self.num_layers
        L_per_G = L // G
        
        # Expand group dimension to align with layers: [B, G, L_per_G, Q, H]
        v_expanded = v_out_blocks.unsqueeze(2).expand(-1, -1, L_per_G, -1, -1)
        
        # Reshape: [B, L, Q, H] -> [L, B, Q, H]
        v_aligned = v_expanded.reshape(B, L, Q, H).transpose(0, 1).contiguous()
        
        # BMM: [L, B*Q, H] @ [L, H, KV_dim] -> [L, B*Q, KV_dim]
        v_bmm = v_aligned.view(L, B * Q, H)
        flat_kv = torch.bmm(v_bmm, self.kv_proj_weights)
        
        # Reshape: [L, B, Q, KV_dim] -> [B, L, Q, KV_dim]
        flat_kv = flat_kv.view(L, B, Q, self.kv_dim_per_layer).transpose(0, 1)
        
        # Reshape to LLaMA format: [B, L, Q, 2, num_kv_heads, head_dim]
        flat_kv = flat_kv.view(B, L, Q, 2, self.num_kv_heads, self.head_dim)
        
        # Permute: [L, 2, B, num_kv_heads, Q, head_dim]
        flat_kv = flat_kv.permute(1, 3, 0, 4, 2, 5)
        
        # Apply per-layer gates
        gates = self.layer_gates.view(L, 1, 1, 1, 1, 1).to(flat_kv.dtype)
        flat_kv = flat_kv * gates

        past_key_values = []
        for l in range(L):
            k = flat_kv[l, 0]  # [B, num_kv_heads, Q, head_dim]
            v = flat_kv[l, 1]  # [B, num_kv_heads, Q, head_dim]
            past_key_values.append((k, v))
            
        return tuple(past_key_values)

    def pre_kv_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Pre-process hidden states before KV projection."""
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
        """
        Compress hidden states into memory vectors.
        
        Args:
            hidden: [B, L, H] - Hidden states from compressor
            attention_mask: [B, L] - Mask for valid tokens (1 = valid)
            return_metrics: Whether to return diagnostic metrics
            
        Returns:
            If return_metrics=False:
                (output, cos_similarity) where output is [B, G, Q, H]
            If return_metrics=True:
                (output, metrics_dict, cos_similarity)
        """
        x = self.pre_kv_hidden(hidden)  # [B, L, H]
        k_full = self.wk(x)             # [B, L, H]
        v_full = self.wv(x)             # [B, L, H]

        B = int(x.size(0))
        G = self.num_query_group
        Q_len = self.num_queries
        H = self.hidden_size
        seq_len = int(k_full.size(1))
        
        # Compute delta-q (adaptive query bias based on chunk content)
        if attention_mask is None:
            chunk_mean = x.mean(dim=1)
        else:
            m = attention_mask.to(dtype=x.dtype, device=x.device).view(B, seq_len, 1)
            denom = m.sum(dim=1).clamp(min=1.0)
            chunk_mean = (x * m).sum(dim=1) / denom
        delta_q = self.q_proj(chunk_mean) 
        delta_q = delta_q.view(B, self.num_queries, H).unsqueeze(1)
        
        # q_tensor: [B, G, Q, H]
        q_tensor = self.q_base.unsqueeze(0).expand(B, -1, -1, -1) + delta_q
        
        # Register gradient hook for orthogonality monitoring
        if self.training and return_metrics and q_tensor.requires_grad:
            def _q_grad_hook(grad: torch.Tensor) -> None:
                grad_q0 = grad[:, :, 0, :].float()
                grad_q1 = grad[:, :, 1, :].float()
                cos_sim = F.cosine_similarity(grad_q0, grad_q1, dim=-1, eps=1e-8).mean()
                self.current_q_grad_cos = float(cos_sim.item())
            q_tensor.register_hook(_q_grad_hook)

        h_heads = self.num_heads
        d_k = self.hidden_size // h_heads

        # Multi-head attention
        # q: [B, G, Q, h, d] -> [B, G, h, Q, d]
        q = q_tensor.view(B, G, Q_len, h_heads, d_k).transpose(2, 3)
        
        # k, v: [B, L_seq, h, d] -> [B, 1, h, L_seq, d]
        k = k_full.view(B, seq_len, h_heads, d_k).unsqueeze(1).permute(0, 1, 3, 2, 4)
        v = v_full.view(B, seq_len, h_heads, d_k).unsqueeze(1).permute(0, 1, 3, 2, 4)

        # Attention: [B, G, h, Q, d] @ [B, 1, h, d, L_seq] -> [B, G, h, Q, L_seq]
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if attention_mask is not None:
            m = attention_mask.to(dtype=torch.bool, device=logits.device)
            m = m.view(B, 1, 1, 1, seq_len)
            logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)

        attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        
        # Context: [B, G, h, Q, L_seq] @ [B, 1, h, L_seq, d] -> [B, G, h, Q, d]
        ctx = torch.matmul(attn, v)
        
        # Reshape: [B, G, Q, H]
        ctx = ctx.transpose(2, 3).contiguous().view(B, G, Q_len, self.hidden_size)

        # Output projection with group LayerNorm
        out = self.wo(ctx)  # [B, G, Q, H]
        
        if getattr(self, "ln_out_enabled", False):
            outs = []
            for g in range(G):
                outs.append(self.group_ln_out[g](out[:, g, :, :]))
            out = torch.stack(outs, dim=1)

        # Compute cosine similarity between query outputs
        current_out_cos = torch.zeros((), device=out.device, dtype=out.dtype)
        if Q_len >= 2:
            current_out_cos = F.cosine_similarity(out[:, :, 0, :], out[:, :, 1, :], dim=-1, eps=1e-8).mean()

        # Shortcut connection with global mean
        if attention_mask is None:
            global_mean = hidden.mean(dim=1, keepdim=True)
        else:
            m = attention_mask.to(dtype=hidden.dtype, device=hidden.device).view(B, seq_len, 1)
            denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
            global_mean = (hidden * m).sum(dim=1, keepdim=True) / denom
        shortcut = global_mean.view(B, 1, 1, self.hidden_size).expand(-1, G, Q_len, -1)
        final_out = out + shortcut * 0.5

        if return_metrics:
            metrics = self._compute_metrics(out, global_mean, attn, attention_mask, seq_len, Q_len, current_out_cos)
            return final_out, metrics, current_out_cos
        return final_out, current_out_cos

    def _compute_metrics(
        self,
        out: torch.Tensor,
        global_mean: torch.Tensor,
        attn: torch.Tensor,
        attention_mask: torch.Tensor | None,
        seq_len: int,
        Q_len: int,
        current_out_cos: torch.Tensor,
    ) -> dict[str, float]:
        """Compute diagnostic metrics for logging."""
        metrics = {}
        with torch.no_grad():
            out_detached = out.detach().float()
            mean_detached = global_mean.detach().float()
            out_pair = out_detached[:, :, :2, :]
            norm_out = out_pair.norm(p=2, dim=-1).mean()
            norm_mean = mean_detached.norm(p=2, dim=-1).mean()
            metrics["javis_norm_ratio"] = float((norm_out / (norm_mean + 1e-9)).item())
            metrics["javis_out_cos"] = float(current_out_cos.item())
            
            attn_detached = attn.detach().float()  # [B, G, h, Q, L_seq]
            
            # Attention position analysis
            if attention_mask is None:
                if seq_len >= 3:
                    attn_last3 = attn_detached[:, :, :, :, -3:]
                    attn_last3_pos_means = attn_last3.mean(dim=(0, 1, 2, 3))
                    metrics["javis_attn_pos_-3"] = float(attn_last3_pos_means[0].item())
                    metrics["javis_attn_pos_-2"] = float(attn_last3_pos_means[1].item())
                    metrics["javis_attn_pos_-1"] = float(attn_last3_pos_means[2].item())
                else:
                    fallback_mean = float(attn_detached.mean().item()) if seq_len > 0 else 0.0
                    metrics["javis_attn_pos_-3"] = fallback_mean
                    metrics["javis_attn_pos_-2"] = fallback_mean
                    metrics["javis_attn_pos_-1"] = fallback_mean
            else:
                valid_lens = attention_mask.to(dtype=torch.long, device=attn_detached.device).sum(dim=1)
                for off in (3, 2, 1):
                    ok = valid_lens >= off
                    if ok.any():
                        idx = (valid_lens[ok] - off).view(-1, 1, 1, 1, 1)
                        idx = idx.expand(-1, attn_detached.size(1), attn_detached.size(2), attn_detached.size(3), 1)
                        vals = attn_detached[ok].gather(dim=-1, index=idx).squeeze(-1)
                        metrics[f"javis_attn_pos_-{off}"] = float(vals.mean().item())
                    else:
                        metrics[f"javis_attn_pos_-{off}"] = 0.0

            # KL divergence between query attention distributions
            token_len = seq_len
            if token_len > 0 and Q_len >= 2:
                attn_q0 = attn_detached[:, :, :, 0, :token_len].mean(dim=(1, 2)) 
                attn_q1 = attn_detached[:, :, :, 1, :token_len].mean(dim=(1, 2))
                kl_div = F.kl_div((attn_q1 + 1e-9).log(), attn_q0, reduction="batchmean")
                metrics["javis_attn_kl"] = float(kl_div.item())
            
            # Layer gate values for target layers
            target_layers = [15, 23, 31]
            gates_val = self.layer_gates.detach().float()
            metrics["javis_gate_avg_target"] = float(gates_val[target_layers].mean().item())
            metrics["javis_gate_15"] = float(gates_val[15].item())
            metrics["javis_gate_23"] = float(gates_val[23].item())
            metrics["javis_gate_31"] = float(gates_val[31].item())

        return metrics
