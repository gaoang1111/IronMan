import math
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb  # [Added] 引入 WandB
from dataclasses import dataclass

from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from transformers import HfArgumentParser

from src.data_processor import IronCellCollator
from src.modeling_iron_cell import IronCellModel

import types
DEEP_KV_CONTEXT = {
    "layer_kvs": None,      # Tuple[Tuple[K, V]], 32 layers
    "memory_positions": None, # [B, C] 占位符在序列中的起始索引
    "num_queries": 2,
    "javis_module": None,   
    "memory_vectors": None,
}

ORIGINAL_LLAMA_ATTENTION_FORWARD = LlamaAttention.forward


def smart_hybrid_attention_forward(
    self,
    hidden_states: torch.Tensor,
    *args,
    **kwargs,
):
    # =========================================================
    # 🚀 1. 偷渡包裹解包
    # =========================================================
    layer_idx = getattr(self, "layer_idx", -1) 
    
    # 设定注入规则：每 4 层注入一次（比如 3, 7, 11, 15... 层）
    # 或者直接指定层：target_layers = [15, 23, 31]
    # is_target_layer = (layer_idx % 4 == 3) 
    is_target_layer = layer_idx in [15, 23, 31]

    attention_mask = kwargs.get("attention_mask", args[0] if len(args) > 0 else None)
    position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
    past_key_value = kwargs.get("past_key_value", args[2] if len(args) > 2 else None)
    output_attentions = kwargs.get("output_attentions", args[3] if len(args) > 3 else False)

    javis_kv = kwargs.get("javis_kv", None)
    javis_meta = kwargs.get("javis_meta", None)

    bsz, q_len, _ = hidden_states.size()
    
    # =========================================================
    # 🧩 2. 原生投影
    # =========================================================
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    # =========================================================
    # 💉 3. 深层 KV 残差门控注入 (Zero-Initialized Gating)
    # =========================================================
    # if 0:
    if javis_kv is not None and javis_meta is not None and is_target_layer:
        # print(f"================={layer_idx=} deep kv")
        mem_pos, num_q = javis_meta
        # 这里的 k_javis_gated 已经是乘过 0.1 门控的了！
        k_javis_gated, v_javis_gated = javis_kv 
        
        # 🚀 必须 clone！防止 PyTorch 报 In-place 修改错误
        key_states = key_states.clone()
        value_states = value_states.clone()

        for b in range(bsz):
            for c in range(mem_pos.size(1)):
                start_idx = int(mem_pos[b, c].item())
                if start_idx >= 0 and start_idx + num_q <= q_len:
                    # 提取原生占位符 KV (为了安全也加个 clone)
                    orig_k = key_states[b, :, start_idx : start_idx + num_q, :].clone()
                    orig_v = value_states[b, :, start_idx : start_idx + num_q, :].clone()
                    
                    # 🚀 直接残差无脑加！逻辑极其干净！
                    key_states[b, :, start_idx : start_idx + num_q, :] = orig_k + k_javis_gated[b, c].to(key_states.dtype)
                    value_states[b, :, start_idx : start_idx + num_q, :] = orig_v + v_javis_gated[b, c].to(value_states.dtype)
    # =========================================================
    # 🌀 4. RoPE 旋转与 GQA 展开
    # =========================================================
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is None:
         raise ValueError("position_embeddings must be provided in kwargs")

    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(bsz, self.config.num_key_value_heads, num_key_value_groups, q_len, self.head_dim).reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)
        value_states = value_states[:, :, None, :, :].expand(bsz, self.config.num_key_value_heads, num_key_value_groups, q_len, self.head_dim).reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)
    key_states = key_states.transpose(2, 3) 

    # =========================================================
    # ⚡️ 5. 分块注意力计算
    # =========================================================
    attn_output = torch.zeros_like(query_states)
    
    CHUNK_SIZE = 1024
    for i in range(0, q_len, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, q_len)
        q_chunk = query_states[:, :, i:end, :]
        
        attn_weights = torch.matmul(q_chunk, key_states) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, i:end, :]
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output[:, :, i:end, :] = torch.matmul(attn_weights, value_states)
        
        del attn_weights, q_chunk

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    
    del query_states, key_states, value_states
    
    if output_attentions:
        return attn_output, None, past_key_value
    else:
        return attn_output, past_key_value



class TrainStepModuleForFullLayersKVInjection(nn.Module):
    def __init__(
        self,
        iron: IronCellModel,
        *,
        phase: str,
        l2_coeff: float = 1e-4,
        javis_q_cos_coeff: float = 1.0,
        distill_coeff: float = 0.0,
        distill_layers: list[int] | None = None,
        chunk_size: int = 16,
    ) -> None:
        super().__init__()
        self.iron = iron
        self.phase = str(phase)
        self.l2_coeff = float(l2_coeff)
        self.javis_q_cos_coeff = float(javis_q_cos_coeff)
        self.distill_coeff = float(distill_coeff)
        self.distill_layers = distill_layers or []
        self.chunk_size = int(chunk_size)
        self.grad_probe = False
        self.grad_probe_sums: dict[str, torch.Tensor] = {}

        for layer in self.iron.generator.model.layers:
            # 将我们写的函数，动态绑定为该层 self_attn 实例的方法
            layer.self_attn.forward = types.MethodType(smart_hybrid_attention_forward, layer.self_attn)

    def reset_grad_probe(self) -> None:
        self.grad_probe_sums = {}

    def _register_grad_probe(self, name: str, tensor: torch.Tensor) -> None:
        if not self.grad_probe or not tensor.requires_grad:
            return
        def _hook(grad: torch.Tensor) -> None:
            val = (grad.float() ** 2).sum()
            if name in self.grad_probe_sums:
                self.grad_probe_sums[name] = self.grad_probe_sums[name] + val
            else:
                self.grad_probe_sums[name] = val
        tensor.register_hook(_hook)

    def forward(self, batch, *, return_metrics: bool = False):  # type: ignore[override]
        device = self.iron.device
        
        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        attn_2d = batch.attention_mask_2d.to(device)
        position_ids = batch.position_ids.to(device)
        labels = batch.labels.to(device)

        memory_out = self.iron.compute_compressed_vectors(
            chunk_input_ids=chunk_ids,
            chunk_attention_mask=chunk_mask,
            return_metrics=return_metrics,
        )
        
        javis_metrics = None
        if return_metrics:
            memory_hook, memory_vectors, deep_layer_kvs, javis_metrics, current_out_cos = memory_out
        else:
            memory_hook, memory_vectors, deep_layer_kvs, current_out_cos = memory_out
            
        self._register_grad_probe("memory_vectors", memory_hook)

        # 构建浅层的 inputs_embeds
        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors, 
            memory_positions=mem_pos,
        )

        self._register_grad_probe("inputs_embeds", inputs_embeds)

        out = self.iron(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_2d,
            position_ids=position_ids, 
            labels=labels,
            use_cache=False,
            javis_all_layer_kvs=deep_layer_kvs, 
            javis_meta=(mem_pos, self.iron.javis.num_queries)
        )
        
        if hasattr(out, "logits") and out.logits.requires_grad:
            self._register_grad_probe("logits", out.logits)

        gen_loss = out.loss
        
        l2_loss = (
            memory_vectors.norm(p=2, dim=-1).mean() 
            if self.phase == "phase2" 
            else torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
        )
        
        q_params = self.iron.javis.q_base
        Q = q_params.size(1)
        
        if Q >= 2:
            # 1. 沿着 H 维度做 L2 归一化: [G, Q, H]
            q_norm = F.normalize(q_params, p=2, dim=-1)
            
            # 2. 组内计算所有 Query 两两之间的相似度矩阵: [G, Q, H] @ [G, H, Q] -> [G, Q, Q]
            sim_matrix = torch.bmm(q_norm, q_norm.transpose(1, 2))
            
            # 3. 过滤掉对角线（自己和自己的相似度必定是 1，不需要惩罚）
            eye_mask = torch.eye(Q, device=gen_loss.device, dtype=torch.bool).unsqueeze(0)
            off_diag_sim = sim_matrix.masked_select(~eye_mask)
            
            # 4. 计算非对角线元素的绝对值平均（防止正负抵消），或者直接用原值
            # 建议用 abs()，因为完全相反（-1）也是一种高度线性相关，同样浪费容量
            mean_q_cos = off_diag_sim.abs().mean()
            
            # 5. 超过 0.1 的部分进行惩罚
            ortho_penalty = torch.relu(mean_q_cos.to(gen_loss.dtype) - 0.1)
        else:
            ortho_penalty = torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
            mean_q_cos = torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
            
        total_loss = gen_loss + (self.l2_coeff * l2_loss) + (self.javis_q_cos_coeff * ortho_penalty)

        
        if return_metrics:
            with torch.no_grad():
                # 安全提取 Tensor
                gates = self.iron.javis.layer_gates.clone().detach().float()
                
                # 就算用了 FSDP，因为咱们只取平均值，可以直接算
                # target_layer = [15, 23, 31]
                # gate_val = gates[target_layer].mean().item() if gates.numel() > 0 else 0.0

            clean_metrics = None
            if javis_metrics is not None:
                clean_metrics = {k: (v.detach().item() if isinstance(v, torch.Tensor) else v) for k, v in javis_metrics.items()}
            
            # clean_metrics["gate_val"] = gate_val
            return total_loss, l2_loss.detach(), clean_metrics, mean_q_cos.detach()
        else:
            return total_loss, l2_loss.detach(), mean_q_cos.detach()
        


    def _forward(self, batch, *, return_metrics: bool = False):  # type: ignore[override]
        device = self.iron.device
        # 判断是否开启特征蒸馏
        
        
        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        attn_2d = batch.attention_mask_2d.to(device)
        position_ids = batch.position_ids.to(device)
        labels = batch.labels.to(device)

        memory_out = self.iron.compute_compressed_vectors(
            chunk_input_ids=chunk_ids,
            chunk_attention_mask=chunk_mask,
            return_metrics=return_metrics,
        )
        javis_metrics = None
        if return_metrics:
            memory_hook, memory_vectors, deep_layer_kvs, javis_metrics, current_out_cos = memory_out
        else:
            memory_hook, memory_vectors, deep_layer_kvs, current_out_cos = memory_out
            
        self._register_grad_probe("memory_vectors", memory_hook)

        # =========================================================
        # 🚀 因为 use_reentrant=False，重复梯度 Bug 已消失，直接全连通！
        # =========================================================
        DEEP_KV_CONTEXT["layer_kvs"] = deep_layer_kvs
        DEEP_KV_CONTEXT["memory_vectors"] = memory_vectors 
        DEEP_KV_CONTEXT["memory_positions"] = mem_pos
        DEEP_KV_CONTEXT["num_queries"] = self.iron.javis.num_queries

        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors, 
            memory_positions=mem_pos,
        )

        out = self.iron(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_2d,
            position_ids=position_ids,
            labels=labels,
            use_cache=False,
        )
        
        gen_loss = out.loss
        
        l2_loss = (
            memory_vectors.norm(p=2, dim=-1).mean() 
            if self.phase == "phase2" 
            else torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
        )
        
        q_params = self.iron.javis.q
        if q_params.size(1) >= 2:
            q_cos = F.cosine_similarity(q_params[:, 0, :], q_params[:, 1, :], dim=-1)
            mean_q_cos = q_cos.mean()
            ortho_penalty = torch.relu(mean_q_cos.to(gen_loss.dtype) - 0.1)
        else:
            ortho_penalty = torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
            mean_q_cos = torch.tensor(0.0)
            
        # 🚀 把惩罚项直接加到主 loss 上
        total_loss = gen_loss + (self.l2_coeff * l2_loss) + (self.javis_q_cos_coeff * ortho_penalty)

        if return_metrics:
            clean_metrics = None
            if javis_metrics is not None:
                clean_metrics = {k: (v.detach().item() if isinstance(v, torch.Tensor) else v) for k, v in javis_metrics.items()}
            # 统一返回 total_loss
            return total_loss, l2_loss.detach(), clean_metrics, mean_q_cos.detach()
        else:
            return total_loss, l2_loss.detach(), mean_q_cos.detach()
    
        
    def manual_backward_sync(self, grad_accum_steps: int):
        total_sync_loss = torch.tensor(0.0, device=self.iron.device, dtype=torch.float32)
        has_grad = False

        # A. 收集 memory_vectors 的梯度 (来自 inputs_embeds)
        if hasattr(self, "_det_mem_vecs") and self._det_mem_vecs is not None:
            if self._det_mem_vecs.grad is not None:
                llama_grad = self._det_mem_vecs.grad.detach()
                total_sync_loss = total_sync_loss + torch.sum(self._orig_mem_vecs * llama_grad)
                has_grad = True

        # B. 收集 32 层 K 和 V 的总计 64 个梯度 (来自 32 层的 Attention)
        if hasattr(self, "_det_layer_kvs") and self._det_layer_kvs:
            for orig_t, det_t in zip(self._orig_layer_kvs, self._det_layer_kvs):
                if det_t.grad is not None:
                    total_sync_loss = total_sync_loss + torch.sum(orig_t * det_t.grad.detach())
                    has_grad = True

        # C. 加上外挂 Loss (L2 / Ortho)
        if hasattr(self, "_extra_loss") and self._extra_loss is not None:
            total_sync_loss = total_sync_loss + (self._extra_loss / grad_accum_steps)
            has_grad = True

        # D. 终极一发，万流归宗！
        if has_grad:
            total_sync_loss.backward()

        # E. 物理清理，防止爆显存
        DEEP_KV_CONTEXT["layer_kvs"] = None
        DEEP_KV_CONTEXT["memory_vectors"] = None
        self._orig_mem_vecs = None
        self._det_mem_vecs = None
        self._orig_layer_kvs = []
        self._det_layer_kvs = []
        self._extra_loss = None