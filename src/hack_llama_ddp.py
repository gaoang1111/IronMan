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
    # 🚀 1. 唯一路由判断：是否开启 Deep KV 注入
    # =========================================================
    # ⚡️ 安全解析原生参数
    attention_mask = kwargs.get("attention_mask", args[0] if len(args) > 0 else None)
    position_ids = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
    past_key_value = kwargs.get("past_key_value", args[2] if len(args) > 2 else None)
    output_attentions = kwargs.get("output_attentions", args[3] if len(args) > 3 else False)

    bsz, q_len, _ = hidden_states.size()
    layer_idx = getattr(self, "layer_idx", None)
    
    # =========================================================
    # 🧩 2. 原生投影与深层 KV 覆写 (Deep KV Overwrite)
    # =========================================================
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    if layer_idx is not None and DEEP_KV_CONTEXT.get("layer_kvs") is not None:
        # 直接拿属于这一层的 (K_det, V_det)
        k_javis_raw, v_javis_raw = DEEP_KV_CONTEXT["layer_kvs"][layer_idx]
        
        # 调整形状为 [B, C, num_q, num_kv_heads, head_dim] 以备提取
        # 因为前面 reshape 已经是 [B, C, num_kv_heads, Q, head_dim]
        # k_javis_raw shape 已经是你需要的，只需要转置对齐 Llama
        k_javis = k_javis_raw
        v_javis = v_javis_raw

        mem_pos = DEEP_KV_CONTEXT["memory_positions"]
        num_q = DEEP_KV_CONTEXT["num_queries"]

        key_states = key_states.clone()
        value_states = value_states.clone()

        for b in range(bsz):
            for c in range(mem_pos.size(1)):
                start_idx = int(mem_pos[b, c].item())
                if start_idx >= 0 and start_idx + num_q <= q_len:
                    key_states[b, :, start_idx : start_idx + num_q, :] = k_javis[b, c].to(key_states.dtype)
                    value_states[b, :, start_idx : start_idx + num_q, :] = v_javis[b, c].to(value_states.dtype)

    # =========================================================
    # 🌀 3. RoPE 旋转与 GQA 展开
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
    # ⚡️ 4. 分块注意力计算
    # =========================================================
    attn_output = torch.zeros_like(query_states)
    
    CHUNK_SIZE = 1024
    for i in range(0, q_len, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, q_len)
        q_chunk = query_states[:, :, i:end, :]
        
        # 使用严格数学缩放
        attn_weights = torch.matmul(q_chunk, key_states) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, i:end, :]
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output[:, :, i:end, :] = torch.matmul(attn_weights, value_states)
        
        del attn_weights, q_chunk

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    
    del query_states, key_states, value_states
    
    # =========================================================
    # 🎯 5. 严格签名对齐
    # =========================================================
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
        # 🛡️ 2. 建立全维度隔离带 (防止 Llama 的 GC 炸毁 Javis 图)
        # =========================================================
        # A. 隔离 memory_vectors (给浅层拼接用)
        memory_vectors_det = memory_vectors.detach().requires_grad_(True)
        self._orig_mem_vecs = memory_vectors
        self._det_mem_vecs = memory_vectors_det

        # B. 隔离 32 层的 K 和 V (给深层注入用)
        self._orig_layer_kvs = []
        self._det_layer_kvs = []
        deep_layer_kvs_det = []
        
        for k_raw, v_raw in deep_layer_kvs:
            k_det = k_raw.detach().requires_grad_(True)
            v_det = v_raw.detach().requires_grad_(True)
            
            # 存起来等会儿算 surrogate_loss 用
            self._orig_layer_kvs.extend([k_raw, v_raw])
            self._det_layer_kvs.extend([k_det, v_det])
            
            # 组装给 Context 用
            deep_layer_kvs_det.append((k_det, v_det))

        # =========================================================
        # 🚀 3. 挂载给 Generator (只允许接触隔离后的 det 张量)
        # =========================================================
        DEEP_KV_CONTEXT["layer_kvs"] = deep_layer_kvs_det
        DEEP_KV_CONTEXT["memory_vectors"] = memory_vectors_det 
        DEEP_KV_CONTEXT["memory_positions"] = mem_pos
        DEEP_KV_CONTEXT["num_queries"] = self.iron.javis.num_queries


        # 🚀 3. 这里的 inputs_embeds 必须用隔离后的版本！
        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors_det, # 使用 det 版
            memory_positions=mem_pos,
        )
        self._register_grad_probe("inputs_embeds", inputs_embeds)

        out = self.iron(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_2d,
                position_ids=position_ids,
                labels=labels,
                use_cache=False,
            )
        

        gen_loss = out.loss
        
        # l2_loss = memory_vectors.norm(p=2, dim=-1).mean() if self.phase == "phase2" else torch.zeros_like(gen_loss)
        l2_loss = (
            memory_vectors.norm(p=2, dim=-1).mean() 
            if self.phase == "phase2" 
            else torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
        )
        # ortho_penalty = torch.relu(current_out_cos.to(gen_loss.dtype) - 0.1)
        
        q_params = self.iron.javis.q  # shape: [G, Q, H], 例如 [8, 2, 4096]
        if q_params.size(1) >= 2:
            # 计算每组内 Q0 和 Q1 的余弦相似度
            q_cos = F.cosine_similarity(q_params[:, 0, :], q_params[:, 1, :], dim=-1)
            # 取所有组的平均值
            mean_q_cos = q_cos.mean()
            # 惩罚项：如果余弦相似度大于 0.1，则施加惩罚
            ortho_penalty = torch.relu(mean_q_cos.to(gen_loss.dtype) - 0.1)
        else:
            ortho_penalty = torch.tensor(0.0, device=gen_loss.device, dtype=gen_loss.dtype)
            mean_q_cos = torch.tensor(0.0)
            
        # 存在抽屉里，一会儿和代理 Loss 一起引爆
        self._extra_loss = (self.l2_coeff * l2_loss) + (self.javis_q_cos_coeff * ortho_penalty)

        # ⚠️ 必须加 .detach()，切断 DDP 的最后一条偷家路线！
        if return_metrics:
            # 只有要 metrics 的时候，才清理并返回 4 个值
            clean_metrics = None
            if javis_metrics is not None:
                clean_metrics = {k: (v.detach().item() if isinstance(v, torch.Tensor) else v) for k, v in javis_metrics.items()}
            return gen_loss, l2_loss.detach(), clean_metrics, current_out_cos.detach()
        else:
            # 正常 Step，严禁返回 metrics，老老实实返回 3 个值！
            return gen_loss, l2_loss.detach(), current_out_cos.detach()
    
        
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