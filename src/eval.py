from __future__ import annotations

from datetime import timedelta
import os
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
from src.train_utils import (
    JsonlDataset,
    _build_fsdp_auto_wrap_policy,
    _is_no_weight_decay_param,
    configure_special_embedding_mode,
    load_checkpoint,
    load_model,
    load_tokenizer,
    save_checkpoint,
    save_checkpoint_fsdp,
    warmup_init_javis_query,
)

import math
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer
import math

# =========================================================
# 0. 保存原生方法 (这就相当于给你的车装了双系统)
# =========================================================
ORIGINAL_LLAMA_ATTENTION_FORWARD = LlamaAttention.forward

# 全局上下文
DISTILL_CONTEXT = {
    "enabled": False,
    "target_layers": set(),
    "mask": None,
    "collected_energies": {} 
}

# =========================================================
# 1. 智能 Attention Patch
# =========================================================
def smart_hybrid_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    
    bsz, q_len, _ = hidden_states.size()
    
    # 获取 layer_idx (防御性)
    layer_idx = getattr(self, "layer_idx", None)
    
    # =========================================================
    # 🕵️‍♂️ 路由判断逻辑 (最关键的一步)
    # =========================================================
    should_use_distill = False
    
    if DISTILL_CONTEXT["enabled"]:
        # 1. 必须是目标层
        if layer_idx is not None and layer_idx in DISTILL_CONTEXT["target_layers"]:
            # 2. 必须有 Mask
            mask = DISTILL_CONTEXT["mask"]
            if mask is not None:
                # 3. 【核心防呆】长度必须匹配！
                # 如果 q_len 是 16 (Compressor)，但 mask 是 6216 -> 不匹配 -> 走原生
                # 如果 q_len 是 6216 (Generator)，mask 是 6216 -> 匹配 -> 走蒸馏
                if q_len == mask.size(-1): # 或者 mask.size(2) 取决于你的维度
                     should_use_distill = True

    # =========================================================
    # 🛣️ 路径 A: 原生 Flash Attention (省显存，快！)
    #    适用于：非蒸馏层、Compressor、或者没开开关时
    # =========================================================
    if not should_use_distill:
        # 直接调用保存下来的原生函数
        return ORIGINAL_LLAMA_ATTENTION_FORWARD(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs
        )

    # =========================================================
    # 🛣️ 路径 B: 自定义分块蒸馏 (为了算 Energy)
    #    适用于：Generator 的目标层
    # =========================================================
    
    # ... (这里放入之前那个完美的 分块计算 逻辑) ...
    # 为了代码整洁，我把核心逻辑简写在这里，请务必使用之前给你的“分块版”代码填充这里
    # -------------------------------------------------------------------------
    # 1. Projections
    query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

    # RoPE (这里你需要适配你的源码，如果源码 forward 传的是 position_embeddings)
    # 注意：ORIGINAL_LLAMA_ATTENTION_FORWARD 的参数签名可能和你的 patched 签名不一样
    # 你的源码里 LlamaAttention.forward 接收的是 position_embeddings
    # 请确认 kwargs 里或者参数里有没有 position_embeddings
    
    # ⚠️ 特别注意：因为我们现在的函数签名要兼容原生，所以参数要对齐
    # 你的源码 forward(self, hidden_states, position_embeddings, ...)
    # 所以我们需要从 args/kwargs 里拿
    
    position_embeddings = kwargs.get("position_embeddings", None)
    if position_embeddings is None:
         # 尝试从 args 解析，或者根据你的源码调整
         pass 

    # --- 这里是你的源码特定的 RoPE 调用 ---
    # 假设你使用的是之前那个源码版本
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    # GQA Repeat
    num_key_value_groups = self.config.num_attention_heads // self.config.num_key_value_heads
    if num_key_value_groups > 1:
        key_states = key_states[:, :, None, :, :].expand(bsz, self.config.num_key_value_heads, num_key_value_groups, q_len, self.head_dim).reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)
        value_states = value_states[:, :, None, :, :].expand(bsz, self.config.num_key_value_heads, num_key_value_groups, q_len, self.head_dim).reshape(bsz, self.config.num_attention_heads, q_len, self.head_dim)
    key_states = key_states.transpose(2, 3) # [B, H, D, S]

    # --- Chunked Attention ---
    attn_output = torch.zeros_like(query_states)
    energy_val = None
    
    # Energy 初始化
    energy_val = torch.zeros((bsz, self.config.num_attention_heads, q_len), device=query_states.device, dtype=torch.float32)
    distill_mask = DISTILL_CONTEXT["mask"]

    CHUNK_SIZE = 1024
    for i in range(0, q_len, CHUNK_SIZE):
        end = min(i + CHUNK_SIZE, q_len)
        q_chunk = query_states[:, :, i:end, :]
        
        # Score
        attn_weights = torch.matmul(q_chunk, key_states) * self.scaling
        if attention_mask is not None:
             attn_weights = attn_weights + attention_mask[:, :, i:end, :]
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Distill
        d_mask = distill_mask[:, :, i:end, :]
        energy_val[:, :, i:end] = (attn_weights * d_mask).sum(dim=-1)
        
        # Output
        attn_output[:, :, i:end, :] = torch.matmul(attn_weights, value_states)
        del attn_weights, q_chunk

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    attn_output = self.o_proj(attn_output)
    
    # 显式释放
    del query_states, key_states, value_states
    
    # 返回 (output, energy)
    # 注意：原生 forward 返回 (output, weights)，我们这里把 energy 放在 weights 的位置返回
    return attn_output, energy_val


# =========================================================
# 2. Patch Decoder Layer (接收 Energy)
# =========================================================
# 保存原生的 Decoder forward
ORIGINAL_DECODER_FORWARD = LlamaDecoderLayer.forward

def smart_hybrid_decoder_forward(self, hidden_states, **kwargs):
    # 1. 正常执行 Layer (里面会调我们的 smart_hybrid_attention_forward)
    # 注意：这里我们调用 ORIGINAL_DECODER_FORWARD 会导致它去调 self.self_attn
    # 因为 self.self_attn 已经被我们 Patch 了，所以它会进入我们的 smart 逻辑
    
    # ❌ 不能直接调 ORIGINAL_DECODER_FORWARD，因为我们需要拦截返回值
    # 你的源码 LlamaDecoderLayer.forward 可能会丢弃 attention weights
    # 所以我们必须把源码里的 forward 逻辑搬过来修改一下
    
    # ... (这里复制你源码里 LlamaDecoderLayer.forward 的代码) ...
    # 简写如下：
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    
    # 调用 Attention
    hidden_states, attn_weights = self.self_attn(
        hidden_states=hidden_states,
        **kwargs
    )
    
    # 【拦截】如果拿到了 Energy，存起来
    if attn_weights is not None and DISTILL_CONTEXT["enabled"]:
        # 再次检查是不是真的是 Energy (通过 Shape 或者 Flag)
        # 我们的 smart attention 只有在路由 B 才会返回非 None
        DISTILL_CONTEXT["collected_energies"][self.self_attn.layer_idx] = attn_weights

    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    
    return hidden_states

# =========================================================
# 应用 Patch
# =========================================================
# print("--> [System] Applying SMART HYBRID Patch...")
# LlamaAttention.forward = smart_hybrid_attention_forward
# LlamaDecoderLayer.forward = smart_hybrid_decoder_forward



def _parse_layers(spec: str) -> list[int]:
    return [int(x.strip()) for x in str(spec).split(",") if x.strip()]


@dataclass(frozen=True)
class TrainArgs:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    phase: str = "phase1"
    # [Config] 数据路径
    data_path: str = "data/phase1_train.jsonl"
    # [Config] 输出路径
    output_dir: str = "checkpoints"
    
    resume_path: str | None = None
    load_weights_only: bool = False
    chunk_size: int = 16
    batch_size: int = 2  # A800 上可以尝试大一点，比如 4 或 8
    lr: float = 5e-5
    lr_projector: float | None = None
    lr_generator: float | None = None
    lr_compressor: float | None = None
    weight_decay: float = 0.0
    parallel: str = "none"  # none|ddp|fsdp
    ddp_find_unused_parameters: bool = False
    fsdp_wrap: str = "full"  # full|generator_only
    fsdp_use_orig_params: bool = True
    fsdp_cpu_offload: bool = False
    train_only_special_token_embeddings: bool = False
    grad_accum_steps: int = 1
    warmup_steps: int = 0
    reset_step_on_resume: bool = False
    eval_data_path: str | None = None
    eval_steps: int = 0
    eval_max_batches: int = 0
    steps: int = 2000    # Phase 1 跑 2000 步即可
    save_steps: int = 500 # 每 500 步保存一次
    log_steps: int = 10
    global_epoch: int = 0
    random_gate: float = 0.0
    teacher_targets_path: str | None = None
    teacher_hidden_targets_path: str | None = None
    distill_layers: str = "24,26,28,30,31"
    distill_coeff: float = 0.0
    javis_query_warmup_samples: int | None = 100
    javis_query_warmup_save_path: str | None = None
    javis_num_queries: int = 2
    javis_q_cos_coeff: float = 1.0
    wandb_project: str = "soulbone" 
    wandb_run_name: str | None = None
    wandb_run_tags: str | None = None
    grad_probe: bool = False

class TrainStepModule(nn.Module):
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

    def _forward(self, batch, *, return_metrics: bool = False):  # type: ignore[override]
        device = self.iron.device
        distill_on = self.distill_coeff != 0.0 and hasattr(batch, "teacher_attn_targets") and batch.teacher_attn_targets.numel() > 0 and len(self.distill_layers) > 0
        
        DISTILL_CONTEXT["enabled"] = False
        DISTILL_CONTEXT["mask"] = None 
        DISTILL_CONTEXT["target_layers"] = set()
        
        # 调试打印：确保这里执行了 (如果还报错，请检查控制台有没有这句话)
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
            memory_vectors, javis_metrics, current_out_cos = memory_out
        else:
            memory_vectors, current_out_cos = memory_out
        self._register_grad_probe("memory_vectors", memory_vectors)

        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors,
            memory_positions=mem_pos,
        )
        self._register_grad_probe("inputs_embeds", inputs_embeds)

        handles = []
        if distill_on:
            # --- A. 预先构建 distill_mask (只做一次) ---
            # 这里的逻辑和你之前写的一样，但是要提前到 forward 前
            distill_mask = batch.attention_mask_2d.clone().to(device) # [B, S, S]
            distill_mask[:, :, :2] = False # 排除 BOS
            
            # 锁定 Prefix 区域
            # 注意：这里我们还没拿到 attn_mean，但我们可以用 seq_len (inputs_embeds.shape[1])
            seq_len = inputs_embeds.size(1)
            col_idx = torch.arange(seq_len, device=device).view(1, 1, -1)
            prefix_limit = batch.prefix_lens.to(device).view(-1, 1, 1)
            distill_mask = distill_mask & (col_idx >= 4) & (col_idx < prefix_limit - 1)
            
            # 为了在 Hook 里广播，扩展为 [B, 1, S, S]
            distill_mask_4d = distill_mask.unsqueeze(1) 

            # --- B. 定义 "偷天换日" Hook ---
            # def get_energy_hook(layer_idx, is_target_layer):
            #     def hook(module, args, output):
            #         # output[1] 是 attention weights [B, H, S, S]
            #         attn_weights = output[1]
                    
            #         # 1. 如果是目标层，提取能量
            #         if is_target_layer and attn_weights is not None:
            #             # 打印调试信息，确认进来了
            #             print(f"DEBUG: Extracting energy for layer {layer_idx}")
            #             energy = (attn_weights * distill_mask_4d.float()).sum(dim=-1)
            #             student_layer_energies[layer_idx] = energy
                    
            #         print(f"DEBUG: ========== for layer {layer_idx}")
            #         # 2. 【无论是不是目标层，统统销毁！】
            #         # 这样 output_attentions=True 产生的垃圾才不会堆积
            #         new_output = (output[0], None) + output[2:]
            #         return new_output
            #     return hook

            # # --- C. 注册 Hook ---
            # # 假设 model.generator 是 LlamaForCausalLM
            # # layers 位于 model.generator.model.layers
            # num_layers = len(self.iron.generator.model.layers)
            # target_layers_set = set(self.distill_layers)
            
            # for i in range(num_layers):
            #     is_target = i in target_layers_set
            #     layer_module = self.iron.generator.model.layers[i].self_attn
                
            #     # 给每一层都挂上 Hook
            #     h = layer_module.register_forward_hook(get_energy_hook(i, is_target))
            #     handles.append(h)

        # 4. 执行 Forward
        # 注意：这里依然要开 output_attentions=True，否则 layer 内部不会算 attn_weights
        # 但是因为我们的 Hook 把它换成了 None，所以 out.attentions 里全是 None，不占内存
        if distill_on:
             # ... (Mask 构建逻辑不变) ...
             DISTILL_CONTEXT["enabled"] = True
             DISTILL_CONTEXT["mask"] = distill_mask_4d
             DISTILL_CONTEXT["target_layers"] = set(self.distill_layers)
             DISTILL_CONTEXT["collected_energies"] = {} # 清空！
        else:
             DISTILL_CONTEXT["enabled"] = False

        out = self.iron(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_2d,
            position_ids=position_ids,
            labels=labels,
            # output_attentions=distill_on, 
            use_cache=False,
        )

        # DISTILL_CONTEXT["enabled"] = False
        # DISTILL_CONTEXT["mask"] = None
        # 5. 清理 Hook
        # for h in handles:
        #     h.remove()

        # =================================================================
        # 6. 计算 Loss
        # =================================================================
        loss = out.loss
        
        # L2 Loss
        l2_loss = torch.zeros((), device=device, dtype=loss.dtype)
        if self.phase == "phase2":
            l2_loss = memory_vectors.norm(p=2, dim=-1).mean()
            loss = loss + self.l2_coeff * l2_loss

        # Distill Loss
        distill_loss = torch.zeros((), device=device, dtype=loss.dtype)
        student_attn_mean_scalar = torch.zeros((), device=device, dtype=loss.dtype)
        teacher_attn_mean_scalar = torch.zeros((), device=device, dtype=loss.dtype)

        if distill_on:
            # 1. 聚合 Student 能量
            # student_layer_energies 里的 value 是 [B, Heads, S]
            # 我们需要 Stack -> Mean(Layers) -> Mean(Heads) -> [B, S]
            
            energies_list = []
            collected = DISTILL_CONTEXT["collected_energies"]
            
            for i in self.distill_layers:
                if i not in collected:
                     raise ValueError(f"Layer {i} energy missing! Patch failed.")
                energies_list.append(collected[i])
            
            energies = torch.stack(energies_list, dim=0)

            # # 这里的 stack 依然保留梯度！
            # energies = torch.stack([student_layer_energies[i] for i in self.distill_layers], dim=0) # [L, B, H, S]
            s_student = energies.mean(dim=0).mean(dim=1).to(dtype=loss.dtype) # [B, S]
            
            # 2. 准备 Teacher Targets (及对齐)
            teacher_targets = batch.teacher_attn_targets.to(device=device, dtype=loss.dtype)
            
            bsz, seq_len = s_student.shape
            p_idx = torch.arange(seq_len, device=device).view(1, -1).expand(bsz, -1)
            prefix_lens = batch.prefix_lens.to(device)
            valid_lens = batch.valid_lens.to(device)

            # Mask: 只在 Raw Token 区域计算 Loss
            raw_mask = (p_idx >= prefix_lens.view(-1, 1)) & (p_idx < valid_lens.view(-1, 1))
            
            # Mapping: Student Pos -> Teacher Pos
            # +1 是修正 BOS 偏移
            raw_pos = p_idx - prefix_lens.view(-1, 1) + 1
            teacher_max = teacher_targets.size(1)
            raw_mask = raw_mask & (raw_pos < teacher_max)
            
            # Gather Teacher Values
            teacher_aligned = teacher_targets.gather(1, raw_pos.clamp(min=0, max=teacher_max-1))
            
            # 3. 计算 MSE
            diff = (s_student - teacher_aligned).masked_fill(~raw_mask, 0.0)
            denom = raw_mask.sum().clamp(min=1)
            
            distill_loss = (diff ** 2).sum() / denom
            loss = loss + self.distill_coeff * distill_loss
            
            # Metrics
            student_attn_mean_scalar = (s_student.masked_fill(~raw_mask, 0.0).sum() / denom).detach()
            teacher_attn_mean_scalar = (teacher_aligned.masked_fill(~raw_mask, 0.0).sum() / denom).detach()

        current_out_cos = current_out_cos.to(dtype=loss.dtype)
        ortho_penalty = torch.relu(current_out_cos - 0.1)
        if self.javis_q_cos_coeff != 0.0:
            loss = loss + self.javis_q_cos_coeff * ortho_penalty

        if return_metrics:
            return loss, l2_loss, distill_loss, student_attn_mean_scalar, teacher_attn_mean_scalar, javis_metrics, current_out_cos
        return loss, l2_loss, distill_loss, current_out_cos

    def forward(self, batch, *, return_metrics: bool = False):  # type: ignore[override]
        device = self.iron.device
        # 判断是否开启特征蒸馏
        distill_on = (
            self.distill_coeff != 0.0
            and hasattr(batch, "teacher_hidden_targets")
            and batch.teacher_hidden_targets.numel() > 0
            and hasattr(batch, "valid_v_lens")
            and batch.valid_v_lens.numel() > 0
            and int(batch.valid_v_lens.max().item()) > 0
        )
        
        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        attn_2d = batch.attention_mask_2d.to(device)
        position_ids = batch.position_ids.to(device)
        labels = batch.labels.to(device)

        # 1. 过 Javis 提取 V
        memory_out = self.iron.compute_compressed_vectors(
            chunk_input_ids=chunk_ids,
            chunk_attention_mask=chunk_mask,
            return_metrics=return_metrics,
        )
        javis_metrics = None
        if return_metrics:
            memory_vectors, javis_metrics, current_out_cos = memory_out
        else:
            memory_vectors, current_out_cos = memory_out
            
        self._register_grad_probe("memory_vectors", memory_vectors)

        memory_vectors = torch.zeros_like(memory_vectors)

        # 2. 拼接底层的 inputs_embeds
        inputs_embeds = self.iron.build_inputs_embeds(
            zipper_input_ids=zipper_ids,
            memory_vectors=memory_vectors,
            memory_positions=mem_pos,
        )
        self._register_grad_probe("inputs_embeds", inputs_embeds)

        # 3. 极其清爽的 Forward！【删除了所有 Hook】
        # 只要开启 output_hidden_states=True
        out = self.iron(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_2d,
            position_ids=position_ids,
            labels=labels,
            output_hidden_states=distill_on, # 关键！
            use_cache=False,
        )

        loss = out.loss
        
        # L2 Loss (保持不变)
        l2_loss = torch.zeros((), device=device, dtype=loss.dtype)
        if self.phase == "phase2":
            l2_loss = memory_vectors.norm(p=2, dim=-1).mean()
            loss = loss + self.l2_coeff * l2_loss

        # =================================================================
        # 【核心修改】：核弹级 Hidden State MSE 蒸馏
        # =================================================================
        distill_loss = torch.zeros((), device=device, dtype=loss.dtype)
        student_attn_mean_scalar = torch.zeros((), device=device, dtype=loss.dtype)
        teacher_attn_mean_scalar = torch.zeros((), device=device, dtype=loss.dtype)

        # if not distill_on:
        #     print(f"⚠️ [WARNING] DISTILL IS OFF! coeff={self.distill_coeff}, target_numel={batch.teacher_hidden_targets.numel()}")
        # else:
            # print(f"🔥 [SUCCESS] DISTILL IS ON! Ready for MSE Nuke!")

        if distill_on:
            target_layer = int(batch.teacher_hidden_target_layer.item())
            if target_layer < 0:
                raise ValueError("teacher_hidden_target_layer is missing; please load teacher hidden distill pack.")
            student_h = out.hidden_states[target_layer]
            
            # 获取老师离线池化好的 V 的标准答案
            teacher_v_targets = batch.teacher_hidden_targets.to(device=device, dtype=loss.dtype)
            valid_v_lens = batch.valid_v_lens.to(device)
            num_v = int(self.iron.javis.num_queries)
            
            bsz = student_h.size(0)
            mse_losses = []
            
            for b in range(bsz):
                v_len = int(valid_v_lens[b].item())
                if v_len <= 0:
                    continue

                starts = mem_pos[b]
                valid_starts = starts[starts >= 0]
                if valid_starts.numel() == 0:
                    continue

                slot_positions = (valid_starts.view(-1, 1) + torch.arange(num_v, device=device).view(1, -1)).reshape(-1)
                v_len = min(v_len, int(slot_positions.numel()), int(teacher_v_targets.size(1)))
                if v_len <= 0:
                    continue

                student_v_feats = student_h[b].index_select(0, slot_positions[:v_len])
                teacher_v_feats = teacher_v_targets[b, :v_len, :]
                mse_losses.append(F.mse_loss(student_v_feats, teacher_v_feats))
            
            if mse_losses:
                distill_loss = torch.stack(mse_losses).mean()
                loss = loss + self.distill_coeff * distill_loss

        # (其余部分保持不变)
        current_out_cos = current_out_cos.to(dtype=loss.dtype)
        ortho_penalty = torch.relu(current_out_cos - 0.1)
        if self.javis_q_cos_coeff != 0.0:
            loss = loss + self.javis_q_cos_coeff * ortho_penalty

        if return_metrics:
            return loss, l2_loss, distill_loss, student_attn_mean_scalar, teacher_attn_mean_scalar, javis_metrics, current_out_cos
        return loss, l2_loss, distill_loss, current_out_cos

def set_phase(model: IronCellModel, phase: str) -> None:
    print(f"--> Setting model to {phase} mode...")
    if phase == "phase1":
        model.freeze_for_phase_1()
        # 双重保险：确保 config 状态正确
        model.config.freeze_compressor = True
        # 手动冻结 Compressor
        for p in model.compressor.parameters():
            p.requires_grad = False
    elif phase == "phase2":
        model.config.freeze_compressor = False
        # 全量解冻
        for p in model.parameters():
            p.requires_grad = True

        # hack for cmp
        model.config.freeze_compressor = True
        for p in model.compressor.parameters():
            p.requires_grad = False
    elif phase == "phase3" or phase == "phase-full":
        model.config.freeze_compressor = False
        # 全量解冻
        for p in model.parameters():
            p.requires_grad = True
    elif phase == "phase-cmp":
        model.config.freeze_compressor = False
        model.freeze_for_phase_1()
        for p in model.compressor.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown phase: {phase}")

def main() -> None:
    # args = TrainArgs()
    parser = HfArgumentParser((TrainArgs,))

    if len(os.sys.argv) == 2 and os.sys.argv[1].endswith(".json"):
        # 支持直接传 json 配置文件
        args = parser.parse_json_file(json_file=os.path.abspath(os.sys.argv[1]))[0]
    else:
        # 支持命令行参数
        args = parser.parse_args_into_dataclasses()[0]
    
    print(args)
    parallel = str(args.parallel).lower()
    use_ddp = parallel == "ddp"
    use_fsdp = parallel == "fsdp"
    use_dist = use_ddp or use_fsdp
    if use_dist:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(seconds=7200))
        device = torch.device("cuda", local_rank)
        is_rank0 = int(os.environ.get("RANK", "0")) == 0
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        is_rank0 = True

    if is_rank0:
        run_name = args.wandb_run_name or f"run-{args.phase}-"
        tags = None
        if args.wandb_run_tags is not None and str(args.wandb_run_tags).strip() != "":
            tags = [t.strip() for t in str(args.wandb_run_tags).split(",") if t.strip()]
        wandb.init(project=str(args.wandb_project), name=run_name, tags=tags, config=args)

    tokenizer, is_resume = load_tokenizer(args)
    model = load_model(args, tokenizer, device, is_resume=is_resume)

    # 3. 优化设置
    # model.generator.config.use_cache = False
    # model.generator.gradient_checkpointing_enable()
    
    # if args.phase in ["phase2", "phase3", "phase-cmp", "phase-full"]:
    #     print(f"--> Enabling Gradient Checkpointing for Compressor (Phase: {args.phase})...")
    #     model.compressor.gradient_checkpointing_enable()

    # set_phase(model, args.phase)
    # configure_special_embedding_mode(args, model, tokenizer, is_resume=is_resume)

    # 4. 数据加载 (替换 toy text)
    # if not os.path.exists(args.data_path):
    #     raise FileNotFoundError(f"Data file not found at {args.data_path}. Please run prepare_data.py first.")
        
    # dataset = JsonlDataset(args.data_path)
    # teacher_targets = None
    # if args.teacher_targets_path is not None and str(args.teacher_targets_path).strip() != "":
    #     if not os.path.exists(args.teacher_targets_path):
    #         raise FileNotFoundError(f"Teacher targets not found at {args.teacher_targets_path}")
    #     teacher_pack = torch.load(args.teacher_targets_path, map_location="cpu")
    #     if isinstance(teacher_pack, dict) and "targets" in teacher_pack:
    #         teacher_targets = teacher_pack["targets"]
    #     else:
    #         teacher_targets = teacher_pack

    # teacher_hidden_targets = None
    # teacher_hidden_valid_lens = None
    # teacher_hidden_target_layer = None
    # if args.teacher_hidden_targets_path is not None and str(args.teacher_hidden_targets_path).strip() != "":
    #     if not os.path.exists(args.teacher_hidden_targets_path):
    #         raise FileNotFoundError(f"Teacher hidden targets not found at {args.teacher_hidden_targets_path}")
    #     print(f"--> [System] Memory-Mapping massive hidden targets from {args.teacher_hidden_targets_path}...")
    #     hidden_pack = torch.load(
    #         args.teacher_hidden_targets_path, 
    #         map_location="cpu", 
    #         mmap=True  
    #     )

        # print(f"--> [Done]  hidden targets from {args.teacher_hidden_targets_path}...")
        
        # if not (isinstance(hidden_pack, dict) and "targets" in hidden_pack and "valid_v_lens" in hidden_pack):
        #     raise ValueError("teacher_hidden_targets_path must point to a dict with keys: targets, valid_v_lens, target_layer, chunk_size, q_num.")
        # teacher_hidden_targets = hidden_pack["targets"]
        # teacher_hidden_valid_lens = hidden_pack["valid_v_lens"]
        # teacher_hidden_target_layer = int(hidden_pack.get("target_layer", -1))
        # if teacher_hidden_target_layer < 0:
        #     raise ValueError("Hidden distill pack missing target_layer.")
        # pack_chunk_size = int(hidden_pack.get("chunk_size", -1))
        # if pack_chunk_size > 0 and int(pack_chunk_size) != int(args.chunk_size):
        #     raise ValueError(f"Hidden distill pack chunk_size={pack_chunk_size} mismatches args.chunk_size={args.chunk_size}.")
        # pack_q_num = int(hidden_pack.get("q_num", -1))
        # if pack_q_num > 0 and int(pack_q_num) != int(args.javis_num_queries):
        #     raise ValueError(f"Hidden distill pack q_num={pack_q_num} mismatches args.javis_num_queries={args.javis_num_queries}.")
    # print(f"collator numv {args.javis_num_queries=}")
    # collator = IronCellCollator(
    #     tokenizer,
    #     chunk_size=args.chunk_size,
    #     num_v=args.javis_num_queries,
    #     random_gate=args.random_gate,
    #     teacher_targets=teacher_targets,
    #     teacher_hidden_targets=teacher_hidden_targets,
    #     teacher_hidden_valid_lens=teacher_hidden_valid_lens,
    #     teacher_hidden_target_layer=teacher_hidden_target_layer,
    # )

    collator_eval = IronCellCollator(
        tokenizer,
        chunk_size=args.chunk_size,
        num_v=args.javis_num_queries,
        random_gate=args.random_gate,
        # teacher_targets=teacher_targets,
        # teacher_hidden_targets=teacher_hidden_targets,
        # teacher_hidden_valid_lens=teacher_hidden_valid_lens,
        # teacher_hidden_target_layer=teacher_hidden_target_layer,
    )
    # if use_dist:
    #     sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
    #     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler, num_workers=0, collate_fn=collator)
    # else:
    #     sampler = None
    #     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    eval_loader = None
    eval_sampler = None
    if args.eval_data_path is not None and str(args.eval_data_path).strip() != "":
        if not os.path.exists(args.eval_data_path):
            raise FileNotFoundError(f"Eval file not found at {args.eval_data_path}")
        eval_dataset = JsonlDataset(args.eval_data_path)
        if use_dist:
            eval_sampler = DistributedSampler(eval_dataset, shuffle=False, drop_last=False)
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=eval_sampler,
                num_workers=0,
                collate_fn=collator_eval,
            )
        else:
            eval_loader = DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collator_eval,
            )

    # if args.phase == "phase1" and not is_resume:
    #     warmup_init_javis_query(
    #         model,
    #         loader,
    #         num_samples=getattr(model.config, "javis_query_warmup_samples", None),
    #         save_path=getattr(model.config, "javis_query_warmup_save_path", None),
    #         use_dist=use_dist,
    #     )

    print("Starting Eval...")
    model.eval()
    total_micro_loss = 0.0
    total_micro_q_cos = 0.0
    total_micro_distill = 0.0
    total_micro_student_attn = 0.0
    total_micro_teacher_attn = 0.0

    micro_count = 0
    
    distill_layers = _parse_layers(args.distill_layers)
    step_module = TrainStepModule(
        model,
        phase=args.phase,
        javis_q_cos_coeff=args.javis_q_cos_coeff,
        distill_coeff=args.distill_coeff,
        distill_layers=distill_layers,
        chunk_size=args.chunk_size,
    )
    step_module.grad_probe = bool(args.grad_probe)
    if use_fsdp:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        cpu_offload = CPUOffload(offload_params=True) if bool(args.fsdp_cpu_offload) else None
        auto_wrap_policy = _build_fsdp_auto_wrap_policy(model, args.fsdp_wrap)
        step_module = FSDP(
            step_module,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp,
            cpu_offload=cpu_offload,
            use_orig_params=bool(args.fsdp_use_orig_params),
            device_id=device,
            sync_module_states=True,
        )
    elif use_ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        step_module = DDP(
            step_module,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
        )

    step_impl = step_module.module if use_dist else step_module
    iron = step_impl.iron

    # lr_projector = float(args.lr_projector) if args.lr_projector is not None else float(args.lr)
    # lr_generator = float(args.lr_generator) if args.lr_generator is not None else float(args.lr)
    # lr_compressor = float(args.lr_compressor) if args.lr_compressor is not None else float(args.lr)

    seen: set[int] = set()

    def _append_param_groups(named_params, *, prefix: str, lr: float) -> list[dict]:
        decay = []
        no_decay = []
        for name, p in named_params:
            if not p.requires_grad:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            full_name = f"{prefix}.{name}" if prefix else str(name)
            if _is_no_weight_decay_param(full_name):
                no_decay.append(p)
            else:
                decay.append(p)

        out = []
        wd = float(args.weight_decay)
        if decay:
            out.append({"params": decay, "lr": float(lr), "weight_decay": wd})
        if no_decay:
            out.append({"params": no_decay, "lr": float(lr), "weight_decay": 0.0})
        return out

    # param_groups = []
    # param_groups.extend(_append_param_groups(iron.javis.named_parameters(), prefix="javis", lr=lr_projector))
    # param_groups.extend(
    #     _append_param_groups(iron.special_token_embeddings.named_parameters(), prefix="special_token_embeddings", lr=lr_projector)
    # )
    # param_groups.extend(_append_param_groups(iron.generator.named_parameters(), prefix="generator", lr=lr_generator))
    # param_groups.extend(_append_param_groups(iron.compressor.named_parameters(), prefix="compressor", lr=lr_compressor))

    # if not param_groups:
    #     raise ValueError("No trainable parameters found (all requires_grad=False).")

    # optimizer = torch.optim.AdamW(param_groups, lr=float(args.lr), weight_decay=0.0)

    # if args.resume_path:
    #     loaded_step = load_checkpoint(optimizer, args)
    # else:
    #     loaded_step = 0

    # if bool(args.reset_step_on_resume):
    #     step = 0
    # else:
    #     step = int(loaded_step)

    # base_lrs = [float(g.get("lr", float(args.lr))) for g in optimizer.param_groups]

    # 6. 训练循环
    # 使用 iter(loader) 配合 while 循环可以防止 epoch 结束导致的重置，或者直接用 cycle
    # epoch = args.global_epoch
    # if sampler is not None:
    #     sampler.set_epoch(epoch)
    # data_iter = iter(loader)
    
    # grad_accum_steps = max(1, int(args.grad_accum_steps))
    # eval_steps = max(0, int(args.eval_steps))
    # eval_max_batches = max(0, int(args.eval_max_batches))
    if eval_sampler is not None:
        eval_sampler.set_epoch(0)

    def _run_eval() -> float | None:
        if eval_loader is None:
            return None

        step_module.eval()
        sum_loss = torch.zeros((), device=device, dtype=torch.float32)
        count = torch.zeros((), device=device, dtype=torch.float32)
        with torch.no_grad():
            for i, batch in enumerate(eval_loader):
                # if eval_max_batches > 0 and i >= eval_max_batches:
                #     break
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    loss, _, _, _ = step_module(batch, return_metrics=False)
                sum_loss += loss.detach().float()
                count += 1.0

        if use_dist:
            torch.distributed.all_reduce(sum_loss, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)

        step_module.train()

        denom = float(count.clamp(min=1.0).item())
        return float((sum_loss / denom).item())

    eval_loss = _run_eval()
    if eval_loss is not None and is_rank0:
        print(f"{'='*20}\n | EvalLoss: {eval_loss:.4f}  \n{'='*20}")
        print("Eval finished.")
    while 0:
    # while step < args.steps:
        warmup_steps = max(0, int(args.warmup_steps))
        if warmup_steps > 0 and step < warmup_steps:
            warmup_factor = float(step + 1) / float(warmup_steps)
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = float(base_lr) * warmup_factor
        else:
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = float(base_lr)

        optimizer.zero_grad(set_to_none=True)
        last_l2_loss = torch.zeros((), device=device)
        if step_impl.grad_probe:
            step_impl.reset_grad_probe()

        javis_metrics = None
        for micro in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iter = iter(loader)  # 新的 Epoch
                batch = next(data_iter)

            if use_dist and micro < grad_accum_steps - 1:
                sync_ctx = step_module.no_sync()
            else:
                sync_ctx = torch.enable_grad()

            with sync_ctx:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
                    want_metrics = step % args.log_steps == 0 and micro == grad_accum_steps - 1
                    if want_metrics:
                        loss, l2_loss, distill_loss, student_attn_mean, teacher_attn_mean, javis_metrics, current_out_cos = step_module(batch, return_metrics=True)
                    else:
                        loss, l2_loss, distill_loss, current_out_cos = step_module(batch, return_metrics=False)
                    last_l2_loss = l2_loss
                    (loss / grad_accum_steps).backward()

            total_micro_loss += float(loss.item())
            total_micro_q_cos += float(current_out_cos.item())
            total_micro_distill += float(distill_loss.item())
            if want_metrics:
                total_micro_student_attn += float(student_attn_mean.item())
                total_micro_teacher_attn += float(teacher_attn_mean.item())
            micro_count += 1

        if step % args.log_steps == 0:
            gen_grad_norm = torch.zeros((), device=device)
            for name, p in iron.generator.named_parameters():
                if "layers.0.self_attn.q_proj" in name and p.grad is not None:
                    gen_grad_norm = (p.grad.float() ** 2).sum()
                    break

            javis_grad_norm = torch.zeros((), device=device)
            if iron.javis.q.grad is not None:
                javis_grad_norm = (iron.javis.q.grad.float() ** 2).sum()

            cmp_grad_norm = torch.zeros((), device=device)
            for name, p in iron.compressor.named_parameters():
                if "layers.31.self_attn.o_proj" in name and p.grad is not None:
                    cmp_grad_norm = (p.grad.float() ** 2).sum()
                    break

            if use_dist and torch.distributed.is_initialized():
                torch.distributed.all_reduce(gen_grad_norm, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(javis_grad_norm, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(cmp_grad_norm, op=torch.distributed.ReduceOp.SUM)

            # if is_rank0:
            #     gen_grad = gen_grad_norm.sqrt().item()
            #     javis_grad = javis_grad_norm.sqrt().item()
            #     cmp_grad = cmp_grad_norm.sqrt().item()
            #     print(
            #         f"  --> [Gradient X-Ray] GEN_L0_q: {gen_grad:.4f} | JAVIS_q: {javis_grad:.4f} | CMP_L31_o: {cmp_grad:.4f}"
            #     )
            if step_impl.grad_probe:
                names = ["memory_vectors", "inputs_embeds", "logits"]
                probe_vals = [step_impl.grad_probe_sums.get(n, torch.zeros((), device=device)) for n in names]
                if use_dist and torch.distributed.is_initialized():
                    for i in range(len(probe_vals)):
                        torch.distributed.all_reduce(probe_vals[i], op=torch.distributed.ReduceOp.SUM)
                if is_rank0:
                    out_vals = [v.sqrt().item() for v in probe_vals]
                    msg = " | ".join(f"{n}: {v:.4f}" for n, v in zip(names, out_vals))
                    print(f"  --> [Grad Probe] {msg}")

        if use_fsdp:
            grad_norm = FSDP.clip_grad_norm_(step_module, 1.0)
        else:
            grad_norm = clip_grad_norm_(step_module.parameters(), 1.0)

        optimizer.step()

        

        eval_loss = _run_eval(step=step)
        if eval_loss is not None and is_rank0:
            print(f"[{args.phase}] Step {step} | EvalLoss: {eval_loss:.4f}")
            wandb.log({"eval/loss": eval_loss}, step=step)

        # [Logging]
        if step % args.log_steps == 0:
            denom = max(1, micro_count)
            loss_micro_avg = total_micro_loss / denom
            javis_q_cos_micro_avg = total_micro_q_cos / denom
            distill_micro_avg = total_micro_distill / denom
            student_attn_avg = total_micro_student_attn / denom
            teacher_attn_avg = total_micro_teacher_attn / denom
            print(
                f"[{args.phase}] Step {step} | Loss: {loss_micro_avg:.4f} | Distill: {distill_micro_avg:.4f} | JavisQCos: {javis_q_cos_micro_avg:.4f} | GradNorm: {grad_norm:.4f}"
            )
            if javis_metrics is not None and is_rank0:
                metrics_msg = " | ".join(f"{k}: {v:.4f}" for k, v in javis_metrics.items())
                if metrics_msg:
                    print(f"  --> [Javis Metrics] {metrics_msg}")
            
            if is_rank0:
                lrs = [float(g.get("lr", 0.0)) for g in optimizer.param_groups]
                log_dict = {
                    "loss": loss_micro_avg,
                    "distill_loss": distill_micro_avg,
                    "attn_student_mean": student_attn_avg,
                    "attn_teacher_mean": teacher_attn_avg,
                    "javis_q_cos": javis_q_cos_micro_avg,
                    "l2_reg": last_l2_loss.item(),
                    "grad_norm": grad_norm,
                    "lr": float(args.lr),
                    "lr_group0": lrs[0] if len(lrs) > 0 else 0.0,
                    "lr_group1": lrs[1] if len(lrs) > 1 else 0.0,
                    "lr_group2": lrs[2] if len(lrs) > 2 else 0.0,
                }
                if javis_metrics is not None:
                    log_dict.update(javis_metrics)
                javis_grad_cos = getattr(iron.javis, "current_q_grad_cos", None)
                if javis_grad_cos is not None:
                    log_dict["javis_q_grad_cos"] = float(javis_grad_cos)
                wandb.log(log_dict, step=step)
            total_micro_loss = 0.0
            total_micro_q_cos = 0.0
            total_micro_distill = 0.0
            total_micro_student_attn = 0.0
            total_micro_teacher_attn = 0.0
            micro_count = 0

        step += 1
        # [Checkpointing]
        if step % args.save_steps == 0:
            if use_fsdp:
                save_checkpoint_fsdp(step_module, optimizer, tokenizer, args, step, is_rank0=is_rank0)
            else:
                if is_rank0:
                    to_save = step_module.module.iron if use_ddp else step_module.iron
                    save_checkpoint(to_save, optimizer, tokenizer, args, step)

    # print("Eval finished.")
    if is_rank0:
        wandb.finish()

if __name__ == "__main__":
    main()
