import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from pathlib import Path

# 导入你自己的模块
from src.data_processor import IronCellCollator
from src.models import IronCellModel


import math

import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaDecoderLayer



@dataclass
class EvalArgs:
    eval_mode: str = field(
        default="mark42", 
        metadata={"help": "评估模式: 'oracle', 'amnesiac', 'mark42'"}
    )
    model_name: str = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={"help": "HuggingFace 基础模型路径或名称"}
    )
    resume_path: str = field(
        default="",
        metadata={"help": "Mark-42 训练好的 checkpoint 路径"}
    )
    eval_data_path: str = field(
        default="../data/wikitext_8k_eval.jsonl",
        metadata={"help": "测试数据路径"}
    )
    output_dir: str = field(
        default="./eval_results",
        metadata={"help": "结果保存目录"}
    )
    chunk_size: int = field(default=16)
    javis_num_queries: int = field(default=2)
    max_tokens: int = field(default=8192)



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

    is_target_layer = False
    if is_target_layer and javis_kv is not None and javis_meta is not None:
        print(f"================={layer_idx=} ")
        # print(f"xxxxxxxxxxx {javis_kv.shape=}")
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
    if javis_kv is not None and javis_meta is not None and is_target_layer:
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




@torch.no_grad()
def main():
    # 解析 Shell 传进来的参数
    parser = HfArgumentParser((EvalArgs,))
    # ⚠️ 加上 return_remaining_strings 防止未知参数报错
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # 🚀 核心修复：根据模式动态选择 Tokenizer 路径

    is_resume = bool(args.resume_path and os.path.exists(args.resume_path))
    print(f"Resume Path: {args.resume_path}  is exists: {is_resume}")
    if args.eval_mode == "mark42" and is_resume:
        tok_path = args.resume_path
        print(f"Loading Custom Tokenizer (with special tokens) from {tok_path}...")
    else:
        tok_path = args.model_name
        print(f"Loading Native Tokenizer from {tok_path}...")
        
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    # 防御性设置：确保有 pad 和 bos (对齐你训练时的逻辑)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    # 读取你精心准备的那条 8K 长文本
    print(f"Loading long text from {args.eval_data_path}...")
    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        # 读取第一行 JSON
        data = json.loads(f.readline())
        long_text = data["text"]

    # ==========================================
    # 模式 A: Oracle (原生模型全血前向)
    # ==========================================
    if args.eval_mode == "oracle":
        print("\n>>> Running Oracle Baseline (Full 8K Context)...")
        raw_input_ids = tokenizer(long_text, return_tensors="pt").input_ids[:, :args.max_tokens].to(device)
        seq_len = raw_input_ids.size(1)
        
        native_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
        native_model.eval()
        
        outputs = native_model(raw_input_ids)
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = raw_input_ids[0, 1:]
        
        losses = []
        num_chunks = seq_len // args.chunk_size
        for k in range(1, num_chunks):
            start = k * args.chunk_size - 1
            end = (k + 1) * args.chunk_size - 1
            loss = F.cross_entropy(shift_logits[start:end], shift_labels[start:end])
            losses.append(loss.item())
            
        _save_results(losses, args.output_dir, "oracle")

    # ==========================================
    # 模式 B: Amnesiac (原生模型滑动窗口)
    # ==========================================
    elif args.eval_mode == "amnesiac":
        print("\n>>> Running Amnesiac Baseline (Sliding Window)...")
        raw_input_ids = tokenizer(long_text, return_tensors="pt").input_ids[:, :args.max_tokens].to(device)
        seq_len = raw_input_ids.size(1)
        bos_id = raw_input_ids[:, 0:1]
        
        native_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
        native_model.eval()
        
        losses = []
        num_chunks = seq_len // args.chunk_size
        for k in range(1, num_chunks):
            start_raw = (k - 1) * args.chunk_size
            end_raw = (k + 1) * args.chunk_size
            
            prev_chunk = raw_input_ids[:, start_raw : start_raw + args.chunk_size]
            curr_chunk = raw_input_ids[:, start_raw + args.chunk_size : end_raw]
            input_chunk = torch.cat([bos_id, prev_chunk, curr_chunk], dim=1)
            
            outputs = native_model(input_ids=input_chunk)
            shift_logits = outputs.logits[0, :-1, :]
            shift_labels = input_chunk[0, 1:]
            
            curr_logits = shift_logits[-args.chunk_size:]
            curr_labels = shift_labels[-args.chunk_size:]
            loss = F.cross_entropy(curr_logits, curr_labels)
            losses.append(loss.item())
            
        _save_results(losses, args.output_dir, "amnesiac")

    
    elif args.eval_mode == "mark42":
        print(f"\n>>> Running IronCell Mark-42...")
        import types # 用于动态绑定方法

        # 1. 组装专用的 Collator
        collator = IronCellCollator(
            tokenizer=tokenizer,
            chunk_size=args.chunk_size,
            num_v=args.javis_num_queries,
            truncate_len=args.max_tokens,
            random_gate=1
        )
        batch = collator([long_text])
        for k, v in batch.__dict__.items():
            if isinstance(v, torch.Tensor): 
                setattr(batch, k, v.to(device))

        # 2. 加载模型与配置
        is_resume = bool(args.resume_path and os.path.exists(args.resume_path))
        if is_resume:
            from src.models import IronCellConfig
            print(f"Loading config and weights from {args.resume_path}...")
            config = IronCellConfig.from_pretrained(args.resume_path)
            setattr(config, "tokenizer_vocab_size", len(tokenizer))
            
            model = IronCellModel.from_pretrained(
                args.resume_path, 
                config=config,
                torch_dtype=torch.bfloat16
            ).to(device)
        else:
            print(f"⚠️ [警告] 没有提供 resume_path，将从基座 {args.model_name} 加载未经训练的 Mark-42 模型！")
            model = IronCellModel.from_pretrained(
                args.model_name,
                torch_dtype=torch.bfloat16
            ).to(device)

        model.eval()

        # =========================================================
        # 🚀 极其关键：Monkey Patch 劫持注意力层！对齐训练环境
        # =========================================================
        # 请确保能从你存放该函数的地方正确 import，这里假设它在 src.hack_llama_ddp 中
        # try:
        #     from src.hack_llama_fsdp import smart_hybrid_attention_forward
        # except ImportError:
        #     # 如果路径不对，请修改上面这行 import
        #     raise ImportError("无法导入 smart_hybrid_attention_forward，请检查路径！")

        print("Patching Llama Attention for Deep KV Injection...")
        for layer in model.generator.model.layers:
            layer.self_attn.forward = types.MethodType(smart_hybrid_attention_forward, layer.self_attn)
        # =========================================================

        print("Running Forward Pass (Compressor -> Embeddings -> Generator)...")
        
        # 3. 严格对齐训练时的三步走 Forward 逻辑
        print(f" batch.chunk_input_ids  {batch.chunk_input_ids.shape}")
        print(f" batch.zipper_input_ids  {batch.zipper_input_ids.shape}")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            # 步骤 A: 压缩 (提取 memory_vectors 和 深层 KV)
            memory_out = model.compute_compressed_vectors(
                chunk_input_ids=batch.chunk_input_ids,
                chunk_attention_mask=batch.chunk_attention_mask,
                return_metrics=False
            )
            memory_hook, memory_vectors, deep_layer_kvs, current_out_cos = memory_out

            # 步骤 B: 拼装 Embeddings (插入占位符)
            inputs_embeds = model.build_inputs_embeds(
                zipper_input_ids=batch.zipper_input_ids,
                memory_vectors=memory_vectors, 
                memory_positions=batch.memory_positions,
            )

            # 步骤 C: 最终前向传播 (注入 javis 参数)
            
            outputs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=batch.attention_mask_2d,
                position_ids=batch.position_ids,
                use_cache=False,
                javis_all_layer_kvs=deep_layer_kvs, 
                javis_meta=(batch.memory_positions, model.javis.num_queries)
            )
        
        # 4. 计算 Chunk Loss (用来画长文衰减图)
        logits = outputs.logits[0, :-1, :]
        labels = batch.labels[0, 1:]
        prefix_len = batch.prefix_lens[0].item()
        
        losses = []
        num_chunks = (batch.valid_lens[0] - prefix_len) // args.chunk_size
        for k in range(1, num_chunks):
            s = prefix_len + k * args.chunk_size - 1
            e = prefix_len + (k + 1) * args.chunk_size - 1
            
            # 使用 ignore_index=-100 自动过滤掉你的 <soc>, <eoc> 等填充
            loss = F.cross_entropy(logits[s:e], labels[s:e], ignore_index=-100)
            losses.append(loss.item())
            
        _save_results(losses, args.output_dir, "mark42")
    else:
        raise ValueError(f"未知的评估模式: {args.eval_mode}")

# 辅助函数：将跑出来的 Loss 存为 JSON
def _save_results(losses, output_dir, mode_name):
    out_path = os.path.join(output_dir, f"{mode_name}_losses.json")
    with open(out_path, "w") as f:
        json.dump(losses, f)
    print(f"[Done] {mode_name} evaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()