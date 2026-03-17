import os
import json
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from tqdm import tqdm  # 加入进度条缓解焦虑

# 导入你自己的模块
from src.modeling_iron_cell import IronCellModel

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
    buffer_size: int = field(default=1)

# ==========================================
# 🚀 核心评估逻辑：绝对真实的流式物理模拟 (带 1 个 Chunk Buffer)
# ==========================================
@torch.no_grad()
def _evaluate_mark42_sequentially(iron_model, tokenizer, raw_input_ids, chunk_size=32, num_v=2):
    device = raw_input_ids.device
    seq_len = raw_input_ids.size(1)
    num_chunks = seq_len // chunk_size
    
    # 精准提取特殊 Token
    bos_id = raw_input_ids[:, 0:1] 
    soc_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<soc>")]], device=device)
    eoc_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<eoc>")]], device=device)
    v_none_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<v_none>")]], device=device) 
    
    mark42_losses = []
    
    # 模拟真实的时间步推移
    for k in tqdm(range(1, num_chunks), desc="Streaming Eval"):
        # 此时：Chunk k-1 是高清明文 Buffer，Chunk 0 到 k-2 是被压缩的 V 向量
        num_compressed = k - 1 
        
        # =========================================================
        # 1. 独立运行 Compressor 提取历史 V 向量 (动态生长)
        # =========================================================
        if num_compressed > 0:
            chunk_input_ids = raw_input_ids[:, :num_compressed * chunk_size].view(1, num_compressed, chunk_size)
            chunk_attention_mask = torch.ones_like(chunk_input_ids)
            
            _, memory_vectors, deep_layer_kvs, _ = iron_model.compute_compressed_vectors(
                chunk_input_ids=chunk_input_ids,
                chunk_attention_mask=chunk_attention_mask,
                return_metrics=False
            )
            # 严格计算占位符在 Prefix 里的位置索引
            memory_positions = torch.tensor([[2 + (num_v + 1) * i for i in range(num_compressed)]], device=device)
        else:
            # 🚀 修复 Bug: 使用 generator.config.hidden_size
            hidden_size = iron_model.generator.config.hidden_size
            memory_vectors = torch.empty((1, 0, hidden_size), device=device, dtype=torch.bfloat16)
            deep_layer_kvs = None
            memory_positions = torch.empty((1, 0), dtype=torch.long, device=device)

        # =========================================================
        # 2. 组装当前时间步 k 的【完整物理序列】
        # =========================================================
        prefix_ids = [bos_id, soc_id]
        for _ in range(num_compressed):
            for _ in range(num_v): prefix_ids.append(v_none_id)
            prefix_ids.append(eoc_id)
        prefix_tensor = torch.cat(prefix_ids, dim=1)
        
        # 核心架构体现：Buffer (Chunk k-1) + Current (Chunk k)
        start_raw = (k - 1) * chunk_size
        end_raw = (k + 1) * chunk_size
        prev_chunk = raw_input_ids[:, start_raw : start_raw + chunk_size]
        curr_chunk = raw_input_ids[:, start_raw + chunk_size : end_raw]
        
        current_input_ids = torch.cat([prefix_tensor, prev_chunk, curr_chunk], dim=1)
        current_seq_len = current_input_ids.size(1)
        
        # =========================================================
        # 3. 生成绝对连续的 Pos IDs 和 Causal Mask (解决 RoPE 漂移)
        # =========================================================
        position_ids = torch.arange(current_seq_len, dtype=torch.long, device=device).unsqueeze(0)
        
        # 生成标准的下三角因果掩码 [1, seq_len, seq_len]
        attn_mask = torch.tril(torch.ones((1, current_seq_len, current_seq_len), dtype=torch.bool, device=device))
        
        # =========================================================
        # 4. 前向传播
        # =========================================================
        inputs_embeds = iron_model.build_inputs_embeds(
            zipper_input_ids=current_input_ids,
            memory_vectors=memory_vectors, 
            memory_positions=memory_positions,
        )
        
        # 这里会利用你魔改的 HF 源码，把 javis_all_layer_kvs 精准发给每一层
        outputs = iron_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            use_cache=False,
            javis_all_layer_kvs=deep_layer_kvs, 
            javis_meta=(memory_positions, num_v)
        )
        
        # =========================================================
        # 5. 计算当前 Chunk 的精准 Loss
        # =========================================================
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = current_input_ids[0, 1:]
        
        # 严格截取最后 chunk_size 个 token 作为评测目标 (只评测 Current Chunk)
        curr_logits = shift_logits[-chunk_size:]
        curr_labels = shift_labels[-chunk_size:]
        
        loss = F.cross_entropy(curr_logits, curr_labels)
        mark42_losses.append(loss.item())
        
    return mark42_losses



@torch.no_grad()
def _evaluate_mark42_compact_sequential(iron_model, tokenizer, raw_input_ids, args):
    device = raw_input_ids.device
    chunk_size = args.chunk_size
    num_v = args.javis_num_queries
    num_chunks = raw_input_ids.size(1) // chunk_size
    
    bos_id = torch.tensor([[tokenizer.bos_token_id]], device=device)
    soc_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<soc>")]], device=device)
    eoc_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<eoc>")]], device=device)
    v_none_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<v_none>")]], device=device)

    mark42_losses = []

    for k in tqdm(range(1, num_chunks), desc="Sequential Compact Eval"):
        num_compressed = k - 1 # 已压缩块的数量 (v0...vk-2)

        # =========================================================
        # 1. 构建物理紧凑序列 (BOS + SOC + [V_-1...V_{k-2}] + EOC + Raw_{k-1} + Raw_k)
        # =========================================================
        current_ids = [bos_id, soc_id]
        
        # 🚀 聚簇所有 V 占位符 (共 k 组：Group -1 和 Groups 0...k-2)
        for _ in range(k):
            for _ in range(num_v):
                current_ids.append(v_none_id)
        
        # 🚀 全局唯一的 EOC，作为历史与现实的分界线
        current_ids.append(eoc_id)
            
        # 拼接 Buffer 和 当前待测块
        start_raw = (k - 1) * chunk_size
        end_raw = (k + 1) * chunk_size
        current_ids.append(raw_input_ids[:, start_raw : end_raw])
        
        input_ids = torch.cat(current_ids, dim=1)
        curr_len = input_ids.size(1)

        # =========================================================
        # 2. 原生位置分配
        # =========================================================
        # 紧凑序列下，arange 就是最真实的 RoPE 相对距离
        position_ids = torch.arange(curr_len, device=device).unsqueeze(0)
        # 1. 创建全 0.0 的浮点矩阵 (形状 [1, 1, Seq, Seq])
        attn_mask = torch.zeros((1, 1, curr_len, curr_len), device=device, dtype=iron_model.dtype)
        
        # 2. 获取上三角布尔矩阵 (diagonal=1 表示不包含对角线，纯粹代表"未来")
        future_mask = torch.triu(torch.ones((curr_len, curr_len), dtype=torch.bool, device=device), diagonal=1)
        
        # 3. 将"未来"区域狠狠地填上负无穷！锁死泄露！
        attn_mask.masked_fill_(future_mask, float("-inf"))

        # =========================================================
        # 3. 注入位置对齐 (聚簇索引公式)
        # =========================================================
        if num_compressed > 0:
            hist_chunks = raw_input_ids[:, :num_compressed * chunk_size].view(1, num_compressed, chunk_size)
            _, memory_vectors, deep_layer_kvs, _ = iron_model.compute_compressed_vectors(
                chunk_input_ids=hist_chunks,
                chunk_attention_mask=torch.ones_like(hist_chunks),
                return_metrics=False
            )
            
            # 🚀 精准注入点索引：
            # 0:BOS, 1:SOC
            # 2 ~ 2+num_v-1: Group -1 (v-1)
            # 2+num_v ~ 2+2*num_v-1: Group 0 (v0)
            # 所以 Group m (vm) 的起始位置 = 2 + (m + 1) * num_v
            active_mem_pos = torch.tensor([[
                2 + (m + 1) * num_v for m in range(num_compressed)
            ]], device=device)
        else:
            memory_vectors = torch.empty((1, 0, iron_model.generator.config.hidden_size), device=device)
            deep_layer_kvs = None
            active_mem_pos = torch.empty((1, 0), dtype=torch.long, device=device)

        # =========================================================
        # 4. Forward & Loss
        # =========================================================
        # print(f"  {input_ids.shape} {input_ids}")
        # print(f"  {memory_vectors.shape} ")
        # print(f"  {active_mem_pos.shape} {active_mem_pos}")
        
        inputs_embeds = iron_model.build_inputs_embeds(
            zipper_input_ids=input_ids,
            memory_vectors=memory_vectors, 
            memory_positions=active_mem_pos,
        )
        
        outputs = iron_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask, 
            position_ids=position_ids,
            use_cache=False,
            javis_all_layer_kvs=deep_layer_kvs, 
            javis_meta=(active_mem_pos, num_v)
        )
        
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        
        # 评测目标依然是序列最后 chunk_size 个词
        loss = F.cross_entropy(shift_logits[-chunk_size:], shift_labels[-chunk_size:])
        mark42_losses.append(loss.item())
        
    return mark42_losses


@torch.no_grad()
def evaluate_mark42_compact_sequential(iron_model, tokenizer, raw_input_ids, args):
    print(f"{args=}")
    device = raw_input_ids.device
    chunk_size = args.chunk_size
    num_v = args.javis_num_queries
    num_chunks = raw_input_ids.size(1) // chunk_size
    buffer_size = args.buffer_size
    
    bos_id = torch.tensor([[tokenizer.bos_token_id]], device=device)
    soc_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<soc>")]], device=device)
    eoc_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<eoc>")]], device=device)
    v_none_id = torch.tensor([[tokenizer.convert_tokens_to_ids("<v_none>")]], device=device)

    mark42_losses = []

    for k in tqdm(range(1, num_chunks), desc=f"Compact Eval (Buffer={buffer_size})"):
        # 🚀 核心改动 1：根据 buffer_size 动态计算被压缩的 Chunk 数量
        num_compressed = max(0, k - buffer_size)

        # =========================================================
        # 1. 构建物理紧凑序列
        # =========================================================
        current_ids = [bos_id, soc_id]
        
        # 需要的 V 组数量：1 (固定的 v_-1) + num_compressed
        num_groups = 1 + num_compressed
        for _ in range(num_groups):
            for _ in range(num_v):
                current_ids.append(v_none_id)
        current_ids.append(eoc_id)
            
        # 🚀 核心改动 2：拼接 Buffer (保留 k - buffer_size 到 k 的明文)
        start_raw = max(0, k - buffer_size) * chunk_size
        end_raw = (k + 1) * chunk_size
        current_ids.append(raw_input_ids[:, start_raw : end_raw])
        
        input_ids = torch.cat(current_ids, dim=1)
        curr_len = input_ids.size(1)

        # =========================================================
        # 2. 原生位置分配 & 强制浮点掩码
        # =========================================================
        position_ids = torch.arange(curr_len, device=device).unsqueeze(0)
        attn_mask = torch.zeros((1, 1, curr_len, curr_len), device=device, dtype=iron_model.dtype)
        future_mask = torch.triu(torch.ones((curr_len, curr_len), dtype=torch.bool, device=device), diagonal=1)
        attn_mask.masked_fill_(future_mask, float("-inf"))

        # =========================================================
        # 3. 注入位置对齐
        # =========================================================
        if num_compressed > 0:
            hist_chunks = raw_input_ids[:, :num_compressed * chunk_size].view(1, num_compressed, chunk_size)
            _, memory_vectors, deep_layer_kvs, _ = iron_model.compute_compressed_vectors(
                chunk_input_ids=hist_chunks,
                chunk_attention_mask=torch.ones_like(hist_chunks),
                return_metrics=False
            )
            
            # 索引公式在任何 buffer_size 下都保持不变
            active_mem_pos = torch.tensor([[
                2 + (m + 1) * num_v for m in range(num_compressed)
            ]], device=device)
        else:
            h_size = iron_model.config.hidden_size if hasattr(iron_model.config, 'hidden_size') else iron_model.generator.config.hidden_size
            memory_vectors = torch.empty((1, 0, h_size), device=device, dtype=iron_model.dtype)
            deep_layer_kvs = None
            active_mem_pos = torch.empty((1, 0), dtype=torch.long, device=device)

        # =========================================================
        # 4. Forward & Loss
        # =========================================================
        inputs_embeds = iron_model.build_inputs_embeds(
            zipper_input_ids=input_ids,
            memory_vectors=memory_vectors, 
            memory_positions=active_mem_pos,
        )
        
        outputs = iron_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask, 
            position_ids=position_ids,
            use_cache=False,
            javis_all_layer_kvs=deep_layer_kvs, 
            javis_meta=(active_mem_pos, num_v)
        )
        
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = input_ids[0, 1:]
        
        loss = F.cross_entropy(shift_logits[-chunk_size:], shift_labels[-chunk_size:])
        mark42_losses.append(loss.item())
        
    return mark42_losses

# ==========================================
# Main 流程
# ==========================================
@torch.no_grad()
def main():
    parser = HfArgumentParser((EvalArgs,))
    args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    is_resume = bool(args.resume_path and os.path.exists(args.resume_path))
    print(f"Resume Path: {args.resume_path}  is exists: {is_resume}")
    if args.eval_mode == "mark42" and is_resume:
        tok_path = args.resume_path
        print(f"Loading Custom Tokenizer (with special tokens) from {tok_path}...")
    else:
        tok_path = args.model_name
        print(f"Loading Native Tokenizer from {tok_path}...")
        
    tokenizer = AutoTokenizer.from_pretrained(tok_path)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token = tokenizer.eos_token

    print(f"Loading long text from {args.eval_data_path}...")
    with open(args.eval_data_path, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
        long_text = data["text"]

    # long_text = "我"*70
    raw_input_ids = tokenizer(long_text, add_special_tokens=False, return_tensors="pt").input_ids[:, :args.max_tokens].to(device)
    print(raw_input_ids.shape)
    
    seq_len = raw_input_ids.size(1)

    # ==========================================
    # 模式 A: Oracle (原生模型全血前向)
    # ==========================================
    if args.eval_mode == "oracle":
        print("\n>>> Running Oracle Baseline (Full 8K Context)...")
        native_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
        native_model.eval()
        
        outputs = native_model(raw_input_ids)
        shift_logits = outputs.logits[0, :-1, :]
        shift_labels = raw_input_ids[0, 1:]
        
        losses = []
        num_chunks = seq_len // args.chunk_size
        for k in tqdm(range(1, num_chunks), desc="Oracle Eval"):
            start = k * args.chunk_size - 1
            end = (k + 1) * args.chunk_size - 1
            loss = F.cross_entropy(shift_logits[start:end], shift_labels[start:end])
            losses.append(loss.item())
            
        _save_results(losses, args.output_dir, "oracle")

    # ==========================================
    # 模式 B: Amnesiac (原生模型滑动窗口 - 同样带 1 Chunk Buffer)
    # ==========================================
    
    elif args.eval_mode == "amnesiac":
        # 假设你在 args 里加了 buffer_size，或者在这里硬编码测试
        buffer_size = getattr(args, "buffer_size", 1) 
        print(f"\n>>> Running Amnesiac Baseline (Sliding Window with {buffer_size} Chunk Buffer)...")
        
        # 🚀 修复点 1：保证拿到绝对干净的 BOS
        bos_id = torch.tensor([[tokenizer.bos_token_id]], device=device)
        
        native_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)
        native_model.eval()
        
        losses = []
        num_chunks = seq_len // args.chunk_size
        for k in tqdm(range(1, num_chunks), desc=f"Amnesiac Eval (Buf={buffer_size})"):
            # 🚀 核心逻辑：动态计算滑动窗口的起点
            start_raw = max(0, k - buffer_size) * args.chunk_size
            end_raw = (k + 1) * args.chunk_size
            
            # Amnesiac 同样享有 k - buffer_size 到 k 的明文优势！绝对公平！
            raw_chunks = raw_input_ids[:, start_raw : end_raw]
            input_chunk = torch.cat([bos_id, raw_chunks], dim=1)
            
            outputs = native_model(input_ids=input_chunk)
            shift_logits = outputs.logits[0, :-1, :]
            shift_labels = input_chunk[0, 1:]
            
            curr_logits = shift_logits[-args.chunk_size:]
            curr_labels = shift_labels[-args.chunk_size:]
            loss = F.cross_entropy(curr_logits, curr_labels)
            losses.append(loss.item())
            
        _save_results(losses, args.output_dir, f"amnesiac_buf{buffer_size}")
    # ==========================================
    # 模式 C: Mark42 (完美的时序模拟流)
    # ==========================================
    elif args.eval_mode == "mark42":
        print(f"\n>>> Running IronCell Mark-42...")
        import types

        if is_resume:
            from src.configuration_iron_cell import IronCellConfig
            print(f"Loading config and weights from {args.resume_path}...")
            config = IronCellConfig.from_pretrained(args.resume_path)
            setattr(config, "tokenizer_vocab_size", len(tokenizer))
            model = IronCellModel.from_pretrained(args.resume_path, config=config, torch_dtype=torch.bfloat16).to(device)
        else:
            model = IronCellModel.from_pretrained(args.model_name, torch_dtype=torch.bfloat16).to(device)

        model.eval()

        try:
            from src.hack_llama_fsdp import smart_hybrid_attention_forward # 根据你的实际模块调整
        except ImportError:
            raise ImportError("无法导入 smart_hybrid_attention_forward，请检查路径！")

        print("Patching Llama Attention for Deep KV Injection...")
        for layer in model.generator.model.layers:
            layer.self_attn.forward = types.MethodType(smart_hybrid_attention_forward, layer.self_attn)

        print("Running Sequential Evaluation (Perfect RoPE & Physics)...")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            losses = evaluate_mark42_compact_sequential(
                iron_model=model, 
                tokenizer=tokenizer,
                raw_input_ids=raw_input_ids, 
                args=args
            )
            
        _save_results(losses, args.output_dir, f"mark42-buffer{args.buffer_size}")
    else:
        raise ValueError(f"未知的评估模式: {args.eval_mode}")

def _save_results(losses, output_dir, mode_name):
    out_path = os.path.join(output_dir, f"{mode_name}_losses.json")
    with open(out_path, "w") as f:
        json.dump(losses, f)
    print(f"[Done] {mode_name} evaluation complete. Results saved to {out_path}")

if __name__ == "__main__":
    main()