import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import math


def _parse_layers(spec: str) -> list[int]:
    return [int(x.strip()) for x in spec.split(",") if x.strip()]

def _load_texts(path: str) -> list[str]:
    texts: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
    return texts

def _compute_prev_chunk_attention(attn_mean: torch.Tensor, valid_len: int, chunk_size: int) -> torch.Tensor:
    attn = attn_mean[:valid_len, :valid_len].float().clone()
    attn[:, 0] = 0.0  
    prefix = attn.cumsum(dim=-1)
    idx = torch.arange(valid_len, device=attn.device)
    logical_idx = idx - 1
    chunk_idx = torch.div(logical_idx, chunk_size, rounding_mode='floor')
    end_idx = chunk_idx * chunk_size
    end_idx = end_idx.clamp(min=0)
    row = torch.arange(valid_len, device=attn.device)
    sums = prefix[row, end_idx]
    sums = torch.where(chunk_idx > 0, sums, torch.zeros_like(sums))
    return sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--target_layer", type=int, default=15) # 【修改1】只取中间最稳的一层，比如 15
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--q_num", type=int, default=2) # 每个 chunk 对应的 query 数量
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    texts = _load_texts(args.data_path)
    num_samples = len(texts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--> Loading model (Eager Attention)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        # attn_implementation="eager",
    ).to(args.device)
    model.eval()

    max_num_chunks = int(args.max_seq_len) // int(args.chunk_size)
    max_v_len = max(0, max_num_chunks) * int(args.q_num)
    
    # 存储 Teacher 的浓缩 Hidden State 目标 (相比于存全量，这个极小)
    # [num_samples, max_v_len, 4096]
    hidden_dim = model.config.hidden_size
    targets = torch.zeros((num_samples, max_v_len, hidden_dim), dtype=torch.bfloat16)
    valid_v_lens = torch.zeros((num_samples,), dtype=torch.long)

    for start in range(0, num_samples, args.batch_size):
        end = min(start + args.batch_size, num_samples)
        batch_texts = texts[start:end]

        tok = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].to(args.device)
        attention_mask = tok["attention_mask"].to(args.device)

        with torch.no_grad():
            # 【修改2】直接开 output_hidden_states，不要 Hook！
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True, 
                use_cache=False,
            )
            
            # 取出目标层的特征 [B, SeqLen, 4096]
            layer_h = outputs.hidden_states[args.target_layer]

        for b in range(input_ids.size(0)):
            global_idx = start + b
            total_len = int(attention_mask[b].sum().item())
            raw_len = total_len - 1
            # 【修改3】按 Chunk 浓缩特征
            # 取出有效序列
            valid_h = layer_h[b, 1:total_len, :]

            num_chunks = raw_len // int(args.chunk_size)
            left_over = raw_len % int(args.chunk_size)
            num_cmp_chunks = num_chunks - 1 if left_over == 0 else num_chunks
            
            pooled_chunks = []
            for c in range(num_cmp_chunks):
                chunk_start = c * int(args.chunk_size)
                chunk_end = chunk_start + int(args.chunk_size)
                
                # 取出这个 chunk 的特征并平均 (Mean Pooling)
                chunk_feat = valid_h[chunk_start:chunk_end, :].mean(dim=0, keepdim=True)
                
                # 因为 Javis 用了 q_num (比如2) 个 query 去压缩这 16 个 token
                # 我们强迫这两个 query 都去学习这个 chunk 的全局平均语义
                chunk_feat_expanded = chunk_feat.expand(int(args.q_num), -1)
                pooled_chunks.append(chunk_feat_expanded)
            
            if pooled_chunks:
                # 拼起来就是当前样本 Teacher 给出的所有 V 的绝对标准答案！
                teacher_v_targets = torch.cat(pooled_chunks, dim=0)
                actual_v_len = teacher_v_targets.size(0)
                
                targets[global_idx, :actual_v_len, :] = teacher_v_targets.cpu().to(torch.bfloat16)
                valid_v_lens[global_idx] = actual_v_len

        if (start // args.batch_size) % 5 == 0:
            print(f"Processed {end}/{num_samples}")

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "targets": targets, # 这就是给学生 V 的标准答案
            "valid_v_lens": valid_v_lens,
            "target_layer": args.target_layer,
            "chunk_size": args.chunk_size,
            "q_num": args.q_num,
        },
        out_path,
    )

if __name__ == "__main__":
    main()
