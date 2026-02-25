import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument("--layers", type=str, default="24,26,28,30,31")
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # # 1. 强制 Batch Size 为 1
    # if args.batch_size > 1:
    #     print(f"WARNING: Forcing batch_size=1 to avoid OOM with eager attention at length {args.max_seq_len}.")
    #     args.batch_size = 1

    texts = _load_texts(args.data_path)
    num_samples = len(texts)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("--> Loading model with attn_implementation='eager'...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        attn_implementation="eager",
    ).to(args.device)
    model.eval()

    layer_ids = _parse_layers(args.layers)
    targets = torch.zeros((num_samples, args.max_seq_len), dtype=torch.float32)
    valid_lens = torch.zeros((num_samples,), dtype=torch.long)
    max_valid_len = 0

    # 用字典存储当前 forward 的结果
    captured_attentions = {}

    # [关键修改]：Hook 内部直接降维 + CPU 转移
    def get_attention_hook(layer_idx):
        def hook(module, input, output):
            # output[1] 是 attention weights: [B, Heads, Seq, Seq]
            attn_weights = output[1]
            if attn_weights is not None:
                # 1. 立即平均 Heads: [1, 32, 10k, 10k] -> [1, 10k, 10k]
                # 数据量从 12.5GB 瞬间变为 390MB
                attn_mean = attn_weights.mean(dim=1)
                
                # 2. 立即转存 CPU，释放 GPU 显存
                captured_attentions[layer_idx] = attn_mean.detach().cpu()
        return hook

    handles = []
    for layer_idx in layer_ids:
        handle = model.model.layers[layer_idx].self_attn.register_forward_hook(get_attention_hook(layer_idx))
        handles.append(handle)

    print(f"--> Registered hooks on layers: {layer_ids} (Optimized for Memory)")

    for start in range(0, num_samples, args.batch_size):
        end = min(start + args.batch_size, num_samples)
        batch_texts = texts[start:end]
        
        captured_attentions = {}

        tok = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=args.max_seq_len,
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].to(args.device)
        attention_mask = tok["attention_mask"].to(args.device)

        with torch.no_grad():
            model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False, 
                use_cache=False,
            )

        # 此时 captured_attentions 里存的是 CPU 上的 [B, S, S] 小矩阵
        # 我们可以安全地在 CPU 上做累加
        
        sum_attn = None
        for i in layer_ids:
            layer_attn = captured_attentions[i].float() # 已经在 CPU 了
            if sum_attn is None:
                sum_attn = layer_attn
            else:
                sum_attn += layer_attn
            del captured_attentions[i]
            
        # 平均并移回 GPU 做 cumsum 计算 (这步 GPU 显存是够的，因为只有 390MB)
        # 或者全程 CPU 计算也行，为了快我们放回 GPU
        attn_mean = (sum_attn / len(layer_ids)).to(args.device)

        for b in range(input_ids.size(0)):
            global_idx = start + b
            vlen = int(attention_mask[b].sum().item())
            valid_lens[global_idx] = vlen
            if vlen > max_valid_len:
                max_valid_len = vlen
            sums = _compute_prev_chunk_attention(attn_mean[b], vlen, args.chunk_size)
            targets[global_idx, :vlen] = sums.cpu()

        if (start // args.batch_size) % 5 == 0:
            print(f"Processed {end}/{num_samples}")

    for handle in handles:
        handle.remove()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "targets": targets,
            "valid_lens": valid_lens,
            "chunk_size": args.chunk_size,
            "max_seq_len": args.max_seq_len,
            "layers": layer_ids,
            "max_valid_len": max_valid_len,
        },
        out_path,
    )

if __name__ == "__main__":
    main()
