from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data_processor import AdaptiveZipperBuilder, AdaptiveIronCellCollator
from src.train_utils import load_tokenizer


class _Args:
    def __init__(self):
        self.model_name = "/default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B"
        self.phase = "phase2"
        self.resume_path = None
        self.load_weights_only = True
        self.parallel = "none"


def show_tensor(name, x, rows=8, cols=40):
    print(f"\n{name}")
    if isinstance(x, torch.Tensor):
        print("shape:", tuple(x.shape))
        if x.ndim == 1:
            print(x[:rows])
        elif x.ndim == 2:
            print(x[:rows, :cols])
        elif x.ndim == 3:
            print(x[:1, :rows, :cols])
    else:
        print(x)


def decode_chunks(tokenizer, text: str, chunk_lens: list[int]):
    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    assert sum(chunk_lens) == len(raw_ids), (
        f"chunk_lens sum mismatch: {sum(chunk_lens)} vs raw_len={len(raw_ids)}"
    )

    out = []
    cur = 0
    for i, ln in enumerate(chunk_lens):
        ids = raw_ids[cur:cur + ln]
        out.append((i, ln, tokenizer.decode(ids, skip_special_tokens=False)))
        cur += ln
    return out


def main():
    # 改成你自己的输出文件
    jsonl_path = "/default-vepfs/public/user/ga/Iron/IronMan/data/eval_adaptive.jsonl"
    jsonl_path = "/default-vepfs/public/user/ga/Iron/IronMan/data/test_collator_adaptive.jsonl"

    args = _Args()
    tokenizer, _ = load_tokenizer(args)

    # 只取第一条样本验证
    with open(jsonl_path, "r", encoding="utf-8") as f:
        obj = json.loads(next(f))

    text = obj["text"]
    chunk_lens = obj["chunk_lens"]

    print("=" * 80)
    print("TEXT:")
    # print(text)
    print("=" * 80)
    # print("chunk_lens:", chunk_lens)
    print("num_chunks:", len(chunk_lens))

    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    print("raw_len:", len(raw_ids))
    print("sum(chunk_lens):", sum(chunk_lens))
    assert sum(chunk_lens) == len(raw_ids)

    print("\n=== Decoded raw chunks ===")
    for idx, ln, chunk_text in decode_chunks(tokenizer, text, chunk_lens):
        print(f"[Chunk {idx}] len={ln}")
        print(chunk_text)
        print("-" * 60)

    # -----------------------------
    # Builder 验证
    # -----------------------------
    builder = AdaptiveZipperBuilder(
        tokenizer=tokenizer,
        prompt=text,
        raw_chunk_lens=chunk_lens,
        buffer_size=0,
        num_v=1,
        random_gate=0,
        truncate_len=None,
    )

    print("\n=== Builder fields ===")
    print("raw_len:", builder.raw_len)
    print("num_raw_segments:", builder.num_raw_segments)
    print("num_cmp_chunks:", builder.num_cmp_chunks)
    print("prefix_len:", builder.prefix_len)
    print("valid_len:", builder.valid_len)
    print("memory_positions:", builder.memory_positions)

    print("\nraw_chunks lens:", [len(x) for x in builder.raw_chunks])
    print("cmp_wrapped_chunks lens:", [len(x) for x in builder.cmp_wrapped_chunks])

    assert builder.raw_len == len(raw_ids)
    assert sum(len(x) for x in builder.raw_chunks) == builder.raw_len
    assert [len(x) for x in builder.raw_chunks] == chunk_lens
    assert len(builder.cmp_wrapped_chunks) == len(chunk_lens) - 1

    gen_tokens = tokenizer.convert_ids_to_tokens(builder.gen_input_ids)
    print("\n=== gen_input_ids preview ===")
    print(gen_tokens[:80])

    labels = builder.build_gen_labels(device=torch.device("cpu"))
    attn, pos = builder.build_gen_attention_and_pos(
        seq_len=builder.valid_len,
        device=torch.device("cpu"),
    )

    show_tensor("labels", labels)
    show_tensor("attention_mask_2d", attn)
    show_tensor("position_ids", pos)

    # -----------------------------
    # Collator 验证
    # -----------------------------
    collator = AdaptiveIronCellCollator(tokenizer)
    batch = collator([
        {
            "text": text,
            "idx": 0,
            "chunk_lens": chunk_lens,
        }
    ])

    print("\n=== Collator outputs ===")
    show_tensor("zipper_input_ids", batch.zipper_input_ids)
    show_tensor("labels", batch.labels)
    show_tensor("attention_mask_2d", batch.attention_mask_2d)
    show_tensor("position_ids", batch.position_ids)
    show_tensor("chunk_input_ids", batch.chunk_input_ids)
    show_tensor("chunk_attention_mask", batch.chunk_attention_mask)
    show_tensor("memory_positions", batch.memory_positions)
    show_tensor("prefix_lens", batch.prefix_lens)
    show_tensor("valid_lens", batch.valid_lens)

    valid_len = int(batch.valid_lens[0].item())
    zipper_ids = batch.zipper_input_ids[0, :valid_len].tolist()

    print("\n=== Decoded zipper_input_ids (valid part) ===")
    print(tokenizer.convert_ids_to_tokens(zipper_ids)[:80])

    # 简单断言
    assert batch.chunk_input_ids.shape[1] == len(chunk_lens) - 1
    assert int(batch.prefix_lens[0].item()) == builder.prefix_len
    assert int(batch.valid_lens[0].item()) == builder.valid_len

    print("\n✅ Builder / Collator geometry validation passed.")


if __name__ == "__main__":
    main()
