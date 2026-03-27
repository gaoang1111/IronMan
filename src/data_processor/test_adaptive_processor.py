# src/experiment/debug_adaptive_builder.py
from __future__ import annotations

import torch

from src.train_utils import load_tokenizer
from src.eval_ppl_adaptive import _Args   # 如果不方便引用，就自己手写个 mock args
from src.data_processor import AdaptiveZipperBuilder, AdaptiveIronCellCollator


def print_tensor(name, x, max_rows=20, max_cols=40):
    print(f"\n{name}:")
    if isinstance(x, torch.Tensor):
        print("shape:", tuple(x.shape))
        if x.ndim == 1:
            print(x[:max_rows])
        elif x.ndim == 2:
            print(x[:max_rows, :max_cols])
        elif x.ndim == 3:
            print(x[:min(2, x.shape[0]), :max_rows, :max_cols])
    else:
        print(x)


def main():
    args = _Args(
        model_name="",
        phase="phase2",
        resume_path=None,
        load_weights_only=True,
        parallel="none",
    )
    tokenizer, _ = load_tokenizer(args)

    text = "The secret passcode is tiger. It is very important."
    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    print("raw_ids len:", len(raw_ids))
    print("raw tokens:", tokenizer.convert_ids_to_tokens(raw_ids))

    # 假设离线 segmentation 已经给出
    chunk_lens = [7, len(raw_ids) - 7]
    print("chunk_lens:", chunk_lens)

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
    print("raw_chunks:", builder.raw_chunks)
    print("cmp_wrapped_chunks:", builder.cmp_wrapped_chunks)

    gen_ids = torch.tensor(builder.gen_input_ids, dtype=torch.long)
    print("gen tokens:")
    print(tokenizer.convert_ids_to_tokens(gen_ids.tolist()))

    labels = builder.build_gen_labels(device=torch.device("cpu"))
    attn, pos = builder.build_gen_attention_and_pos(
        seq_len=builder.valid_len,
        device=torch.device("cpu"),
    )

    print_tensor("labels", labels)
    print_tensor("attention_mask_2d", attn)
    print_tensor("position_ids", pos)

    print("\n=== Collator check ===")
    collator = AdaptiveIronCellCollator(tokenizer)
    batch = collator([
        {"text": text, "idx": 0, "chunk_lens": chunk_lens}
    ])

    print_tensor("zipper_input_ids", batch.zipper_input_ids)
    print_tensor("labels", batch.labels)
    print_tensor("attention_mask_2d", batch.attention_mask_2d)
    print_tensor("position_ids", batch.position_ids)
    print_tensor("chunk_input_ids", batch.chunk_input_ids)
    print_tensor("chunk_attention_mask", batch.chunk_attention_mask)
    print_tensor("memory_positions", batch.memory_positions)
    print_tensor("prefix_lens", batch.prefix_lens)
    print_tensor("valid_lens", batch.valid_lens)

    print("\nDecoded zipper_input_ids[0]:")
    z = batch.zipper_input_ids[0, : batch.valid_lens[0]].tolist()
    print(tokenizer.convert_ids_to_tokens(z))


if __name__ == "__main__":
    main()
