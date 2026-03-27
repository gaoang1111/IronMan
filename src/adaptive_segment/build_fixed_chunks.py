from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def build_fixed_chunk_lens(tokenizer, text: str, chunk_size: int = 16):
    ids = tokenizer.encode(text, add_special_tokens=False)
    n = len(ids)
    return [min(chunk_size, n - i) for i in range(0, n, chunk_size)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str):
                raise ValueError("Each line must contain 'text' as string.")

            chunk_lens = build_fixed_chunk_lens(tokenizer, text, chunk_size=args.chunk_size)

            obj["chunk_lens"] = chunk_lens
            obj["num_chunks"] = len(chunk_lens)
            obj["avg_chunk_len"] = sum(chunk_lens) / len(chunk_lens) if chunk_lens else 0.0

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done. Wrote {count} samples to {out_path}")


if __name__ == "__main__":
    main()