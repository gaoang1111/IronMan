from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

from src.adaptive_segment.adaptive_segment import (
    load_model,
    build_input_ids,
    compute_surprisal,
    adaptive_segment,
)


def segments_to_chunk_lens(segments: List[Tuple[int, int]]) -> List[int]:
    return [int(e - s) for s, e in segments]


def process_text(
    tokenizer,
    model,
    device,
    text: str,
    min_base_len: int = 4,
    max_base_len: int = 64,
    target_min_len: int = 8,
    target_max_len: int = 20,
    merge_upper_bound: int = 24,
    event_boundary_threshold: float = 5.0,
    split_max_len: int = 24,
    split_min_len: int = 8,
) -> Dict[str, Any]:
    input_ids, _, _, tokens = build_input_ids(
        tokenizer=tokenizer,
        text=text,
        needle=None,
        insert_pos=0,
    )

    surprisal = compute_surprisal(model, input_ids, device=device)

    segments = adaptive_segment(
        tokens=tokens,
        surprisal=surprisal,
        min_base_len=min_base_len,
        max_base_len=max_base_len,
        target_min_len=target_min_len,
        target_max_len=target_max_len,
        merge_upper_bound=merge_upper_bound,
        event_boundary_threshold=event_boundary_threshold,
        split_max_len=split_max_len,
        split_min_len=split_min_len,
    )

    chunk_lens = segments_to_chunk_lens(segments)

    if len(chunk_lens) == 0:
        raise ValueError("empty segments")

    chunk_lens[0] -= 1

    if chunk_lens[0] <= 0:
        raise ValueError("first chunk became invalid after BOS removal")

    raw_len = len(tokenizer.encode(text, add_special_tokens=False))
    if sum(chunk_lens) != raw_len:
        raise ValueError(
            f"chunk_lens mismatch: sum={sum(chunk_lens)} vs raw_len={raw_len}"
        )

    return {
        "chunk_lens": chunk_lens,
        "num_chunks": len(chunk_lens),
        "avg_chunk_len": sum(chunk_lens) / len(chunk_lens) if chunk_lens else 0.0,
        "segments": segments,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--min_base_len", type=int, default=4)
    parser.add_argument("--max_base_len", type=int, default=64)
    parser.add_argument("--target_min_len", type=int, default=8)
    parser.add_argument("--target_max_len", type=int, default=20)
    parser.add_argument("--merge_upper_bound", type=int, default=24)
    parser.add_argument("--event_boundary_threshold", type=float, default=5.0)
    parser.add_argument("--split_max_len", type=int, default=24)
    parser.add_argument("--split_min_len", type=int, default=8)

    args = parser.parse_args()

    device = None if args.device == "auto" else args.device
    tokenizer, model, device = load_model(
        model_path=args.model_path,
        device=device,
        dtype=args.dtype,
    )

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

            seg_info = process_text(
                tokenizer=tokenizer,
                model=model,
                device=device,
                text=text,
                min_base_len=args.min_base_len,
                max_base_len=args.max_base_len,
                target_min_len=args.target_min_len,
                target_max_len=args.target_max_len,
                merge_upper_bound=args.merge_upper_bound,
                event_boundary_threshold=args.event_boundary_threshold,
                split_max_len=args.split_max_len,
                split_min_len=args.split_min_len,
            )

            obj["chunk_lens"] = seg_info["chunk_lens"]
            obj["num_chunks"] = seg_info["num_chunks"]
            obj["avg_chunk_len"] = seg_info["avg_chunk_len"]

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1

    print(f"Done. Wrote {count} samples to {out_path}")


if __name__ == "__main__":
    main()