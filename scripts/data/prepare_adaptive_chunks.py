from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import os
os.sys.path.insert(0, str(PROJECT_ROOT))

from IronMan.src.adaptive_segment.adaptive_segment import adaptive_segment, compute_surprisal, load_model


def _segments_to_chunk_lens(*, segs: list[tuple[int, int]], raw_len: int, has_bos: bool) -> list[int]:
    boundaries: list[int] = []
    for _, e in segs:
        e_raw = int(e - 1) if has_bos else int(e)
        if e_raw <= 0:
            continue
        boundaries.append(e_raw)
    if len(boundaries) == 0 or int(boundaries[-1]) != int(raw_len):
        boundaries.append(int(raw_len))

    prev = 0
    lens: list[int] = []
    for b in boundaries:
        b_i = int(b)
        if b_i > prev:
            lens.append(b_i - prev)
            prev = b_i
    if prev != int(raw_len):
        lens.append(int(raw_len) - prev)
    return lens


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--min_base_len", type=int, default=4)
    p.add_argument("--max_base_len", type=int, default=64)
    p.add_argument("--target_min_len", type=int, default=8)
    p.add_argument("--target_max_len", type=int, default=20)
    p.add_argument("--merge_upper_bound", type=int, default=24)
    p.add_argument("--event_boundary_threshold", type=float, default=5.0)
    p.add_argument("--split_max_len", type=int, default=24)
    p.add_argument("--split_min_len", type=int, default=8)
    args = p.parse_args()

    tokenizer, model, device_str = load_model(str(args.model_path), device=None if args.device == "auto" else args.device, dtype=str(args.dtype))
    device = torch.device(device_str)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_total = 0
    n_written = 0
    n_skipped = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if str(line).strip() == "":
                continue
            n_total += 1
            if args.max_samples and n_written >= int(args.max_samples):
                break

            obj = json.loads(line)
            text = obj.get("text")
            if not isinstance(text, str) or str(text).strip() == "":
                n_skipped += 1
                continue

            raw_ids = tokenizer.encode(text, add_special_tokens=False)
            raw_len = int(len(raw_ids))
            if raw_len <= 0:
                n_skipped += 1
                continue

            if tokenizer.bos_token_id is not None:
                full_ids = [int(tokenizer.bos_token_id)] + raw_ids
                has_bos = True
            else:
                full_ids = raw_ids
                has_bos = False

            input_ids = torch.tensor([full_ids], dtype=torch.long)
            tokens = tokenizer.convert_ids_to_tokens(full_ids)
            surprisal = compute_surprisal(model, input_ids, device=str(device))

            segs = adaptive_segment(
                tokens=tokens,
                surprisal=surprisal,
                min_base_len=int(args.min_base_len),
                max_base_len=int(args.max_base_len),
                target_min_len=int(args.target_min_len),
                target_max_len=int(args.target_max_len),
                merge_upper_bound=int(args.merge_upper_bound),
                event_boundary_threshold=float(args.event_boundary_threshold),
                split_max_len=int(args.split_max_len),
                split_min_len=int(args.split_min_len),
            )

            chunk_lens = _segments_to_chunk_lens(segs=segs, raw_len=raw_len, has_bos=has_bos)
            if int(sum(chunk_lens)) != int(raw_len) or any(int(x) <= 0 for x in chunk_lens) or len(chunk_lens) < 2:
                n_skipped += 1
                continue

            obj["chunk_lens"] = [int(x) for x in chunk_lens]
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_written += 1

    if n_total > 0:
        kept = n_written / max(1, n_total)
        print(f"total={n_total} written={n_written} skipped={n_skipped} kept={kept:.3f}")
    else:
        print("No input samples found.")


if __name__ == "__main__":
    main()

