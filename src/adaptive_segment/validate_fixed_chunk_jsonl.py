from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer


def validate_one(tokenizer, text: str, chunk_lens: list[int], chunk_size: int):
    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    raw_len = len(raw_ids)

    result = {
        "ok": True,
        "raw_len": raw_len,
        "sum_chunk_lens": sum(chunk_lens),
        "num_chunks": len(chunk_lens),
        "errors": [],
    }

    if sum(chunk_lens) != raw_len:
        result["ok"] = False
        result["errors"].append(
            f"sum(chunk_lens)={sum(chunk_lens)} != raw_len={raw_len}"
        )

    if len(chunk_lens) == 0:
        result["ok"] = False
        result["errors"].append("empty chunk_lens")
        return result

    # 前面所有 chunk 是否都等于 chunk_size
    if len(chunk_lens) > 1:
        prefix = chunk_lens[:-1]
        bad_prefix_idx = [i for i, x in enumerate(prefix) if x != chunk_size]
        if bad_prefix_idx:
            result["ok"] = False
            result["errors"].append(
                f"non-final chunks not all == {chunk_size}, bad idx={bad_prefix_idx[:10]}"
            )

    # 最后一段是否合法
    last_len = chunk_lens[-1]
    if not (1 <= last_len <= chunk_size):
        result["ok"] = False
        result["errors"].append(
            f"last chunk invalid: last_len={last_len}, expected in [1, {chunk_size}]"
        )

    return result


def decode_chunks(tokenizer, text: str, chunk_lens: list[int]):
    raw_ids = tokenizer.encode(text, add_special_tokens=False)
    assert sum(chunk_lens) == len(raw_ids), (
        f"sum(chunk_lens)={sum(chunk_lens)} != raw_len={len(raw_ids)}"
    )

    out = []
    cur = 0
    for i, ln in enumerate(chunk_lens):
        ids = raw_ids[cur:cur + ln]
        out.append({
            "idx": i,
            "len": ln,
            "text": tokenizer.decode(ids, skip_special_tokens=False),
            "tokens": tokenizer.convert_ids_to_tokens(ids),
        })
        cur += ln
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--max_bad_print", type=int, default=20)
    parser.add_argument("--sample_decode", type=int, default=3)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    in_path = Path(args.input_jsonl)

    total = 0
    bad = 0
    bad_cases = []
    good_cases = []

    with in_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            obj = json.loads(line)
            text = obj.get("text")
            chunk_lens = obj.get("chunk_lens")

            if not isinstance(text, str):
                bad += 1
                bad_cases.append({
                    "line_no": line_no,
                    "errors": ["missing or invalid text"],
                })
                continue

            if not isinstance(chunk_lens, list) or not all(isinstance(x, int) for x in chunk_lens):
                bad += 1
                bad_cases.append({
                    "line_no": line_no,
                    "errors": ["missing or invalid chunk_lens"],
                })
                continue

            total += 1
            res = validate_one(tokenizer, text, chunk_lens, args.chunk_size)
            if not res["ok"]:
                bad += 1
                bad_cases.append({
                    "line_no": line_no,
                    "chunk_lens": chunk_lens,
                    **res,
                    "text_preview": text[:200].replace("\n", "\\n"),
                })
            else:
                good_cases.append({
                    "line_no": line_no,
                    "text": text,
                    "chunk_lens": chunk_lens,
                    **res,
                })

    print("=" * 80)
    print(f"file: {in_path}")
    print(f"total_valid_samples: {total}")
    print(f"bad_samples: {bad}")
    print(f"bad_ratio: {bad / total if total > 0 else 0:.6f}")
    print("=" * 80)

    if bad_cases:
        print("\n=== BAD CASES ===")
        for case in bad_cases[: args.max_bad_print]:
            print(f"\n[line {case['line_no']}]")
            print("errors:", case["errors"])
            if "chunk_lens" in case:
                print("chunk_lens:", case["chunk_lens"])
                print("raw_len:", case.get("raw_len"))
                print("sum_chunk_lens:", case.get("sum_chunk_lens"))
            if "text_preview" in case:
                print("text_preview:", case["text_preview"])

    # 抽样 decode 正常 case
    # if good_cases and args.sample_decode > 0:
    #     print("\n=== SAMPLE DECODE (GOOD CASES) ===")
    #     samples = random.sample(good_cases, min(args.sample_decode, len(good_cases)))
    #     for case in samples:
    #         print("\n" + "=" * 80)
    #         print(f"[line {case['line_no']}] raw_len={case['raw_len']} chunk_lens={case['chunk_lens']}")
    #         decoded = decode_chunks(tokenizer, case["text"], case["chunk_lens"])
    #         for chunk in decoded:
    #             print(f"\n[Chunk {chunk['idx']}] len={chunk['len']}")
    #             print(chunk["text"])


if __name__ == "__main__":
    main()