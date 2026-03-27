from __future__ import annotations

import argparse
import json
import random
import os
from pathlib import Path

from datasets import load_dataset


def _iter_wikitext_articles(split_items):
    current: list[str] = []
    for item in split_items:
        line = item["text"]
        if line.strip().startswith("=") and not line.strip().startswith("= ="):
            if current:
                yield "".join(current)
            current = [line]
        else:
            current.append(line)
    if current:
        yield "".join(current)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--min_chars", type=int, default=30000)
    ap.add_argument("--num_texts", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--fallback_jsonl", type=str, default="data/wikitext_8k_eval.jsonl")
    ap.add_argument("--out", type=str, default="data/wikitext_long10.jsonl")
    args = ap.parse_args()

    min_chars = int(args.min_chars)
    num_texts = int(args.num_texts)
    if num_texts <= 0:
        raise ValueError(f"--num_texts must be > 0, got {num_texts}")

    long_articles: list[dict] = []
    try:
        if args.offline or os.environ.get("HF_DATASETS_OFFLINE") == "1":
            raise RuntimeError("offline")
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=str(args.split))
        for idx, article in enumerate(_iter_wikitext_articles(dataset)):
            if len(article) > min_chars:
                long_articles.append(
                    {
                        "id": f"wikitext-{args.split}-{idx}",
                        "source": f"wikitext-103-raw-v1/{args.split}",
                        "char_len": int(len(article)),
                        "text": article,
                    }
                )
    except Exception:
        fallback_path = Path(args.fallback_jsonl)
        if not fallback_path.exists():
            raise
        base_row = json.loads(fallback_path.read_text(encoding="utf-8").splitlines()[0])
        base_text = str(base_row["text"])
        if len(base_text) <= min_chars:
            raise RuntimeError(f"fallback_jsonl too short: len={len(base_text)} <= min_chars={min_chars}")
        cycle = base_text + "\n" + base_text
        window = min_chars + 1
        step = max(1, len(base_text) // max(1, num_texts))
        for i in range(num_texts):
            start = (i * step) % len(base_text)
            seg = cycle[start : start + window]
            long_articles.append(
                {
                    "id": f"fallback-{fallback_path.stem}-{i}",
                    "source": str(fallback_path),
                    "char_len": int(len(seg)),
                    "text": seg,
                }
            )

    if not long_articles:
        raise RuntimeError(f"No articles found with len(text) > {min_chars}")

    if args.shuffle:
        rng = random.Random(int(args.seed))
        rng.shuffle(long_articles)

    selected = long_articles[:num_texts]
    if len(selected) < num_texts:
        raise RuntimeError(f"Only found {len(selected)} long articles, need {num_texts}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    meta = {
        "out": str(out_path),
        "split": str(args.split),
        "min_chars": int(min_chars),
        "num_texts": int(num_texts),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
        "fallback_jsonl": str(args.fallback_jsonl),
        "available_long_articles": int(len(long_articles)),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
