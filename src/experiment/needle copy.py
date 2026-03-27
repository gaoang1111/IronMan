from __future__ import annotations

import csv
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from tqdm import tqdm

# DEFAULT_ANIMALS = (
#     "capybara",
#     "lemur",
#     "alpaca",
#     "tapir",
#     "axolotl",
#     "manatee",
#     "pangolin",
# )

DEFAULT_ANIMALS = (
    "tiger",
    "elephant",
    "whale",
    "monkey",
    "snake",

)


@dataclass(frozen=True)
class NeedleScaleConfig:
    num_positions: int = 20
    seed: int = 0
    animals: tuple[str, ...] = DEFAULT_ANIMALS
    max_new_tokens: int = 20
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    output_preview_chars: int = 240
    tag: str = "needle"


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    rows: list[dict] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def build_needle_sentence(animal: str) -> str:
    return f"The pass key to the core is {animal}."
    # return f"Inside the vault, the secret code is {animal}"
    # return f"The hidden animal mentioned earlier is a {animal}."


def build_query_text() -> str:
    return f"The pass key to the core is "
    # return f"The secret code in the vault is "
    # return "The hidden animal mentioned earlier is a"


def choose_depths(num_positions: int) -> list[float]:
    n = int(num_positions)
    if n <= 0:
        raise ValueError(f"num_positions must be > 0, got {n}")
    return [i / n for i in range(n)]


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _safe_needle_chunk_ids(tokenizer, chunk_size: int, animals: tuple[str, ...], rng: random.Random) -> tuple[str, list[int]]:
    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

    animals = tuple(str(a) for a in animals if str(a).strip())
    if not animals:
        raise ValueError("animals must be non-empty")

    max_tries = max(16, len(animals) * 3)
    last_len: int | None = None
    last_sentence: str | None = None
    for _ in range(max_tries):
        animal = rng.choice(animals)
        sentence = build_needle_sentence(animal)
        ids = _token_ids(tokenizer, sentence)
        last_len = len(ids)
        last_sentence = sentence
        if len(ids) <= chunk_size:
            pad_len = chunk_size - len(ids)
            if pad_len > 0:
                pad_id = None
                for cand in (" ", ".", "the", "\n"):
                    cand_ids = _token_ids(tokenizer, cand)
                    if cand_ids:
                        pad_id = int(cand_ids[0])
                        break
                if pad_id is None:
                    pad_id = 0
                # ids = ids + [pad_id] * pad_len
            return animal, ids

    raise ValueError(
        f"Needle sentence too long for chunk_size={chunk_size}. "
        f"last_len={last_len}, last_sentence={last_sentence!r}"
    )


def _hit_animal(output_text: str, animal: str) -> bool:
    out = str(output_text).lower()
    a = str(animal).lower().strip()
    if not a:
        return False
    return re.search(rf"\b{re.escape(a)}\b", out) is not None


def _preview(text: str, max_chars: int) -> str:
    s = str(text)
    n = int(max_chars)
    if n <= 0 or len(s) <= n:
        return s
    return s[:n] + "..."


def run_needle_scale_from_jsonl(
    infra_engine,
    tokenizer,
    *,
    data_jsonl_path: str | Path,
    out_dir: str | Path,
    config: NeedleScaleConfig = NeedleScaleConfig(),
    max_texts: int | None = None,
) -> dict:
    # infra_engine.model.javis.alpha = 0.0
    tag = config.tag
    rows = _read_jsonl(data_jsonl_path)
    if max_texts is not None:
        rows = rows[: int(max_texts)]

    out_dir = f"{out_dir}/{tag}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(config.seed))
    chunk_size = int(getattr(infra_engine, "chunk_size"))
    query_ids = _token_ids(tokenizer, build_query_text())
    depths = choose_depths(int(config.num_positions))

    runs: list[dict] = []
    regressions: list[dict] = []

    for text_idx, row in tqdm(enumerate(rows)):
        text_id = row.get("id", text_idx)
        bg_text = row.get("text")
        if bg_text is None:
            raise ValueError(f"Missing 'text' in jsonl row: {row.keys()}")
        bg_text = bg_text.replace(" @-@ ", "-").replace(" @,@ ", ",")

        bg_ids = _token_ids(tokenizer, str(bg_text))
        total_chunks = len(bg_ids) // chunk_size
        if total_chunks <= 0:
            raise ValueError(f"text_id={text_id}: bg too short for chunk_size={chunk_size}, tokens={len(bg_ids)}")

        for position_idx, depth in enumerate(depths):
            animal, needle_chunk_ids = _safe_needle_chunk_ids(tokenizer, chunk_size, config.animals, rng)
            target_chunk_idx = int(total_chunks * float(depth))
            if target_chunk_idx >= total_chunks:
                target_chunk_idx = total_chunks - 1
            insert_pos = target_chunk_idx * chunk_size

            offset = chunk_size - len(needle_chunk_ids)
            insert_pos += offset
            front_ids = bg_ids[:insert_pos]
            back_ids = bg_ids[insert_pos:]
            full_input_ids = front_ids + needle_chunk_ids + back_ids + query_ids

            generated_text = infra_engine.generate(
                input_ids=full_input_ids,
                max_new_tokens=int(config.max_new_tokens),
                temperature=float(config.temperature),
                repetition_penalty=float(config.repetition_penalty),
            )
            output_text = str(generated_text).strip().split("\n", 1)[0]
            hit = _hit_animal(output_text, animal)

            run = {
                "text_id": text_id,
                "position_idx": int(position_idx),
                "depth": float(depth),
                "target_chunk_idx": int(target_chunk_idx),
                "insert_pos": int(insert_pos),
                "animal": animal,
                "hit": bool(hit),
                "output_preview": _preview(output_text, int(config.output_preview_chars)),
                "bg_tokens_len": int(len(bg_ids)),
                "bg_char_len": int(len(str(bg_text))),
                "chunk_size": int(chunk_size),
                "buffer_num": int(getattr(infra_engine, "buffer_num", -1)),
                "q_num": int(getattr(infra_engine, "q_num", -1)),
                "target_layers": list(getattr(infra_engine, "target_layers", ())),
                "seed": int(config.seed),
            }
            runs.append(run)

            if not hit:
                ctx_start = max(0, insert_pos - chunk_size)
                ctx_end = min(len(bg_ids), insert_pos + 2 * chunk_size)
                regressions.append(
                    {
                        **run,
                        "needle_sentence": build_needle_sentence(animal),
                        "query_text": build_query_text(),
                        "context_text": tokenizer.decode(bg_ids[ctx_start:ctx_end], skip_special_tokens=False),
                        "context_token_range": [int(ctx_start), int(ctx_end)],
                    }
                )

    runs_path = out_dir / "needle_scale_runs.jsonl"
    regressions_path = out_dir / "needle_scale_regressions.jsonl"
    _write_jsonl(runs_path, runs)
    _write_jsonl(regressions_path, regressions)

    by_pos: dict[int, dict[str, int]] = {}
    by_text: dict[str, dict[str, int]] = {}
    for r in runs:
        p = int(r["position_idx"])
        by_pos.setdefault(p, {"n": 0, "hit": 0})
        by_pos[p]["n"] += 1
        by_pos[p]["hit"] += int(bool(r["hit"]))

        t = str(r["text_id"])
        by_text.setdefault(t, {"n": 0, "hit": 0})
        by_text[t]["n"] += 1
        by_text[t]["hit"] += int(bool(r["hit"]))

    pos_summary: list[dict] = []
    for p in sorted(by_pos.keys()):
        n = by_pos[p]["n"]
        h = by_pos[p]["hit"]
        hit_rate = (h / n) if n else 0.0
        pos_summary.append(
            {
                "position_idx": int(p),
                "n": int(n),
                "hit": int(h),
                "hit_rate": float(hit_rate),
                "regression_rate": float(1.0 - hit_rate),
            }
        )

    text_summary: list[dict] = []
    for t in sorted(by_text.keys()):
        n = by_text[t]["n"]
        h = by_text[t]["hit"]
        hit_rate = (h / n) if n else 0.0
        text_summary.append(
            {
                "text_id": t,
                "n": int(n),
                "hit": int(h),
                "hit_rate": float(hit_rate),
            }
        )

    _write_csv(out_dir / "needle_scale_summary_by_position.csv", pos_summary)
    _write_csv(out_dir / "needle_scale_summary_by_text.csv", text_summary)

    meta = {
        "config": asdict(config),
        "data_jsonl_path": str(Path(data_jsonl_path)),
        "out_dir": str(out_dir),
        "num_texts": int(len(rows)),
        "num_runs": int(len(runs)),
        "num_regressions": int(len(regressions)),
        "chunk_size": int(chunk_size),
        "buffer_num": int(getattr(infra_engine, "buffer_num", -1)),
        "q_num": int(getattr(infra_engine, "q_num", -1)),
        "target_layers": list(getattr(infra_engine, "target_layers", ())),
    }
    (out_dir / "needle_scale_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
