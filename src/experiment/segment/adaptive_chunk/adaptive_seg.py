from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. Load model
# ============================================================

def load_model(
    model_path: str,
    device: Optional[str] = None,
    dtype: str = "bf16",
    trust_remote_code: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, device


# ============================================================
# 2. Build input with optional needle
# ============================================================

def build_input_ids(
    tokenizer,
    text: str,
    needle: Optional[str] = None,
    insert_pos: int = 0,
):
    bg_ids = tokenizer.encode(text, add_special_tokens=False)
    insert_pos = max(0, min(insert_pos, len(bg_ids)))

    if needle is None:
        needle_ids: List[int] = []
    else:
        needle_ids = tokenizer.encode(needle, add_special_tokens=False)

    full_ids = bg_ids[:insert_pos] + needle_ids + bg_ids[insert_pos:]
    needle_start = insert_pos
    needle_end = insert_pos + len(needle_ids)

    if tokenizer.bos_token_id is not None:
        full_ids = [tokenizer.bos_token_id] + full_ids
        needle_start += 1
        needle_end += 1

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    tokens = tokenizer.convert_ids_to_tokens(full_ids)

    meta = {
        "bg_len": len(bg_ids),
        "needle_len": len(needle_ids),
        "insert_pos": insert_pos,
        "full_len": len(full_ids),
    }
    return input_ids, needle_start, needle_end, tokens, meta


# ============================================================
# 3. Surprisal
# ============================================================

def compute_surprisal(
    model,
    input_ids: torch.Tensor,
    device: Optional[str] = None,
) -> List[float]:
    if device is None:
        device = next(model.parameters()).device

    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

    surprisal = -token_logp[0].float().cpu()
    surprisal = torch.cat([torch.tensor([float("nan")]), surprisal], dim=0)
    return surprisal.tolist()


# ============================================================
# 4. Helpers
# ============================================================

STRONG_PUNCT = {".", "?", "!", "Ġ.", "Ġ?", "Ġ!", "Ċ"}
WEAK_PUNCT = {",", ";", ":", "Ġ,", "Ġ;", "Ġ:"}


def strip_space_marker(tok: str) -> str:
    return tok.lstrip("Ġ")


def is_strong_boundary_token(tok: str) -> bool:
    t = strip_space_marker(tok)
    return tok in STRONG_PUNCT or t in {".", "?", "!"}


def is_weak_boundary_token(tok: str) -> bool:
    t = strip_space_marker(tok)
    return tok in WEAK_PUNCT or t in {",", ";", ":"}


def safe_mean(vals: List[float]) -> float:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def safe_sum(vals: List[float]) -> float:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(vals) if vals else 0.0


def safe_max(vals: List[float]) -> float:
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return max(vals) if vals else float("nan")


def rolling_mean(values: List[float], window: int = 3) -> List[float]:
    assert window >= 1 and window % 2 == 1
    n = len(values)
    half = window // 2
    out: List[float] = []
    for i in range(n):
        vals = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            v = values[j]
            if not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


# ============================================================
# 5. Segmentation: punctuation first + refine
# ============================================================

def split_by_punctuation(
    tokens: List[str],
    min_base_len: int = 4,
    max_base_len: int = 48,
) -> List[Tuple[int, int]]:
    n = len(tokens)
    segments: List[Tuple[int, int]] = []
    start = 0

    for i, tok in enumerate(tokens):
        seg_len = i - start + 1

        if seg_len >= min_base_len and is_strong_boundary_token(tok):
            segments.append((start, i + 1))
            start = i + 1
            continue

        if seg_len >= max_base_len:
            segments.append((start, i + 1))
            start = i + 1

    if start < n:
        segments.append((start, n))

    return segments


def boundary_event_strength(
    surprisal: List[float],
    boundary_idx: int,
    look_left: int = 3,
    look_right: int = 3,
) -> float:
    left = surprisal[max(0, boundary_idx - look_left): boundary_idx]
    right = surprisal[boundary_idx: min(len(surprisal), boundary_idx + look_right)]
    left_m = safe_mean(left)
    right_m = safe_mean(right)

    if math.isnan(left_m):
        left_m = 0.0
    if math.isnan(right_m):
        right_m = 0.0

    return max(left_m, right_m)


def merge_adjacent_chunks(
    tokens: List[str],
    surprisal: List[float],
    base_segments: List[Tuple[int, int]],
    max_merge_len: int = 24,
    event_boundary_threshold: float = 6.0,
) -> List[Tuple[int, int]]:
    if not base_segments:
        return []

    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = base_segments[0]

    for nxt_s, nxt_e in base_segments[1:]:
        total_len = nxt_e - cur_s
        boundary_strength = boundary_event_strength(surprisal, nxt_s)

        can_merge = (
            total_len <= max_merge_len
            and boundary_strength < event_boundary_threshold
        )

        prev_tok = tokens[nxt_s - 1] if nxt_s - 1 >= 0 else ""
        if is_strong_boundary_token(prev_tok):
            can_merge = False

        if can_merge:
            cur_e = nxt_e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = nxt_s, nxt_e

    merged.append((cur_s, cur_e))
    return merged


def choose_split_point_in_long_chunk(
    tokens: List[str],
    surprisal: List[float],
    start: int,
    end: int,
    min_len: int = 8,
) -> Optional[int]:
    seg_len = end - start
    if seg_len <= 2 * min_len:
        return None

    center = (start + end) // 2
    left = start + min_len
    right = end - min_len
    if left >= right:
        return None

    roll = rolling_mean(surprisal, window=3)
    candidates: List[Tuple[float, int]] = []

    for i in range(left, right):
        score = 0.0
        prev_tok = tokens[i - 1] if i - 1 >= 0 else ""

        if is_strong_boundary_token(prev_tok):
            score += 4.0
        elif is_weak_boundary_token(prev_tok):
            score += 2.0

        rs = roll[i]
        if not math.isnan(rs):
            score += max(0.0, 4.0 - rs)

        score += -abs(i - center) * 0.05
        candidates.append((score, i))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def split_long_chunk(
    tokens: List[str],
    surprisal: List[float],
    segments: List[Tuple[int, int]],
    max_len: int = 24,
    min_len: int = 8,
) -> List[Tuple[int, int]]:
    output: List[Tuple[int, int]] = []

    def _split(s: int, e: int):
        if e - s <= max_len:
            output.append((s, e))
            return

        split_pt = choose_split_point_in_long_chunk(
            tokens=tokens,
            surprisal=surprisal,
            start=s,
            end=e,
            min_len=min_len,
        )
        if split_pt is None:
            split_pt = min(e - min_len, max(s + min_len, (s + e) // 2))

        _split(s, split_pt)
        _split(split_pt, e)

    for s, e in segments:
        _split(s, e)

    return output


def segment_punct_then_refine(
    tokens: List[str],
    surprisal: List[float],
    min_base_len: int = 4,
    max_base_len: int = 48,
    max_merge_len: int = 24,
    event_boundary_threshold: float = 6.0,
    final_max_len: int = 24,
    final_min_len: int = 8,
) -> List[Tuple[int, int]]:
    base = split_by_punctuation(
        tokens=tokens,
        min_base_len=min_base_len,
        max_base_len=max_base_len,
    )
    merged = merge_adjacent_chunks(
        tokens=tokens,
        surprisal=surprisal,
        base_segments=base,
        max_merge_len=max_merge_len,
        event_boundary_threshold=event_boundary_threshold,
    )
    refined = split_long_chunk(
        tokens=tokens,
        surprisal=surprisal,
        segments=merged,
        max_len=final_max_len,
        min_len=final_min_len,
    )
    return refined


# ============================================================
# 6. Segment-level detailed stats
# ============================================================

def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def decode_segment_text(tokenizer, segment_ids: List[int]) -> str:
    return tokenizer.decode(segment_ids, skip_special_tokens=False)


def collect_segment_details(
    tokenizer,
    input_ids: torch.Tensor,
    tokens: List[str],
    surprisal: List[float],
    segments: List[Tuple[int, int]],
    needle_start: Optional[int] = None,
    needle_end: Optional[int] = None,
) -> List[Dict[str, Any]]:
    ids = input_ids[0].tolist()
    rows: List[Dict[str, Any]] = []

    for idx, (s, e) in enumerate(segments):
        seg_ids = ids[s:e]
        seg_tokens = tokens[s:e]
        seg_surprisal = surprisal[s:e]
        seg_text = decode_segment_text(tokenizer, seg_ids)

        contains_needle = False
        ov = 0
        needle_density = None

        if needle_start is not None and needle_end is not None:
            ov = overlap_len(s, e, needle_start, needle_end)
            contains_needle = ov > 0
            if contains_needle:
                needle_density = ov / (e - s)

        rows.append({
            "segment_idx": idx,
            "start": s,
            "end": e,
            "length": e - s,
            "input_ids": seg_ids,
            "tokens": seg_tokens,
            "text": seg_text,
            "surprisal_sum": safe_sum(seg_surprisal),
            "surprisal_mean": safe_mean(seg_surprisal),
            "surprisal_max": safe_max(seg_surprisal),
            "contains_needle": contains_needle,
            "needle_overlap_len": ov,
            "needle_density": needle_density,
        })

    return rows


def summarize_segmentation(
    segment_details: List[Dict[str, Any]],
    slots_per_chunk: int = 2,
) -> Dict[str, Any]:
    lengths = [x["length"] for x in segment_details]
    total_tokens = sum(lengths)
    num_segments = len(lengths)

    avg_len = sum(lengths) / num_segments if num_segments > 0 else float("nan")
    median_len = float(pd.Series(lengths).median()) if lengths else float("nan")
    min_len = min(lengths) if lengths else float("nan")
    max_len = max(lengths) if lengths else float("nan")

    estimated_compression_ratio = (
        total_tokens / (slots_per_chunk * num_segments)
        if num_segments > 0 and slots_per_chunk > 0
        else float("nan")
    )

    needle_segments = [x for x in segment_details if x["contains_needle"]]
    needle_num_covering_segments = len(needle_segments)
    needle_intact = needle_num_covering_segments == 1
    needle_best_density = max(
        [x["needle_density"] for x in needle_segments if x["needle_density"] is not None],
        default=None,
    )

    return {
        "num_segments": num_segments,
        "total_tokens": total_tokens,
        "avg_segment_len": avg_len,
        "median_segment_len": median_len,
        "min_segment_len": min_len,
        "max_segment_len": max_len,
        "estimated_compression_ratio": estimated_compression_ratio,
        "needle_num_covering_segments": needle_num_covering_segments,
        "needle_intact": needle_intact,
        "needle_best_density": needle_best_density,
    }


# ============================================================
# 7. One-shot full analysis
# ============================================================

def analyze_text_segmentation(
    tokenizer,
    model,
    text: str,
    needle: Optional[str] = None,
    insert_pos: int = 0,
    device: Optional[str] = None,
    min_base_len: int = 4,
    max_base_len: int = 48,
    max_merge_len: int = 24,
    event_boundary_threshold: float = 6.0,
    final_max_len: int = 24,
    final_min_len: int = 8,
    slots_per_chunk: int = 2,
) -> Dict[str, Any]:
    input_ids, needle_start, needle_end, tokens, meta = build_input_ids(
        tokenizer=tokenizer,
        text=text,
        needle=needle,
        insert_pos=insert_pos,
    )
    surprisal = compute_surprisal(model, input_ids, device=device)

    segments = segment_punct_then_refine(
        tokens=tokens,
        surprisal=surprisal,
        min_base_len=min_base_len,
        max_base_len=max_base_len,
        max_merge_len=max_merge_len,
        event_boundary_threshold=event_boundary_threshold,
        final_max_len=final_max_len,
        final_min_len=final_min_len,
    )

    segment_details = collect_segment_details(
        tokenizer=tokenizer,
        input_ids=input_ids,
        tokens=tokens,
        surprisal=surprisal,
        segments=segments,
        needle_start=needle_start,
        needle_end=needle_end,
    )

    summary = summarize_segmentation(
        segment_details=segment_details,
        slots_per_chunk=slots_per_chunk,
    )

    return {
        "input_ids": input_ids,
        "tokens": tokens,
        "surprisal": surprisal,
        "segments": segments,
        "segment_details": segment_details,
        "summary": summary,
        "needle_start": needle_start,
        "needle_end": needle_end,
        "meta": meta,
    }


# ============================================================
# 8. Pretty print helpers
# ============================================================

def print_segment_texts(segment_details: List[Dict[str, Any]]):
    print("\nSegment texts\n")
    for seg in segment_details:
        print(
            f"[{seg['start']:4d}, {seg['end']:4d}] "
            f"len={seg['length']:3d} "
            f"contains_needle={seg['contains_needle']} "
            f"needle_density={seg['needle_density']}"
        )
        print(seg["text"])
        print()


def segment_details_to_df(segment_details: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for seg in segment_details:
        rows.append({
            "segment_idx": seg["segment_idx"],
            "start": seg["start"],
            "end": seg["end"],
            "length": seg["length"],
            "surprisal_sum": seg["surprisal_sum"],
            "surprisal_mean": seg["surprisal_mean"],
            "surprisal_max": seg["surprisal_max"],
            "contains_needle": seg["contains_needle"],
            "needle_overlap_len": seg["needle_overlap_len"],
            "needle_density": seg["needle_density"],
            "text_preview": seg["text"][:120].replace("\n", "\\n"),
        })
    return pd.DataFrame(rows)


# ============================================================
# 9. Sweep insert positions
# ============================================================

def sweep_insert_positions(
    tokenizer,
    model,
    text: str,
    needle: str,
    stride: int = 4,
    device: Optional[str] = None,
    min_base_len: int = 4,
    max_base_len: int = 48,
    max_merge_len: int = 24,
    event_boundary_threshold: float = 6.0,
    final_max_len: int = 24,
    final_min_len: int = 8,
    slots_per_chunk: int = 2,
) -> pd.DataFrame:
    bg_ids = tokenizer.encode(text, add_special_tokens=False)
    positions = list(range(0, len(bg_ids) + 1, stride))
    if positions[-1] != len(bg_ids):
        positions.append(len(bg_ids))

    rows = []
    for pos in positions:
        result = analyze_text_segmentation(
            tokenizer=tokenizer,
            model=model,
            text=text,
            needle=needle,
            insert_pos=pos,
            device=device,
            min_base_len=min_base_len,
            max_base_len=max_base_len,
            max_merge_len=max_merge_len,
            event_boundary_threshold=event_boundary_threshold,
            final_max_len=final_max_len,
            final_min_len=final_min_len,
            slots_per_chunk=slots_per_chunk,
        )
        summary = result["summary"]
        rows.append({
            "insert_pos": pos,
            "needle_start": result["needle_start"],
            "needle_end": result["needle_end"],
            **summary,
        })

    return pd.DataFrame(rows)