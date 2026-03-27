from __future__ import annotations

import math
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def rolling_mean(values: List[float], window: int = 3) -> List[float]:
    assert window >= 1 and window % 2 == 1
    n = len(values)
    half = window // 2
    out = []
    for i in range(n):
        vals = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            v = values[j]
            if not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        out.append(sum(vals) / len(vals) if vals else float("nan"))
    return out


def load_model(model_path: str, device: Optional[str] = None, dtype: str = "bf16"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch_dtype)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def build_input_ids(tokenizer, text: str, needle: Optional[str] = None, insert_pos: int = 0):
    bg_ids = tokenizer.encode(text, add_special_tokens=False)
    insert_pos = max(0, min(insert_pos, len(bg_ids)))

    needle_ids = tokenizer.encode(needle, add_special_tokens=False) if needle else []

    full_ids = bg_ids[:insert_pos] + needle_ids + bg_ids[insert_pos:]
    needle_start = insert_pos
    needle_end = insert_pos + len(needle_ids)

    if tokenizer.bos_token_id is not None:
        full_ids = [tokenizer.bos_token_id] + full_ids
        needle_start += 1
        needle_end += 1

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    tokens = tokenizer.convert_ids_to_tokens(full_ids)
    return input_ids, needle_start, needle_end, tokens


def compute_surprisal(model, input_ids: torch.Tensor, device: Optional[str] = None) -> List[float]:
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
# Step 1: 基础切分
# ============================================================

def split_by_strong_punct(tokens: List[str], min_base_len: int = 4, max_base_len: int = 64):
    segments = []
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

    if start < len(tokens):
        segments.append((start, len(tokens)))

    return segments


# ============================================================
# Step 2: 短块合并
# ============================================================

def boundary_strength(surprisal: List[float], boundary_idx: int, right_k: int = 3) -> float:
    vals = surprisal[boundary_idx: min(len(surprisal), boundary_idx + right_k)]
    return safe_mean(vals)


def merge_short_chunks(
    tokens: List[str],
    surprisal: List[float],
    base_segments: List[Tuple[int, int]],
    target_min_len: int = 8,
    target_max_len: int = 20,
    merge_upper_bound: int = 24,
    event_boundary_threshold: float = 5.0,
):
    if not base_segments:
        return []

    merged = []
    cur_s, cur_e = base_segments[0]

    for nxt_s, nxt_e in base_segments[1:]:
        cur_len = cur_e - cur_s
        nxt_len = nxt_e - nxt_s
        total_len = nxt_e - cur_s

        b_strength = boundary_strength(surprisal, nxt_s)

        should_merge = False

        # 如果当前块太短，优先考虑合并
        if cur_len < target_min_len:
            if total_len <= merge_upper_bound and b_strength < event_boundary_threshold:
                should_merge = True

        # 或者两个都不长，且边界不强
        elif cur_len <= target_max_len and nxt_len <= target_max_len:
            if total_len <= target_max_len and b_strength < event_boundary_threshold:
                should_merge = True

        if should_merge:
            cur_e = nxt_e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = nxt_s, nxt_e

    merged.append((cur_s, cur_e))
    return merged


# ============================================================
# Step 3: 长块拆分
# ============================================================

def choose_split_point(
    tokens: List[str],
    surprisal: List[float],
    start: int,
    end: int,
    min_len: int = 8,
):
    seg_len = end - start
    if seg_len <= 2 * min_len:
        return None

    center = (start + end) // 2
    left = start + min_len
    right = end - min_len
    if left >= right:
        return None

    roll = rolling_mean(surprisal, window=3)

    candidates = []
    for i in range(left, right):
        prev_tok = tokens[i - 1] if i - 1 >= 0 else ""
        score = 0.0

        # 优先弱标点
        if is_weak_boundary_token(prev_tok):
            score += 4.0
        elif is_strong_boundary_token(prev_tok):
            score += 2.0

        # 其次低 surprisal
        rs = roll[i]
        if not math.isnan(rs):
            score += max(0.0, 3.0 - rs)

        # 靠近中间更好
        score += -abs(i - center) * 0.05

        candidates.append((score, i))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]


def split_overlong_chunks(
    tokens: List[str],
    surprisal: List[float],
    segments: List[Tuple[int, int]],
    split_max_len: int = 24,
    min_len: int = 8,
):
    output = []

    def _split(s, e):
        if e - s <= split_max_len:
            output.append((s, e))
            return

        split_pt = choose_split_point(tokens, surprisal, s, e, min_len=min_len)
        if split_pt is None:
            split_pt = min(e - min_len, max(s + min_len, (s + e) // 2))

        _split(s, split_pt)
        _split(split_pt, e)

    for s, e in segments:
        _split(s, e)

    return output


# ============================================================
# Full pipeline
# ============================================================

def adaptive_segment(
    tokens: List[str],
    surprisal: List[float],
    min_base_len: int = 4,
    max_base_len: int = 64,
    target_min_len: int = 8,
    target_max_len: int = 20,
    merge_upper_bound: int = 24,
    event_boundary_threshold: float = 5.0,
    split_max_len: int = 24,
    split_min_len: int = 8,
):
    base_segments = split_by_strong_punct(
        tokens=tokens,
        min_base_len=min_base_len,
        max_base_len=max_base_len,
    )

    merged_segments = merge_short_chunks(
        tokens=tokens,
        surprisal=surprisal,
        base_segments=base_segments,
        target_min_len=target_min_len,
        target_max_len=target_max_len,
        merge_upper_bound=merge_upper_bound,
        event_boundary_threshold=event_boundary_threshold,
    )

    final_segments = split_overlong_chunks(
        tokens=tokens,
        surprisal=surprisal,
        segments=merged_segments,
        split_max_len=split_max_len,
        min_len=split_min_len,
    )

    return final_segments


def print_segments(tokens: List[str], segments: List[Tuple[int, int]], needle_start: int, needle_end: int):
    print("\nSegments\n")
    for s, e in segments:
        contains = not (e <= needle_start or s >= needle_end)
        print(f"[{s:4d}, {e:4d}] contains_needle={contains}")
        print(" ".join(tokens[s:e]))
        print()