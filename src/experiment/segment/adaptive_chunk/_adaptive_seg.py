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
# 4. Small helpers
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
# 5. Stage 1: punctuation-first base split
# ============================================================

def split_by_punctuation(
    tokens: List[str],
    min_base_len: int = 4,
    max_base_len: int = 48,
) -> List[Tuple[int, int]]:
    """
    First-pass split by strong punctuation / line break.
    Weak punctuation is ignored in this first stage.
    """
    n = len(tokens)
    segments: List[Tuple[int, int]] = []
    start = 0

    for i, tok in enumerate(tokens):
        seg_len = i - start + 1

        # strong boundary
        if seg_len >= min_base_len and is_strong_boundary_token(tok):
            segments.append((start, i + 1))
            start = i + 1
            continue

        # fallback hard cut
        if seg_len >= max_base_len:
            segments.append((start, i + 1))
            start = i + 1

    if start < n:
        segments.append((start, n))

    return segments


# ============================================================
# 6. Stage 2: merge adjacent chunks with surprisal assistance
# ============================================================

def chunk_stats(
    surprisal: List[float],
    start: int,
    end: int,
    head_k: int = 4,
    tail_k: int = 4,
) -> Dict[str, float]:
    seg = surprisal[start:end]
    head = seg[:head_k]
    tail = seg[-tail_k:]
    return {
        "mean": safe_mean(seg),
        "head_mean": safe_mean(head),
        "tail_mean": safe_mean(tail),
        "sum": sum(v for v in seg if not (isinstance(v, float) and math.isnan(v))),
        "len": end - start,
    }


def boundary_event_strength(
    surprisal: List[float],
    boundary_idx: int,
    look_left: int = 3,
    look_right: int = 3,
) -> float:
    """
    Measure how 'event-like' the boundary is.
    Higher means stronger semantic break around this boundary.
    """
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
    """
    Merge neighboring base segments if:
    - merged length stays under max_merge_len
    - boundary between them is NOT a strong event boundary
    """
    if not base_segments:
        return []

    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = base_segments[0]

    for nxt_s, nxt_e in base_segments[1:]:
        cur_len = cur_e - cur_s
        nxt_len = nxt_e - nxt_s
        total_len = nxt_e - cur_s

        boundary_strength = boundary_event_strength(surprisal, nxt_s)

        # prefer not to merge across strong event boundary
        can_merge = (
            total_len <= max_merge_len
            and boundary_strength < event_boundary_threshold
        )

        # if next chunk starts after strong punctuation, be conservative
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


# ============================================================
# 7. Stage 3: split overly long chunk with punctuation + surprisal
# ============================================================

def choose_split_point_in_long_chunk(
    tokens: List[str],
    surprisal: List[float],
    start: int,
    end: int,
    min_len: int = 8,
) -> Optional[int]:
    """
    Choose a split point inside [start, end) for an overly long chunk.

    Priority:
    1. strong punctuation near middle
    2. weak punctuation near middle
    3. local low surprisal near middle
    """
    seg_len = end - start
    if seg_len <= 2 * min_len:
        return None

    center = (start + end) // 2

    candidates: List[Tuple[float, int]] = []

    # search only in a safe interior range
    left = start + min_len
    right = end - min_len
    if left >= right:
        return None

    roll = rolling_mean(surprisal, window=3)

    for i in range(left, right):
        tok_prev = tokens[i - 1] if i - 1 >= 0 else ""
        score = 0.0

        # prefer split AFTER punctuation, so inspect previous token
        if is_strong_boundary_token(tok_prev):
            score += 4.0
        elif is_weak_boundary_token(tok_prev):
            score += 2.0

        # prefer low surprisal valley
        rs = roll[i]
        if not math.isnan(rs):
            score += max(0.0, 4.0 - rs)

        # prefer closer to middle
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
    """
    Recursively split chunks that exceed max_len.
    """
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
            # fallback hard split near center
            split_pt = min(e - min_len, max(s + min_len, (s + e) // 2))

        _split(s, split_pt)
        _split(split_pt, e)

    for s, e in segments:
        _split(s, e)

    return output


# ============================================================
# 8. Full hybrid chunking pipeline
# ============================================================

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
    """
    Full pipeline:
    1) punctuation-first base split
    2) merge adjacent chunks if boundary not event-like
    3) split overly long merged chunks
    """
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
# 9. Needle coverage analysis
# ============================================================

def analyze_needle_coverage(
    segments: List[Tuple[int, int]],
    needle_start: int,
    needle_end: int,
) -> Dict[str, Any]:
    covering: List[Tuple[int, int, int]] = []
    for idx, (s, e) in enumerate(segments):
        if not (e <= needle_start or s >= needle_end):
            covering.append((idx, s, e))

    result: Dict[str, Any] = {
        "num_covering_segments": len(covering),
        "is_split": len(covering) > 1,
        "covering_segments": covering,
        "needle_len": needle_end - needle_start,
    }

    if len(covering) == 1:
        idx, s, e = covering[0]
        chunk_len = e - s
        needle_len = needle_end - needle_start
        result.update({
            "segment_idx": idx,
            "chunk_start": s,
            "chunk_end": e,
            "chunk_len": chunk_len,
            "needle_density": needle_len / chunk_len if chunk_len > 0 else float("nan"),
            "needle_relative_start": needle_start - s,
            "needle_relative_end": needle_end - s,
            "needle_relative_center": (((needle_start + needle_end) / 2.0) - s) / chunk_len if chunk_len > 0 else float("nan"),
        })
    else:
        result.update({
            "segment_idx": None,
            "chunk_start": None,
            "chunk_end": None,
            "chunk_len": None,
            "needle_density": None,
            "needle_relative_start": None,
            "needle_relative_end": None,
            "needle_relative_center": None,
        })

    return result


# ============================================================
# 10. Print helpers
# ============================================================

def print_local_window(
    tokens: List[str],
    surprisal: List[float],
    needle_start: int,
    needle_end: int,
    left: int = 8,
    right: int = 8,
):
    start = max(0, needle_start - left)
    end = min(len(tokens), needle_end + right)

    print(f"\nLocal window: [{start}, {end})\n")
    print(f"{'idx':>4} {'token':>15} {'surprisal':>10} {'mark':>8}")
    for i in range(start, end):
        mark = "needle" if needle_start <= i < needle_end else ""
        s = surprisal[i]
        s_str = "nan" if (isinstance(s, float) and math.isnan(s)) else f"{s:.3f}"
        print(f"{i:4d} {tokens[i]:>15} {s_str:>10} {mark:>8}")


def print_segments(
    tokens: List[str],
    segments: List[Tuple[int, int]],
    needle_start: int,
    needle_end: int,
):
    print("\nSegments\n")
    for s, e in segments:
        contains = not (e <= needle_start or s >= needle_end)
        print(f"[{s:4d}, {e:4d}] contains_needle={contains}")
        print(" ".join(tokens[s:e]))
        print()


# ============================================================
# 11. Sweep insert positions
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
) -> pd.DataFrame:
    bg_ids = tokenizer.encode(text, add_special_tokens=False)
    positions = list(range(0, len(bg_ids) + 1, stride))
    if positions[-1] != len(bg_ids):
        positions.append(len(bg_ids))

    rows: List[Dict[str, Any]] = []

    for pos in positions:
        input_ids, ns, ne, tokens, meta = build_input_ids(
            tokenizer=tokenizer,
            text=text,
            needle=needle,
            insert_pos=pos,
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

        coverage = analyze_needle_coverage(segments, ns, ne)
        row = {
            "insert_pos": pos,
            "needle_start": ns,
            "needle_end": ne,
            "num_segments": len(segments),
            "segments": segments,
            **coverage,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_results(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    intact = int((df["num_covering_segments"] == 1).sum())
    split = int((df["is_split"] == True).sum())  # noqa: E712

    density = df["needle_density"].dropna()
    chunk_len = df["chunk_len"].dropna()
    center = df["needle_relative_center"].dropna()

    return {
        "total_cases": total,
        "intact_cases": intact,
        "split_cases": split,
        "intact_ratio": intact / total if total > 0 else float("nan"),
        "avg_needle_density": density.mean() if len(density) > 0 else float("nan"),
        "avg_chunk_len": chunk_len.mean() if len(chunk_len) > 0 else float("nan"),
        "avg_relative_center": center.mean() if len(center) > 0 else float("nan"),
    }


def show_bad_cases(df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    tmp = df.copy()
    tmp["split_rank"] = tmp["is_split"].astype(int)
    tmp["density_rank"] = tmp["needle_density"].fillna(-1.0)
    tmp = tmp.sort_values(
        by=["split_rank", "density_rank"],
        ascending=[False, True],
    )
    cols = [
        "insert_pos",
        "needle_start",
        "needle_end",
        "num_covering_segments",
        "is_split",
        "chunk_len",
        "needle_density",
        "needle_relative_center",
    ]
    return tmp[cols].head(top_k)