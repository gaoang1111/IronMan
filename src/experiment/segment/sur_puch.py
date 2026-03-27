from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


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

    has_bos = tokenizer.bos_token_id is not None
    if has_bos:
        full_ids = [tokenizer.bos_token_id] + full_ids
        needle_start += 1
        needle_end += 1

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    tokens = tokenizer.convert_ids_to_tokens(full_ids)

    meta = {
        "bg_len": len(bg_ids),
        "needle_len": len(needle_ids),
        "insert_pos": insert_pos,
        "has_bos": has_bos,
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
    token_logp = log_probs.gather(
        -1, shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    surprisal = -token_logp[0].float().cpu()
    surprisal = torch.cat(
        [torch.tensor([float("nan")]), surprisal],
        dim=0,
    )
    return surprisal.tolist()


# ============================================================
# 4. Rolling mean
# ============================================================

def rolling_mean(values: List[float], window: int = 3) -> List[float]:
    assert window >= 1 and window % 2 == 1, "window should be odd >= 1"

    n = len(values)
    half = window // 2
    out: List[float] = []

    for i in range(n):
        vals = []
        for j in range(max(0, i - half), min(n, i + half + 1)):
            v = values[j]
            if not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        if len(vals) == 0:
            out.append(float("nan"))
        else:
            out.append(sum(vals) / len(vals))
    return out


# ============================================================
# 5. Helpers for boundary-aware segmentation
# ============================================================

PUNCT_TOKENS = {
    ".", ",", ";", ":", "!", "?", "Ġ.", "Ġ,", "Ġ;", "Ġ:", "Ġ!", "Ġ?", "Ċ"
}


def is_punct_token(tok: str) -> bool:
    if tok in PUNCT_TOKENS:
        return True
    stripped = tok.strip("Ġ")
    return stripped in {".", ",", ";", ":", "!", "?"}


def is_local_min(arr: List[float], i: int) -> bool:
    if i <= 0 or i >= len(arr) - 1:
        return False
    x = arr[i]
    if math.isnan(x):
        return False
    l = arr[i - 1]
    r = arr[i + 1]
    if math.isnan(l) or math.isnan(r):
        return False
    return x <= l and x <= r


def boundary_score(
    idx: int,
    tokens: List[str],
    roll: List[float],
    punct_bonus: float = 2.0,
    local_min_bonus: float = 1.0,
    high_peak_penalty_scale: float = 0.5,
) -> float:
    """
    Higher score => better cut point.
    Prefer punctuation and low rolling surprisal.
    Penalize cutting in high rolling surprisal region.
    """
    if idx < 0 or idx >= len(tokens):
        return -1e9

    rs = roll[idx]
    if math.isnan(rs):
        return -1e9

    score = 0.0

    # Lower rolling surprisal is better
    score += -rs

    # Punctuation bonus
    if is_punct_token(tokens[idx]):
        score += punct_bonus

    # Local minimum bonus
    if is_local_min(roll, idx):
        score += local_min_bonus

    # Penalize cutting in high-surprisal region
    score -= high_peak_penalty_scale * max(0.0, rs - 4.0)

    return score


# ============================================================
# 6. Main segmentation strategy
# ============================================================

def segment_surprisal_boundary_aware(
    surprisal: List[float],
    tokens: List[str],
    min_len: int = 8,
    max_len: int = 24,
    acc_threshold: float = 20.0,
    roll_window: int = 3,
    event_threshold: float = 6.0,
    event_min_span: int = 7,
    search_window: int = 6,
    punct_bonus: float = 2.0,
    local_min_bonus: float = 1.0,
) -> List[Tuple[int, int]]:
    """
    Segmentation strategy:
    1) accumulate surprisal
    2) when information is enough, arm cut
    3) if inside an event span, protect event span from being cut
    4) after protection, search for best boundary in a small future window

    Returns:
        List of (start, end) segments
    """
    n = len(surprisal)
    roll = rolling_mean(surprisal, window=roll_window)
    segments: List[Tuple[int, int]] = []

    start = 0
    accum = 0.0

    i = 1
    while i < n:
        s = surprisal[i]
        if not (isinstance(s, float) and math.isnan(s)):
            accum += s

        seg_len = i - start + 1

        # Force cut at max_len
        if seg_len >= max_len:
            segments.append((start, i + 1))
            start = i + 1
            accum = 0.0
            i += 1
            continue

        # Not ready yet
        if seg_len < min_len or accum < acc_threshold:
            i += 1
            continue

        # We are now armed and can search for a better cut
        # First detect if we are in a high-surprisal event
        event_start = i
        if not math.isnan(roll[i]) and roll[i] >= event_threshold:
            # protect a minimum event span
            protected_until = min(n - 1, i + event_min_span - 1)
        else:
            protected_until = i

        # Boundary search starts after the protected region
        search_start = protected_until
        search_end = min(n - 1, protected_until + search_window)

        best_idx = None
        best_score = -1e18

        for j in range(search_start, search_end + 1):
            sc = boundary_score(
                idx=j,
                tokens=tokens,
                roll=roll,
                punct_bonus=punct_bonus,
                local_min_bonus=local_min_bonus,
            )
            if sc > best_score:
                best_score = sc
                best_idx = j

        # Fallback
        if best_idx is None:
            best_idx = search_end

        cut_end = best_idx + 1
        segments.append((start, cut_end))
        start = cut_end
        accum = 0.0
        i = cut_end
        continue

    if start < n:
        segments.append((start, n))

    return segments


# ============================================================
# 7. Print and analysis utilities
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
    for start, end in segments:
        contains = not (end <= needle_start or start >= needle_end)
        print(f"[{start:4d}, {end:4d}] contains_needle={contains}")
        print(" ".join(tokens[start:end]))
        print()


def analyze_needle_coverage(
    segments: List[Tuple[int, int]],
    needle_start: int,
    needle_end: int,
) -> Dict[str, Any]:
    covering: List[Tuple[int, int, int]] = []
    for idx, (s, e) in enumerate(segments):
        overlap = not (e <= needle_start or s >= needle_end)
        if overlap:
            covering.append((idx, s, e))

    result: Dict[str, Any] = {
        "num_covering_segments": len(covering),
        "is_split": len(covering) > 1,
        "covering_segments": covering,
    }

    needle_len = needle_end - needle_start
    result["needle_len"] = needle_len

    if len(covering) == 1:
        idx, s, e = covering[0]
        chunk_len = e - s
        result.update({
            "segment_idx": idx,
            "chunk_start": s,
            "chunk_end": e,
            "chunk_len": chunk_len,
            "needle_density": needle_len / chunk_len if chunk_len > 0 else float("nan"),
            "needle_relative_start": needle_start - s,
            "needle_relative_end": needle_end - s,
            "needle_relative_center": (
                (((needle_start + needle_end) / 2.0) - s) / chunk_len
                if chunk_len > 0 else float("nan")
            ),
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
# 8. Sweep insert positions
# ============================================================

def sweep_insert_positions(
    tokenizer,
    model,
    text: str,
    needle: str,
    stride: int = 4,
    device: Optional[str] = None,
    min_len: int = 8,
    max_len: int = 24,
    acc_threshold: float = 20.0,
    roll_window: int = 3,
    event_threshold: float = 6.0,
    event_min_span: int = 7,
    search_window: int = 6,
    punct_bonus: float = 2.0,
    local_min_bonus: float = 1.0,
) -> pd.DataFrame:
    bg_ids = tokenizer.encode(text, add_special_tokens=False)
    positions = list(range(0, len(bg_ids) + 1, stride))
    if positions[-1] != len(bg_ids):
        positions.append(len(bg_ids))

    rows: List[Dict[str, Any]] = []

    for pos in positions:
        input_ids, needle_start, needle_end, tokens, meta = build_input_ids(
            tokenizer=tokenizer,
            text=text,
            needle=needle,
            insert_pos=pos,
        )
        surprisal = compute_surprisal(model, input_ids, device=device)

        segments = segment_surprisal_boundary_aware(
            surprisal=surprisal,
            tokens=tokens,
            min_len=min_len,
            max_len=max_len,
            acc_threshold=acc_threshold,
            roll_window=roll_window,
            event_threshold=event_threshold,
            event_min_span=event_min_span,
            search_window=search_window,
            punct_bonus=punct_bonus,
            local_min_bonus=local_min_bonus,
        )

        coverage = analyze_needle_coverage(segments, needle_start, needle_end)

        row = {
            "insert_pos": pos,
            "needle_start": needle_start,
            "needle_end": needle_end,
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
    rel_center = df["needle_relative_center"].dropna()

    return {
        "total_cases": total,
        "intact_cases": intact,
        "split_cases": split,
        "intact_ratio": intact / total if total > 0 else float("nan"),
        "avg_needle_density": density.mean() if len(density) > 0 else float("nan"),
        "avg_chunk_len": chunk_len.mean() if len(chunk_len) > 0 else float("nan"),
        "avg_relative_center": rel_center.mean() if len(rel_center) > 0 else float("nan"),
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