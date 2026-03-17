# src/experiment/surprisal_eval.py

from __future__ import annotations

import math
from dataclasses import dataclass
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
    """
    Load HF causal LM and tokenizer.
    """
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
# 2. Build input ids with optional needle
# ============================================================

def build_input_ids(
    tokenizer,
    text: str,
    needle: Optional[str] = None,
    insert_pos: int = 0,
):
    """
    Build token ids from background text, optionally inserting needle
    at background token index insert_pos.

    Returns:
        input_ids: torch.LongTensor [1, T]
        needle_start: int
        needle_end: int
        tokens: List[str]
        meta: Dict
    """
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
    """
    Compute per-token surprisal in nats, aligned to token positions.
    Position 0 is NaN because there is no conditional probability for it.
    """
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
# 4. Local window print
# ============================================================

def print_local_window(
    tokens: List[str],
    surprisal: List[float],
    needle_start: int,
    needle_end: int,
    left: int = 8,
    right: int = 8,
):
    """
    Print local token window around needle span.
    """
    start = max(0, needle_start - left)
    end = min(len(tokens), needle_end + right)

    print(f"\nLocal window: [{start}, {end})\n")
    print(f"{'idx':>4} {'token':>15} {'surprisal':>10} {'mark':>8}")

    for i in range(start, end):
        mark = "needle" if needle_start <= i < needle_end else ""
        s = surprisal[i]
        s_str = "nan" if (isinstance(s, float) and math.isnan(s)) else f"{s:.3f}"
        print(f"{i:4d} {tokens[i]:>15} {s_str:>10} {mark:>8}")


# ============================================================
# 5. Protected surprisal segmentation
# ============================================================

def segment_surprisal_protected(
    surprisal: List[float],
    min_len: int = 8,
    max_len: int = 24,
    acc_threshold: float = 20.0,
    low_threshold: float = 3.0,
    grace_tokens: int = 4,
) -> List[Tuple[int, int]]:
    """
    Adaptive segmentation based on surprisal accumulation + peak protection.

    Logic:
    - Accumulate surprisal from current segment start.
    - Once length >= min_len and accum >= acc_threshold, enter armed state.
    - In armed state, do not cut immediately. Wait for a safe cut:
        * current surprisal drops below low_threshold, or
        * waited grace_tokens already, or
        * max_len reached.
    - This helps avoid cutting in the middle of a high-surprisal event span.
    """
    n = len(surprisal)
    segments: List[Tuple[int, int]] = []

    start = 0
    accum = 0.0
    armed = False
    armed_at: Optional[int] = None

    i = 1
    while i < n:
        s = surprisal[i]
        if not (isinstance(s, float) and math.isnan(s)):
            accum += s

        seg_len = i - start + 1

        # Hard max length cut
        if seg_len >= max_len:
            segments.append((start, i + 1))
            start = i + 1
            accum = 0.0
            armed = False
            armed_at = None
            i += 1
            continue

        # Arm cut after enough information is accumulated
        if (not armed) and seg_len >= min_len and accum >= acc_threshold:
            armed = True
            armed_at = i

        # If armed, wait for a safer cut boundary
        if armed:
            waited = i - int(armed_at)
            safe_to_cut = (
                (not (isinstance(s, float) and math.isnan(s)) and s < low_threshold)
                or waited >= grace_tokens
            )
            if safe_to_cut:
                segments.append((start, i + 1))
                start = i + 1
                accum = 0.0
                armed = False
                armed_at = None

        i += 1

    if start < n:
        segments.append((start, n))

    return segments


# ============================================================
# 6. Analyze needle coverage
# ============================================================

def analyze_needle_coverage(
    segments: List[Tuple[int, int]],
    needle_start: int,
    needle_end: int,
) -> Dict[str, Any]:
    """
    Analyze how segmentation covers needle span.
    """
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
        result.update(
            {
                "segment_idx": idx,
                "chunk_start": s,
                "chunk_end": e,
                "chunk_len": chunk_len,
                "needle_density": needle_len / chunk_len if chunk_len > 0 else float("nan"),
                "needle_relative_start": needle_start - s,
                "needle_relative_end": needle_end - s,
                "needle_relative_center": (
                    (((needle_start + needle_end) / 2.0) - s) / chunk_len
                    if chunk_len > 0
                    else float("nan")
                ),
            }
        )
    else:
        result.update(
            {
                "segment_idx": None,
                "chunk_start": None,
                "chunk_end": None,
                "chunk_len": None,
                "needle_density": None,
                "needle_relative_start": None,
                "needle_relative_end": None,
                "needle_relative_center": None,
            }
        )

    return result


# ============================================================
# 7. Print segments
# ============================================================

def print_segments(
    tokens: List[str],
    segments: List[Tuple[int, int]],
    needle_start: int,
    needle_end: int,
):
    """
    Print segmented token spans and whether they overlap the needle.
    """
    print("\nSegments\n")
    for start, end in segments:
        contains = not (end <= needle_start or start >= needle_end)
        print(f"[{start:4d}, {end:4d}] contains_needle={contains}")
        print(" ".join(tokens[start:end]))
        print()


# ============================================================
# 8. Sweep insert positions
# ============================================================

def sweep_insert_positions(
    tokenizer,
    model,
    text: str,
    needle: str,
    stride: int = 4,
    min_len: int = 8,
    max_len: int = 24,
    acc_threshold: float = 20.0,
    low_threshold: float = 3.0,
    grace_tokens: int = 4,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sweep insert positions over background text and evaluate segmentation quality.

    Returns a DataFrame with one row per insert position.
    """
    bg_ids = tokenizer.encode(text, add_special_tokens=False)
    results: List[Dict[str, Any]] = []

    positions = list(range(0, len(bg_ids) + 1, stride))
    if positions[-1] != len(bg_ids):
        positions.append(len(bg_ids))

    for pos in positions:
        input_ids, needle_start, needle_end, tokens, meta = build_input_ids(
            tokenizer=tokenizer,
            text=text,
            needle=needle,
            insert_pos=pos,
        )
        surprisal = compute_surprisal(model, input_ids, device=device)
        segments = segment_surprisal_protected(
            surprisal=surprisal,
            min_len=min_len,
            max_len=max_len,
            acc_threshold=acc_threshold,
            low_threshold=low_threshold,
            grace_tokens=grace_tokens,
        )
        coverage = analyze_needle_coverage(segments, needle_start, needle_end)

        row: Dict[str, Any] = {
            "insert_pos": pos,
            "needle_start": needle_start,
            "needle_end": needle_end,
            "full_len": meta["full_len"],
            "num_segments": len(segments),
            "segments": segments,
        }
        row.update(coverage)
        results.append(row)

    return pd.DataFrame(results)


# ============================================================
# 9. Quick summary helpers
# ============================================================

def summarize_results(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize segmentation robustness over all insert positions.
    """
    total = len(df)
    intact = int((df["num_covering_segments"] == 1).sum())
    split = int((df["is_split"] == True).sum())  # noqa: E712

    density_series = df["needle_density"].dropna()
    chunk_len_series = df["chunk_len"].dropna()
    center_series = df["needle_relative_center"].dropna()

    return {
        "total_cases": total,
        "intact_cases": intact,
        "split_cases": split,
        "intact_ratio": intact / total if total > 0 else float("nan"),
        "avg_needle_density": density_series.mean() if len(density_series) > 0 else float("nan"),
        "avg_chunk_len": chunk_len_series.mean() if len(chunk_len_series) > 0 else float("nan"),
        "avg_relative_center": center_series.mean() if len(center_series) > 0 else float("nan"),
    }


def show_bad_cases(
    df: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Show the worst cases:
    - split cases first
    - then low-density cases
    """
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


def rolling_mean(values, window=3):
    out = []
    n = len(values)
    half = window // 2
    for i in range(n):
        vals = []
        for j in range(max(0, i-half), min(n, i+half+1)):
            v = values[j]
            if not (isinstance(v, float) and math.isnan(v)):
                vals.append(v)
        if len(vals) == 0:
            out.append(float("nan"))
        else:
            out.append(sum(vals) / len(vals))
    return out

# ============================================================
# 10. Example usage
# ============================================================

if __name__ == "__main__":
    # Example: edit these manually
    model_path = "/default-vepfs/public/user/ga/Iron/models/Llama-3.1-8B"
    text = (
        "The Ise-class battleships were a pair of dreadnought battleships built "
        "for the Imperial Japanese Navy during World War I. Originally intended "
        "to be repeats of the preceding Fusō class, they were redesigned before "
        "construction began."
    )
    print(text[:300])
    exit(0)
    needle = "The secret passcode is tiger."

    tokenizer, model, device = load_model(model_path)

    # Single case
    input_ids, needle_start, needle_end, tokens, meta = build_input_ids(
        tokenizer=tokenizer,
        text=text,
        needle=needle,
        insert_pos=40,
    )
    surprisal = compute_surprisal(model, input_ids, device=device)

    print_local_window(tokens, surprisal, needle_start, needle_end, left=8, right=8)

    # segments = segment_surprisal_protected(
    #     surprisal,
    #     min_len=8,
    #     max_len=24,
    #     acc_threshold=20.0,
    #     low_threshold=3.0,
    #     grace_tokens=4,
    # )

    segments = segment_surprisal_protected(
        surprisal,
        min_len=8,
        max_len=24,
        acc_threshold=20.0,
        low_threshold=3.0,
        grace_tokens=4,
    )
    print_segments(tokens, segments, needle_start, needle_end)
    print(analyze_needle_coverage(segments, needle_start, needle_end))

    # Sweep
    df = sweep_insert_positions(
        tokenizer=tokenizer,
        model=model,
        text=text,
        needle=needle,
        stride=4,
        min_len=8,
        max_len=24,
        acc_threshold=20.0,
        low_threshold=3.0,
        grace_tokens=4,
        device=device,
    )
    print("\nSummary:\n", summarize_results(df))
    print("\nWorst cases:\n", show_bad_cases(df, top_k=10))