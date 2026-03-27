# src/experiment/surprisal.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. Load Model
# ============================================================

def load_model(model_path, device=None, dtype="bf16"):
    """
    Load HuggingFace causal LM model.

    Args:
        model_path: local HF model path
        device: cuda/cpu
        dtype: bf16/fp16/fp32

    Returns:
        tokenizer, model, device
    """

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# 2. Build Input With Needle
# ============================================================

def build_input_ids(tokenizer, text, needle=None, insert_pos=0):
    """
    Build input_ids, optionally inserting a needle at token index.

    Args:
        tokenizer
        text
        needle: optional string
        insert_pos: token index in background text

    Returns:
        input_ids
        needle_start
        needle_end
        tokens
        meta
    """

    bg_ids = tokenizer.encode(text, add_special_tokens=False)

    insert_pos = max(0, min(insert_pos, len(bg_ids)))

    if needle is None:
        needle_ids = []
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
        "has_bos": tokenizer.bos_token_id is not None,
        "full_len": len(full_ids),
    }

    return input_ids, needle_start, needle_end, tokens, meta
# ============================================================
# 3. Compute Surprisal
# ============================================================

def compute_surprisal(model, input_ids, device=None):
    """
    Compute per-token surprisal.

    Returns:
        surprisal list aligned with tokens
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
        -1,
        shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    surprisal = -token_logp[0].cpu()

    surprisal = torch.cat(
        [torch.tensor([float("nan")]), surprisal],
        dim=0
    )

    return surprisal.tolist()


# ============================================================
# 4. Hidden Jump
# ============================================================

def compute_hidden_jump(model, input_ids, layer_idx=16, device=None):
    """
    Compute hidden state cosine distance between tokens.
    """

    if device is None:
        device = next(model.parameters()).device

    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids,
            output_hidden_states=True
        )

    hidden = outputs.hidden_states[layer_idx][0]  # [T, D]

    h1 = F.normalize(hidden[:-1], dim=-1)
    h2 = F.normalize(hidden[1:], dim=-1)

    cos = (h1 * h2).sum(-1)

    jump = 1 - cos

    jump = torch.cat(
        [torch.tensor([float("nan")], device=jump.device), jump],
        dim=0
    )

    return jump.cpu().tolist()


# ============================================================
# 5. Fixed Segmentation
# ============================================================

def segment_fixed(seq_len, chunk_size=16):
    """
    Fixed window segmentation.
    """

    segments = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        segments.append((start, end))

    return segments


# ============================================================
# 6. Surprisal Accumulation Segmentation
# ============================================================

def segment_surprisal(
    surprisal,
    min_len=8,
    max_len=24,
    threshold=25
):

    segments = []

    start = 0
    accum = 0

    for i in range(len(surprisal)):

        if i == start:
            continue

        if not torch.isnan(torch.tensor(surprisal[i])):
            accum += surprisal[i]

        length = i - start

        if length >= min_len and accum > threshold:
            segments.append((start, i))
            start = i
            accum = 0

        elif length >= max_len:
            segments.append((start, i))
            start = i
            accum = 0

    segments.append((start, len(surprisal)))

    return segments


# ============================================================
# 7. Hidden Jump Segmentation
# ============================================================

def segment_jump(
    jump,
    min_len=8,
    max_len=24,
    jump_threshold=0.15
):

    segments = []

    start = 0

    for i in range(len(jump)):

        if i == start:
            continue

        length = i - start

        if length >= min_len and jump[i] > jump_threshold:
            segments.append((start, i))
            start = i

        elif length >= max_len:
            segments.append((start, i))
            start = i

    segments.append((start, len(jump)))

    return segments


# ============================================================
# 8. Print Token Analysis
# ============================================================

def print_token_analysis(
    tokens,
    surprisal,
    jump,
    needle_start,
    needle_end
):

    print("\nToken Analysis\n")

    print(
        f"{'idx':>4} {'token':>15} {'surprisal':>10} {'jump':>10} needle"
    )

    for i, tok in enumerate(tokens):

        s = surprisal[i] if i < len(surprisal) else float("nan")
        j = jump[i] if i < len(jump) else float("nan")

        mark = (
            "*"
            if needle_start <= i < needle_end
            else ""
        )

        print(
            f"{i:4d} {tok:>15} {s:10.3f} {j:10.3f} {mark}"
        )


# ============================================================
# 9. Print Segmentation Result
# ============================================================

def print_segments(
    tokens,
    segments,
    needle_start,
    needle_end
):

    print("\nSegments\n")

    needle_in_1 = False
    for start, end in segments:

        contains = not (
            end <= needle_start
            or start >= needle_end
        )

        text = " ".join(tokens[start:end])

        print(
            f"[{start:4d},{end:4d}] "
            f"contains_needle={contains}"
        )

        print(text)

        if contains and not needle_in_1:
            needle_in_1 = True
        elif contains:
            return False

    return True



def print_local_window(tokens, surprisal, jump, center_start, center_end, left=8, right=8):
    """
    Print a local token window around [center_start, center_end).
    """
    start = max(0, center_start - left)
    end = min(len(tokens), center_end + right)

    print(f"\nLocal window: [{start}, {end})\n")
    print(f"{'idx':>4} {'token':>15} {'surprisal':>10} {'jump':>10} {'mark':>6}")

    for i in range(start, end):
        mark = ""
        if center_start <= i < center_end:
            mark = "needle"

        s = surprisal[i]
        j = jump[i]

        print(f"{i:4d} {tokens[i]:>15} {s:10.3f} {j:10.3f} {mark:>6}")




def segment_surprisal_protected(
    surprisal,
    min_len=8,
    max_len=24,
    acc_threshold=20.0,
    low_threshold=3.0,
    grace_tokens=4,
):
    segments = []
    start = 0
    accum = 0.0
    armed = False
    armed_at = None

    i = 1
    while i < len(surprisal):
        s = surprisal[i]
        if s != s:  # nan
            i += 1
            continue

        accum += s
        length = i - start + 1

        # 强制切
        if length >= max_len:
            segments.append((start, i + 1))
            start = i + 1
            accum = 0.0
            armed = False
            armed_at = None
            i += 1
            continue

        # 达到累计阈值，进入 armed 状态
        if (not armed) and length >= min_len and accum >= acc_threshold:
            armed = True
            armed_at = i

        # armed 后寻找安全切点
        if armed:
            waited = i - armed_at
            if s < low_threshold:
                segments.append((start, i + 1))
                start = i + 1
                accum = 0.0
                armed = False
                armed_at = None
            elif waited >= grace_tokens:
                segments.append((start, i + 1))
                start = i + 1
                accum = 0.0
                armed = False
                armed_at = None

        i += 1

    if start < len(surprisal):
        segments.append((start, len(surprisal)))

    return segments







def analyze_needle_coverage(segments, needle_start, needle_end):
    covering = []
    for idx, (s, e) in enumerate(segments):
        if not (e <= needle_start or s >= needle_end):
            covering.append((idx, s, e))

    result = {
        "num_covering_segments": len(covering),
        "is_split": len(covering) > 1,
        "covering_segments": covering,
    }

    if len(covering) == 1:
        _, s, e = covering[0]
        chunk_len = e - s
        needle_len = needle_end - needle_start
        result.update({
            "chunk_start": s,
            "chunk_end": e,
            "chunk_len": chunk_len,
            "needle_len": needle_len,
            "needle_density": needle_len / chunk_len,
            "needle_relative_start": needle_start - s,
            "needle_relative_end": needle_end - s,
            "needle_relative_center": (((needle_start + needle_end) / 2) - s) / chunk_len,
        })

    return result








