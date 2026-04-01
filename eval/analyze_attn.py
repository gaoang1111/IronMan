from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(PROJECT_ROOT))

from src.models import IronCellConfig, IronCellModel
from src.data_processor import IronCellCollator
from src.token_utils import add_iron_cell_special_tokens
from src.train_utils import load_model, load_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--phase", type=str, default="phase1")
    parser.add_argument("--javis_num_queries", type=int, default=None)
    parser.add_argument("--javis_query_warmup_samples", type=int, default=None)
    parser.add_argument("--javis_query_warmup_save_path", type=str, default=None)
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=16)
    parser.add_argument("--buffer_size", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def resolve_device(arg_device: str | None) -> torch.device:
    if arg_device:
        return torch.device(arg_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer_from_args(resume_path: str | None, model_name: str | None) -> tuple[object, bool]:
    if resume_path is None and model_name is None:
        raise ValueError("must_set_resume_path_or_model_name")
    args = SimpleNamespace(resume_path=resume_path, model_name=model_name)
    tokenizer, is_resume = load_tokenizer(args)
    return tokenizer, is_resume


def load_model_from_file(
    ckpt_path: Path, tokenizer, device: torch.device, model_name: str | None
) -> IronCellModel:
    ckpt_dir = ckpt_path.parent
    config_path = ckpt_dir / "config.json"
    if config_path.exists():
        config = IronCellConfig.from_pretrained(str(ckpt_dir))
    else:
        base_name = model_name or str(ckpt_dir)
        config = IronCellConfig(
            compressor_model_name=base_name,
            generator_model_name=base_name,
            freeze_compressor=True,
            projector_init_type="identity",
            trainable_components=["javis", "embed_tokens", "special_tokens"],
        )
    setattr(config, "tokenizer_vocab_size", int(len(tokenizer)))
    model = IronCellModel(config).to(device)
    state = torch.load(str(ckpt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("state_dict_mismatch", {"missing": len(missing), "unexpected": len(unexpected)})
    return model


def read_jsonl_texts(path: Path, max_samples: int) -> list[str]:
    raw = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                raw.append(line)
                continue
            if isinstance(obj, str):
                raw.append(obj)
            elif isinstance(obj, dict):
                if "text" in obj:
                    raw.append(str(obj["text"]))
                elif "prompt" in obj:
                    raw.append(str(obj["prompt"]))
                elif len(obj) > 0:
                    raw.append(str(next(iter(obj.values()))))
            else:
                raw.append(str(obj))
    cap = min(16, max_samples)
    if len(raw) >= 8:
        target = max(8, min(cap, len(raw)))
    else:
        target = min(cap, len(raw))
    return raw[:target]


def compute_javis_attn(
    model: IronCellModel,
    batch,
    device: torch.device,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    attn_store: dict[str, torch.Tensor] = {}
    current_mask: dict[str, torch.Tensor] = {}

    def hook(module, inputs, kwargs, output):
        hidden = inputs[0]
        attention_mask = None
        if kwargs:
            attention_mask = kwargs.get("attention_mask")
        if attention_mask is None:
            attention_mask = current_mask.get("mask")

        x = module.pre_kv_hidden(hidden)
        k = module.wk(x)
        v = module.wv(x)
        n = int(x.size(0))
        q = module.q.unsqueeze(0).expand(n, -1, -1)
        q_len = int(q.size(1))
        seq_len = int(k.size(1))
        h = module.num_heads
        d_k = module.hidden_size // h
        q = q.view(n, q_len, h, d_k).transpose(1, 2)
        k = k.view(n, seq_len, h, d_k).transpose(1, 2)
        v = v.view(n, seq_len, h, d_k).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if attention_mask is not None:
            m = attention_mask.to(dtype=torch.bool, device=logits.device).view(n, 1, 1, seq_len)
            logits = logits.masked_fill(~m, torch.finfo(logits.dtype).min)
        attn = torch.softmax(logits.float(), dim=-1).to(dtype=logits.dtype)
        attn_store["attn"] = attn.detach()

    try:
        handle = model.javis.register_forward_hook(hook, with_kwargs=True)
    except TypeError:
        handle = model.javis.register_forward_hook(lambda m, i, o: hook(m, i, {}, o))

    chunk_ids = batch.chunk_input_ids.to(device)
    chunk_mask = batch.chunk_attention_mask.to(device)
    bsz, num_chunks, chunk_len = chunk_ids.shape
    current_mask["mask"] = chunk_mask.view(bsz * num_chunks, chunk_len)

    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        memory = model.compute_compressed_vectors(
            chunk_input_ids=chunk_ids,
            chunk_attention_mask=chunk_mask,
        )
        memory, _ = memory

    handle.remove()
    attn = attn_store["attn"]
    attn = attn.view(bsz, num_chunks, model.javis.num_heads, model.javis.num_queries, chunk_len)
    start = 1
    end = min(start + chunk_size, chunk_len - 1)
    attn_slice = attn[:, :, :, :, start:end]
    attn_avg = attn_slice.mean(dim=(0, 1, 2))
    return memory, attn_avg


def plot_javis_attn(attn_avg: torch.Tensor, output_path: Path) -> None:
    num_queries, seq_len = attn_avg.shape
    fig, ax = plt.subplots(figsize=(max(8, seq_len * 0.5), 3 + num_queries))
    im = ax.imshow(attn_avg.cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Query")
    ax.set_yticks(list(range(num_queries)))
    ax.set_yticklabels([f"V{q + 1}" for q in range(num_queries)])
    ax.set_xticks(list(range(seq_len)))
    ax.set_xticklabels([str(i) for i in range(seq_len)])
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def build_v_positions(memory_positions: torch.Tensor, num_queries: int) -> tuple[list[int], dict[int, str]]:
    first = memory_positions[0]
    labels: dict[int, str] = {}
    for c in range(first.size(0)):
        start = int(first[c].item())
        if start < 0:
            continue
        for q in range(num_queries):
            pos = start + q
            labels[pos] = f"V{c + 1}-{q + 1}"

    pos_set = set()
    for b in range(memory_positions.size(0)):
        for c in range(memory_positions.size(1)):
            start = int(memory_positions[b, c].item())
            if start < 0:
                continue
            for q in range(num_queries):
                pos_set.add(start + q)
    positions = sorted(pos_set)
    return positions, labels


def plot_gen_attn(
    attn_avg: torch.Tensor, output_path: Path, v_positions: list[int], v_labels: dict[int, str]
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attn_avg.cpu().numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Key Index")
    ax.set_ylabel("Query Index")
    for pos in v_positions:
        ax.axvline(pos, color="red", linewidth=0.8, alpha=0.6)
        ax.axhline(pos, color="red", linewidth=0.8, alpha=0.6)
        if pos in v_labels:
            ax.text(pos, -0.5, v_labels[pos], rotation=90, va="bottom", ha="center", color="red", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    resume_path = args.resume_path
    model_name = args.model_name
    tokenizer, is_resume = load_tokenizer_from_args(resume_path, model_name)
    add_iron_cell_special_tokens(tokenizer)

    if resume_path is not None and Path(resume_path).suffix in {".bin", ".pt"}:
        model = load_model_from_file(Path(resume_path), tokenizer, device, model_name)
    else:
        model_args = SimpleNamespace(
            resume_path=resume_path,
            model_name=model_name,
            phase=args.phase,
            javis_num_queries=args.javis_num_queries,
            javis_query_warmup_samples=args.javis_query_warmup_samples,
            javis_query_warmup_save_path=args.javis_query_warmup_save_path,
        )
        model = load_model(model_args, tokenizer, device, is_resume=is_resume)

    model.eval()

    texts = read_jsonl_texts(Path(args.jsonl), args.max_samples)
    if len(texts) == 0:
        raise ValueError("no_valid_texts")

    collator = IronCellCollator(
        tokenizer,
        chunk_size=args.chunk_size,
        buffer_size=args.buffer_size,
        num_v=model.javis.num_queries,
    )
    batch = collator(texts)

    memory, javis_attn_avg = compute_javis_attn(model, batch, device, args.chunk_size)
    javis_path = PROJECT_ROOT / "javis_attn.png"
    plot_javis_attn(javis_attn_avg, javis_path)

    inputs_embeds = model.build_inputs_embeds(
        zipper_input_ids=batch.zipper_input_ids.to(device),
        memory_vectors=memory,
        memory_positions=batch.memory_positions.to(device),
    )
    with torch.no_grad(), torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    ):
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask_2d.to(device),
            position_ids=batch.position_ids.to(device),
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
    last_attn = out.attentions[-1].float()
    avg_attn = last_attn.mean(dim=(0, 1))
    valid_len = int(batch.valid_lens.max().item())
    avg_attn = avg_attn[:valid_len, :valid_len]

    v_positions, v_labels = build_v_positions(batch.memory_positions, model.javis.num_queries)
    v_positions = [p for p in v_positions if p < valid_len]
    gen_path = PROJECT_ROOT / "gen_attn.png"
    plot_gen_attn(avg_attn, gen_path, v_positions, v_labels)

    print("samples", len(texts))
    print("javis_attn", str(javis_path))
    print("gen_attn", str(gen_path))
    print("v_positions", v_positions)


if __name__ == "__main__":
    main()
