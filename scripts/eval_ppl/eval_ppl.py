from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processor import IronCellCollator
from src.data_processor import AdaptiveIronCellCollator
from src.hack_llama_ddp import TrainStepModuleForFullLayersKVInjection as TrainStepModule
from src.train_utils import JsonlDataset, load_model, load_tokenizer


@dataclass(frozen=True)
class _Args:
    model_name: str
    phase: str
    resume_path: str | None
    load_weights_only: bool
    parallel: str


def _is_dist() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class _JsonlDatasetWithChunkLens(torch.utils.data.Dataset):
    def __init__(self, file_path: str):
        self.items: list[tuple[str, int, list[int]]] = []
        print(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                text = obj.get("text")
                if not isinstance(text, str):
                    raise ValueError("Each jsonl line must contain 'text' as str.")
                lens = obj.get("chunk_lens", obj.get("raw_chunk_lens"))
                if lens is None:
                    raise ValueError("Adaptive eval requires 'chunk_lens' or 'raw_chunk_lens' in jsonl.")
                if not isinstance(lens, list) or not all(isinstance(x, int) for x in lens):
                    raise ValueError("'chunk_lens' must be a list[int].")
                self.items.append((text, int(i), lens))
        print(f"Loaded {len(self.items)} samples.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--phase", type=str, default="phase2", choices=["phase1", "phase2"])
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--chunk_size", type=int, default=16)
    p.add_argument("--chunking", type=str, default="fixed", choices=["fixed", "adaptive"])
    p.add_argument("--max_batches", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    args = p.parse_args()

    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rank0 = True
    if _is_dist():
        rank0 = int(os.environ.get("RANK", "0")) == 0

    run_args = _Args(
        model_name="",
        phase=str(args.phase),
        resume_path=str(args.ckpt_dir),
        load_weights_only=True,
        parallel="none",
    )

    tokenizer, is_resume = load_tokenizer(run_args)
    model = load_model(run_args, tokenizer, device, is_resume=is_resume)
    model.eval()

    if str(args.chunking) == "adaptive":
        dataset = _JsonlDatasetWithChunkLens(str(args.data_path))
        collator = AdaptiveIronCellCollator(tokenizer)
    else:
        dataset = JsonlDataset(str(args.data_path))
        collator = IronCellCollator(tokenizer, chunk_size=int(args.chunk_size))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    step_module = TrainStepModule(model, phase=str(args.phase))
    sum_loss_times_tokens = torch.zeros((), device=device, dtype=torch.float32)
    sum_tokens = torch.zeros((), device=device, dtype=torch.float32)

    use_amp = device.type == "cuda"
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if args.max_batches and i >= int(args.max_batches):
                break
            n_tokens_cpu = (batch.labels != -100).sum().to(dtype=torch.float32)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, *_ = step_module(batch)
            sum_loss_times_tokens += loss.detach().float() * n_tokens_cpu.to(device=device)
            sum_tokens += n_tokens_cpu.to(device=device)

    if _is_dist():
        torch.distributed.all_reduce(sum_loss_times_tokens, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sum_tokens, op=torch.distributed.ReduceOp.SUM)

    denom = float(sum_tokens.clamp(min=1.0).item())
    eval_loss = float((sum_loss_times_tokens / denom).item())
    ppl = math.exp(min(eval_loss, 80.0))

    if rank0:
        print(f"ckpt_dir: {args.ckpt_dir}")
        print(f"data_path: {args.data_path}")
        print(f"phase: {args.phase}")
        print(f"eval_loss: {eval_loss:.6f}")
        print(f"ppl: {ppl:.6f}")


if __name__ == "__main__":
    main()
