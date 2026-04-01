from __future__ import annotations

import math
import os
import json
from dataclasses import dataclass
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processor import IronCellCollator
from src.data_processor import AdaptiveIronCellCollator
from src.train_utils import _build_fsdp_auto_wrap_policy
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
    p.add_argument("--truncate_len", type=int, default=0)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--parallel", type=str, default="none", choices=["none", "ddp", "fsdp"])
    p.add_argument("--fsdp_wrap", type=str, default="generator_only", choices=["generator_only", "full"])
    p.add_argument("--fsdp_cpu_offload", type=int, default=0, choices=[0, 1])
    p.add_argument("--fsdp_use_orig_params", type=int, default=1, choices=[0, 1])
    p.add_argument("--fsdp_sync_module_states", type=int, default=0, choices=[0, 1])
    args = p.parse_args()

    did_init_pg = False
    parallel = str(args.parallel).lower()
    use_ddp = parallel == "ddp"
    use_fsdp = parallel == "fsdp"
    use_dist = use_ddp or use_fsdp

    if use_dist:
        if args.device != "auto":
            raise ValueError("--parallel ddp/fsdp requires --device auto.")
        if not torch.distributed.is_available():
            raise RuntimeError("torch.distributed is not available.")
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed eval requires CUDA.")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")
        did_init_pg = True
        device = torch.device("cuda", local_rank)
        rank0 = int(torch.distributed.get_rank()) == 0
    else:
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

    if use_fsdp:
        from torch.distributed.fsdp import (
            CPUOffload,
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )

    if str(args.chunking) == "adaptive":
        dataset = _JsonlDatasetWithChunkLens(str(args.data_path))
        collator = AdaptiveIronCellCollator(tokenizer, truncate_len=None if int(args.truncate_len) <= 0 else int(args.truncate_len))
    else:
        dataset = JsonlDataset(str(args.data_path))
        collator = IronCellCollator(
            tokenizer,
            num_v=2,
            random_gate=0,
            chunk_size=int(args.chunk_size),
            truncate_len=None if int(args.truncate_len) <= 0 else int(args.truncate_len),
        )
    sampler = None
    if _is_dist():
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        sampler=sampler,
        num_workers=0,
        collate_fn=collator,
    )

    from src.attention import TrainStepModule
    step_module = TrainStepModule(model, phase=str(args.phase))
    step_module.to(torch.bfloat16)
    if use_fsdp:
        mp = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16)
        cpu_offload = CPUOffload(offload_params=True) if int(args.fsdp_cpu_offload) == 1 else None
        auto_wrap_policy = _build_fsdp_auto_wrap_policy(model, str(args.fsdp_wrap))
        step_module = FSDP(
            step_module,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp,
            cpu_offload=cpu_offload,
            use_orig_params=bool(int(args.fsdp_use_orig_params)),
            device_id=device,
            sync_module_states=bool(int(args.fsdp_sync_module_states)),
        )
    from src.evaluation import run_eval_loop
    eval_loss, ppl, _ = run_eval_loop(
        step_module,
        loader,
        device,
        max_batches=int(args.max_batches),
        distributed=_is_dist(),
        use_amp=(device.type == "cuda"),
    )

    if rank0:
        print(f"ckpt_dir: {args.ckpt_dir}")
        print(f"data_path: {args.data_path}")
        print(f"phase: {args.phase}")
        print(f"eval_loss: {eval_loss:.6f}")
        print(f"ppl: {ppl:.6f}")
    if did_init_pg:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
