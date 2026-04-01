"""Checkpoint save/load utilities."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig


def save_checkpoint(model, optimizer, tokenizer, args, step: int) -> None:
    """Save model + optimizer + config for DDP/single-GPU."""
    save_path = Path(args.output_dir) / f"{args.phase}_step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {save_path}...")

    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
    _save_trainer_state(save_path, step=step)
    print("Checkpoint saved.")


def save_checkpoint_fsdp(
    step_module: FSDP, optimizer, tokenizer, args, step: int, *, is_rank0: bool
) -> None:
    """Save checkpoint for FSDP training."""
    save_path = Path(args.output_dir) / f"{args.phase}_step_{step}"
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"Saving checkpoint to {save_path}...")

    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(step_module, StateDictType.FULL_STATE_DICT, cfg):
        full_state = step_module.state_dict()

    if is_rank0:
        print(f"Rank 0 is writing model to {save_path}...")
        iron_state = _extract_iron_state_dict(full_state)
        step_module.module.iron.save_pretrained(save_path, state_dict=iron_state)
        tokenizer.save_pretrained(save_path)
        _save_trainer_state(save_path, step=step)
        print("Model and config saved by Rank 0.")

    rank = int(os.environ.get("RANK", "0"))
    torch.save(optimizer.state_dict(), save_path / f"optimizer_rank{rank}.pt")
    print("Checkpoint saved.")

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if is_rank0:
        print(f"Step {step} checkpoint sync complete.")


def load_checkpoint(optimizer, args) -> int:
    """Load optimizer state and return the step number."""
    if args.resume_path is None:
        return 0

    resume_path = Path(args.resume_path)
    if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {resume_path}")

    if not args.load_weights_only:
        parallel = str(getattr(args, "parallel", "none")).lower()
        if parallel == "fsdp":
            rank = int(os.environ.get("RANK", "0"))
            opt_path = resume_path / f"optimizer_rank{rank}.pt"
        else:
            opt_path = resume_path / "optimizer.pt"

        if opt_path.exists():
            print("Loading optimizer state...")
            optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
        else:
            print("Warning: Optimizer state not found, starting from scratch.")
    else:
        print("Skipping optimizer state (load_weights_only=True).")

    state = _load_trainer_state(resume_path)
    if state is not None and "step" in state:
        return int(state["step"])

    parsed = _parse_step_from_dirname(resume_path)
    if parsed is not None:
        return parsed
    return 0


def _save_trainer_state(save_path: Path, *, step: int) -> None:
    state = {"step": int(step)}
    (save_path / "trainer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def _load_trainer_state(resume_path: Path) -> dict | None:
    p = resume_path / "trainer_state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _parse_step_from_dirname(resume_path: Path) -> int | None:
    m = re.search(r"_step_(\d+)$", resume_path.name)
    return int(m.group(1)) if m else None


def _extract_iron_state_dict(full_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefix = "iron."
    return {k[len(prefix):]: v for k, v in full_state.items() if k.startswith(prefix)}
