from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(PROJECT_ROOT))

from src.configuration_iron_cell import IronCellConfig
from src.data_processor import IronCellCollator
from src.modeling_iron_cell import IronCellModel
from src.token_utils import add_iron_cell_special_tokens, resize_and_smart_init_special_tokens


@dataclass(frozen=True)
class TrainArgs:
    model_name: str = "meta-llama/Meta-Llama-3-8B"
    chunk_size: int = 256
    batch_size: int = 1
    lr: float = 5e-5
    steps: int = 5


def iter_toy_texts() -> Iterable[str]:
    yield "Note: this is a short example text used to validate the pipeline.\nIt is intentionally small."
    yield "Summary: Iron-Cell trains multiple reconstruction stages in one batch using a staircase mask."


def main() -> None:
    args = TrainArgs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    add_iron_cell_special_tokens(tokenizer)

    config = IronCellConfig(
        compressor_model_name=args.model_name,
        generator_model_name=args.model_name,
        freeze_compressor=True,
        projector_init_type="identity",
        trainable_components=["projector", "embed_tokens", "special_tokens"],
    )
    model = IronCellModel(config).to(device)

    resize_and_smart_init_special_tokens(model.generator, tokenizer)
    resize_and_smart_init_special_tokens(model.compressor, tokenizer)

    model.freeze_for_phase_1()
    model.generator.config.use_cache = False
    model.generator.gradient_checkpointing_enable()

    collator = IronCellCollator(tokenizer, chunk_size=args.chunk_size)
    data = list(iter_toy_texts())
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optim_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optim_params, lr=args.lr)

    model.train()
    step = 0
    for batch in loader:
        if step >= args.steps:
            break

        chunk_ids = batch.chunk_input_ids.to(device)
        chunk_mask = batch.chunk_attention_mask.to(device)
        zipper_ids = batch.zipper_input_ids.to(device)
        zipper_pos_ids = batch.zipper_position_ids.to(device)
        mem_pos = batch.memory_positions.to(device)
        labels = batch.labels.to(device)
        attn_2d = batch.attention_mask_2d.to(device)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            memory = model.compute_compressed_vectors(
                chunk_input_ids=chunk_ids,
                chunk_attention_mask=chunk_mask,
            )
            inputs_embeds = model.build_inputs_embeds(
                zipper_input_ids=zipper_ids,
                memory_vectors=memory,
                memory_positions=mem_pos,
            )
            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_2d,
                position_ids=zipper_pos_ids,
                labels=labels,
            )
            loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        print(f"[step {step}] loss={loss.item():.4f}")


if __name__ == "__main__":
    main()

