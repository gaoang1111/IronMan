from __future__ import annotations

import os
from pathlib import Path

import torch
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
os.sys.path.insert(0, str(SRC_DIR))

from configuration_iron_cell import IronCellConfig
from data_processor import IronCellCollator
from modeling_iron_cell import IronCellModel
from token_utils import add_iron_cell_special_tokens, resize_and_smart_init_special_tokens


def main() -> None:
    model_name = "meta-llama/Meta-Llama-3-8B"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    add_iron_cell_special_tokens(tokenizer)

    config = IronCellConfig(compressor_model_name=model_name, generator_model_name=model_name, freeze_compressor=True)
    model = IronCellModel(config).to(device)
    resize_and_smart_init_special_tokens(model.generator, tokenizer)
    resize_and_smart_init_special_tokens(model.compressor, tokenizer)

    model.eval()

    text = "Summary: masked parallel training reduces sequential dependency.\nNow we test a forward pass."
    collator = IronCellCollator(tokenizer, chunk_size=64)
    batch = collator([text])

    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
        memory = model.compute_compressed_vectors(
            chunk_input_ids=batch.chunk_input_ids.to(device),
            chunk_attention_mask=batch.chunk_attention_mask.to(device),
        )
        inputs_embeds = model.build_inputs_embeds(
            zipper_input_ids=batch.zipper_input_ids.to(device),
            memory_vectors=memory,
            memory_positions=batch.memory_positions.to(device),
        )
        out = model(
            inputs_embeds=inputs_embeds,
            attention_mask=batch.attention_mask_2d.to(device),
        )

    logits = out.logits[0, -1]
    next_id = int(torch.argmax(logits).item())
    print("next_token:", tokenizer.decode([next_id]))


if __name__ == "__main__":
    main()

