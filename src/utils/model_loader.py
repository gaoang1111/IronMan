"""Model and tokenizer loading utilities."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from src.models import IronCellConfig, IronCellModel
from src.token_utils import (
    IronCellSpecialTokens,
    add_iron_cell_special_tokens,
    resize_and_smart_init_special_tokens,
)


def load_tokenizer(args) -> tuple[AutoTokenizer, bool]:
    """Load tokenizer from checkpoint or base model."""
    is_resume = args.resume_path is not None
    tok_src = args.resume_path if is_resume else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    _ensure_pad_and_bos(tokenizer)

    if is_resume:
        _validate_required_special_tokens(tokenizer)
    else:
        add_iron_cell_special_tokens(tokenizer)
        _validate_required_special_tokens(tokenizer)
    return tokenizer, is_resume


def load_model(
    args, tokenizer: AutoTokenizer, device: torch.device, *, is_resume: bool
) -> IronCellModel:
    """Load IronCellModel from checkpoint or create new."""
    if is_resume:
        config = IronCellConfig.from_pretrained(args.resume_path)
        setattr(config, "tokenizer_vocab_size", int(len(tokenizer)))
        model = IronCellModel.from_pretrained(args.resume_path, config=config).to(device)

        gen_vocab = int(model.generator.get_input_embeddings().weight.size(0))
        comp_vocab = int(model.compressor.get_input_embeddings().weight.size(0))
        if gen_vocab != len(tokenizer) or comp_vocab != len(tokenizer):
            raise ValueError(
                f"Tokenizer/model vocab mismatch: tokenizer={len(tokenizer)}, "
                f"generator={gen_vocab}, compressor={comp_vocab}. "
                "Load tokenizer from the same checkpoint directory."
            )
        return model

    config = IronCellConfig(
        compressor_model_name=args.model_name,
        generator_model_name=args.model_name,
        freeze_compressor=(args.phase == "phase1"),
        projector_init_type="identity",
        trainable_components=["javis", "embed_tokens", "special_tokens"],
        javis_query_warmup_samples=getattr(args, "javis_query_warmup_samples", None),
        javis_query_warmup_save_path=getattr(args, "javis_query_warmup_save_path", None),
        javis_num_queries=getattr(args, "javis_num_queries", 1),
        javis_query_group_size=getattr(args, "javis_query_group_size", 1),
    )
    model = IronCellModel(config).to(device)
    resize_and_smart_init_special_tokens(model.generator, tokenizer)
    resize_and_smart_init_special_tokens(model.compressor, tokenizer)
    return model


def _ensure_pad_and_bos(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id to set pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must define eos_token_id to set bos_token.")
        tokenizer.bos_token = tokenizer.eos_token


def _get_default_special_token_ids(tokenizer: AutoTokenizer) -> list[int]:
    tokens = IronCellSpecialTokens()
    return [
        int(tokenizer.convert_tokens_to_ids(tokens.soc_token)),
        int(tokenizer.convert_tokens_to_ids(tokens.eoc_token)),
        int(tokenizer.convert_tokens_to_ids(tokens.v_none_token)),
    ]


def _validate_required_special_tokens(tokenizer: AutoTokenizer) -> None:
    ids = _get_default_special_token_ids(tokenizer)
    if any(i < 0 for i in ids):
        raise ValueError(
            "Tokenizer missing required special tokens (<soc>, <eoc>, <v_none>). "
            "For resume: tokenizer must come from the checkpoint directory."
        )
