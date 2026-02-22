from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class IronCellSpecialTokens:
    """Container for Iron-Cell special tokens."""

    soc_token: str = "<soc>"
    eoc_token: str = "<eoc>"
    v_none_token: str = "<v_none>"


def add_iron_cell_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    tokens: IronCellSpecialTokens = IronCellSpecialTokens(),
) -> dict[str, int]:
    """
    Add Iron-Cell special tokens to a tokenizer.

    Returns:
        A dict mapping token name to token id.
    """
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [tokens.soc_token, tokens.eoc_token, tokens.v_none_token]}
    )
    _ = added

    soc_id = tokenizer.convert_tokens_to_ids(tokens.soc_token)
    eoc_id = tokenizer.convert_tokens_to_ids(tokens.eoc_token)
    v_none_id = tokenizer.convert_tokens_to_ids(tokens.v_none_token)
    return {"soc_id": int(soc_id), "eoc_id": int(eoc_id), "v_none_id": int(v_none_id)}


def _mean_embedding_of_text(
    tokenizer: PreTrainedTokenizerBase,
    embedding_weight: torch.Tensor,
    text: str,
) -> torch.Tensor:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) == 0:
        return embedding_weight.mean(dim=0)
    token_ids_t = torch.tensor(token_ids, device=embedding_weight.device, dtype=torch.long)
    return embedding_weight.index_select(0, token_ids_t).mean(dim=0)


def resize_and_smart_init_special_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: IronCellSpecialTokens = IronCellSpecialTokens(),
    soc_init_candidates: Iterable[str] = ("Summary", "Note"),
    eoc_init_text: str = ":",
    v_none_init_text: str = "none",
    std_noise: float = 1e-3,
) -> None:
    """
    Safety + Smart Init:

    - Ensures `resize_token_embeddings(len(tokenizer))` is called.
    - Initializes:
        <soc> from the embedding of "Summary" (or "Note" fallback).
        <eoc> from the embedding of "\\n".

    Notes:
        This function operates on the provided model's input embedding table.
        If you use separate tokenizer/model instances for compressor and generator,
        call this function for both (or at least for the generator).
    """
    model.resize_token_embeddings(len(tokenizer))

    soc_id = int(tokenizer.convert_tokens_to_ids(tokens.soc_token))
    eoc_id = int(tokenizer.convert_tokens_to_ids(tokens.eoc_token))
    v_none_id = int(tokenizer.convert_tokens_to_ids(tokens.v_none_token))

    embed = model.get_input_embeddings()
    weight = embed.weight.data

    soc_vec = None
    for cand in soc_init_candidates:
        cand_ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(cand_ids) > 0:
            soc_vec = weight[cand_ids].mean(dim=0)
            break
    if soc_vec is None:
        soc_vec = weight.mean(dim=0)

    eoc_vec = _mean_embedding_of_text(tokenizer, weight, eoc_init_text)
    v_none_vec = _mean_embedding_of_text(tokenizer, weight, v_none_init_text)

    noise_soc = torch.randn_like(soc_vec) * std_noise
    noise_eoc = torch.randn_like(eoc_vec) * std_noise
    noise_v_none = torch.randn_like(v_none_vec) * std_noise

    weight[soc_id].copy_(soc_vec + noise_soc)
    weight[eoc_id].copy_(eoc_vec + noise_eoc)
    weight[v_none_id].copy_(v_none_vec + noise_v_none)
