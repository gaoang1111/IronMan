from __future__ import annotations

from typing import Literal

from transformers import PretrainedConfig


class IronCellConfig(PretrainedConfig):
    """
    Configuration for Project Iron-Cell (SoulBone).

    This config intentionally focuses on the MVP core routing:
    - a frozen compressor (AutoModel) to produce semantic hidden states
    - a trainable Javis (Cross-Attention projector) to output compressed vectors V
    - a generator (AutoModelForCausalLM) to perform autoregressive modeling
      using a zipper-layout embedding sequence and a custom staircase attention mask.
    """

    model_type = "iron_cell"

    def __init__(
        self,
        compressor_model_name: str = "meta-llama/Meta-Llama-3-8B",
        generator_model_name: str | None = None,
        compression_rate: int = 8,
        projector_init_type: Literal["identity", "gaussian"] = "identity",
        freeze_compressor: bool = True,
        javis_num_heads: int = 16,
        javis_num_queries: int = 1,
        javis_query_group_size: int = 1,
        javis_ln_in: bool = True,
        javis_ln_out: bool = True,
        javis_init_noise_std: float = 0.01,
        javis_query_warmup_samples: int | None = 100,
        javis_query_warmup_save_path: str | None = None,
        trainable_components: list[str] | None = None,
        special_token_ids: list[int] | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            compressor_model_name: HF model id used for the compressor.
            generator_model_name: HF model id used for the generator. If None,
                it defaults to compressor_model_name.
            compression_rate: MVP fixed rate. In this MVP, it is used as a
                semantic knob and for bookkeeping (not a hard constraint).
            projector_init_type: "identity" (default) or "gaussian".
            freeze_compressor: Whether compressor forward runs under no_grad
                and its parameters are frozen.
            javis_num_queries: Number of query vectors per chunk (compression ratio = chunk_size / num_queries).
            javis_query_group_size: Number of layers sharing the same query group.
            trainable_components: Default ["javis", "embed_tokens", "special_tokens"].
        """
        self.compressor_model_name = compressor_model_name
        self.generator_model_name = generator_model_name or compressor_model_name
        self.compression_rate = int(compression_rate)
        self.projector_init_type = projector_init_type
        self.freeze_compressor = bool(freeze_compressor)
        self.javis_num_heads = int(javis_num_heads)
        self.javis_num_queries = int(javis_num_queries)
        self.javis_query_group_size = int(javis_query_group_size)

        self.javis_ln_in = bool(javis_ln_in)
        self.javis_ln_out = bool(javis_ln_out)
        self.javis_init_noise_std = float(javis_init_noise_std)
        self.javis_query_warmup_samples = None if javis_query_warmup_samples is None else int(javis_query_warmup_samples)
        self.javis_query_warmup_save_path = javis_query_warmup_save_path

        default_trainables = ["javis", "embed_tokens", "special_tokens"]
        trainable_components = trainable_components or default_trainables
        if "projector" in trainable_components and "javis" not in trainable_components:
            trainable_components = [("javis" if x == "projector" else x) for x in trainable_components]
        self.trainable_components = trainable_components
        self.special_token_ids = special_token_ids
        super().__init__(**kwargs)
