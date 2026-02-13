from __future__ import annotations

from typing import Literal

from transformers import PretrainedConfig


class IronCellConfig(PretrainedConfig):
    """
    Configuration for Project Iron-Cell (SoulBone).

    This config intentionally focuses on the MVP core routing:
    - a frozen compressor (AutoModel) to produce semantic hidden states
    - a trainable projector (Linear) to output compressed vectors V
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
            trainable_components: Default ["projector", "embed_tokens", "special_tokens"].
        """
        self.compressor_model_name = compressor_model_name
        self.generator_model_name = generator_model_name or compressor_model_name
        self.compression_rate = int(compression_rate)
        self.projector_init_type = projector_init_type
        self.freeze_compressor = bool(freeze_compressor)
        self.trainable_components = trainable_components or [
            "projector",
            "embed_tokens",
            "special_tokens",
        ]
        self.special_token_ids = special_token_ids
        super().__init__(**kwargs)
