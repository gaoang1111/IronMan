"""
Project Iron-Cell (SoulBone)

This directory contains the minimal core modules used by the Iron-Cell MVP:

- configuration_iron_cell.py: config definition
- modeling_iron_cell.py: IronCellModel (compressor + projector + generator)
- data_processor/: zipper layout + staircase mask builder
- token_utils.py: special token handling + smart initialization
"""

from .configuration_iron_cell import IronCellConfig
from .data_processor import AdaptiveIronCellCollator, AdaptiveZipperBuilder, IronCellCollator, ZipperBatch
from .modeling_iron_cell import IronCellModel
from .token_utils import (
    add_iron_cell_special_tokens,
    resize_and_smart_init_special_tokens,
)

__all__ = [
    "IronCellConfig",
    "IronCellModel",
    "IronCellCollator",
    "ZipperBatch",
    "AdaptiveZipperBuilder",
    "AdaptiveIronCellCollator",
    "add_iron_cell_special_tokens",
    "resize_and_smart_init_special_tokens",
]
