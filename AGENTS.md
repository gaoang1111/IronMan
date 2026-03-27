# AGENTS.md — IronCell (Project Iron-Cell / SoulBone)

## Project Overview

IronCell is a PyTorch research project for autoregressive compressed memory using
homologous model differentiation. It pairs two clones of the same LLM (Llama 3.1 8B):
a **Compressor** and a **Generator**, bridged by a trainable cross-attention module
called **Javis**. The system compresses long-context KV caches by 8:1 via a
"zipper layout" embedding sequence with staircase attention masks.

## Repository Layout

```
src/                          Core Python package (imported as `src.*`)
  modeling_iron_cell.py       Javis module + IronCellModel (compressor + generator)
  configuration_iron_cell.py  IronCellConfig (HF PretrainedConfig subclass)
  train.py                    Distributed training entry point
  eval.py                     Eval entry point (loss on eval set)
  train_utils.py              Checkpoint save/load, dataset, tokenizer helpers
  token_utils.py              Special token handling (<soc>, <eoc>, <v_none>)
  hack_llama_ddp.py           DDP-compatible LlamaAttention monkey-patch for deep KV injection
  hack_llama_fsdp.py          FSDP-compatible variant of the above
  data_processor/             Zipper layout + staircase mask builder (fixed + adaptive)
    fixed.py                  Fixed chunk_size collator (IronCellCollator, ZipperBatch)
    data_processor_adaptive.py  Adaptive chunk collator
scripts/                      Shell scripts for training phases and evaluation
  run_phase_warmup.sh         Phase-warmup (Javis + special tokens only)
  run_phase_cmp.sh            Phase-cmp (unfreeze compressor + Javis)
  run_phase_full.sh           Phase-full (unfreeze everything)
  eval_ppl.py                 Perplexity evaluation over a full JSONL dataset
  eval_test.sh                Wrapper for eval/eval_with_buffer.py
eval/                         Standalone evaluation scripts
examples/                     Inference demo
docs/                         Training guide
```

## Build / Install / Run Commands

### Install dependencies
```bash
pip install -r requirements.txt      # torch>=2.1, transformers>=4.40, wandb
pip install -e .                     # optional editable install of `iron-cell` package
```

### Training (distributed, 8xGPU default)
```bash
# Phase warmup — train Javis + special tokens only
bash scripts/run_phase_warmup.sh

# Phase cmp — unfreeze compressor + Javis
bash scripts/run_phase_cmp.sh

# Phase full — unfreeze all parameters
bash scripts/run_phase_full.sh
```

All training scripts use `torchrun --nproc_per_node 8 -m src.train` internally.
Single-GPU: set `PARALLEL=none` and `CUDA_VISIBLE_DEVICES=0`.

### Single-GPU training (quick test)
```bash
PARALLEL=none CUDA_VISIBLE_DEVICES=0 STEPS=10 BATCH_SIZE=1 \
  bash scripts/run_phase_warmup.sh
```

### Evaluation
```bash
# Full-dataset perplexity (fixed chunking)
python scripts/eval_ppl.py \
  --ckpt_dir <checkpoint_dir> \
  --data_path <eval.jsonl> \
  --chunk_size 16 --batch_size 2

# Full-dataset perplexity (adaptive chunking)
python scripts/eval_ppl.py \
  --ckpt_dir <checkpoint_dir> \
  --data_path <eval_adaptive.jsonl> \
  --chunking adaptive

# Eval via torchrun (FSDP)
bash scripts/run_eval.sh

# Buffer-based eval (reads first line of jsonl only)
bash scripts/eval_test.sh
```

### Inference demo
```bash
python examples/infer_demo.py
```

### Running tests
There is no formal test suite or pytest configuration. The closest to tests are:
- `src/data_processor/test_adaptive_processor.py` — run directly with `python -m src.data_processor.test_adaptive_processor`
- Jupyter notebooks in `src/data_processor/` for manual verification
- The inference demo serves as a basic smoke test: `python examples/infer_demo.py`

### Linting / Formatting
No linter or formatter is configured (no flake8, ruff, black, isort, mypy config).
Follow the existing conventions described below.

## Code Style Guidelines

### Python version and typing
- **Python >= 3.10** required (`pyproject.toml`).
- Use `from __future__ import annotations` at the top of every module for PEP 604
  union syntax (`X | None` instead of `Optional[X]`).
- Use modern type hints: `list[int]`, `dict[str, Any]`, `tuple[T, ...]`.
- Use `typing.Literal` for string enums in configs.
- Annotate return types on all public functions; use `-> None` for procedures.
- Use `# type: ignore[override]` sparingly when overriding HF methods.

### Imports
- **Order**: `__future__` imports first, then stdlib, then third-party (`torch`,
  `transformers`), then local (`from src.* import ...` or `from .module import ...`).
- No blank lines between import groups (the codebase does not enforce isort).
- Relative imports within the `src/` package (e.g., `from .configuration_iron_cell import ...`).
- Absolute `src.*` imports in scripts and entry points (e.g., `from src.modeling_iron_cell import ...`).
- Explicit imports only — no wildcard imports.

### Naming conventions
- **Classes**: `PascalCase` — `IronCellModel`, `IronCellConfig`, `ZipperBatch`, `Javis`.
- **Functions/methods**: `snake_case` — `compute_compressed_vectors`, `build_inputs_embeds`.
- **Private helpers**: prefix with underscore — `_init_eye_plus_noise_`, `_to_4d_additive_mask`.
- **Constants/globals**: `UPPER_SNAKE_CASE` — `DEEP_KV_CONTEXT`, `ORIGINAL_LLAMA_ATTENTION_FORWARD`.
- **Config fields**: `snake_case` — `javis_num_heads`, `freeze_compressor`.
- **Dataclass fields matching tensor semantics**: document shape in comments — `# [B, C, H]`.

### Formatting
- 4-space indentation (no tabs).
- No strict line length enforced, but aim for ~100-120 chars.
- Use trailing commas in multi-line argument lists.
- Docstrings: triple-quoted, short summary on first line, `Args:` / `Returns:` sections
  for public API. Internal helpers may use inline `#` comments instead.
- Comments in Chinese are common in training loops and experimental code; this is acceptable.

### Dataclasses and configuration
- Use `@dataclass` (often `frozen=True`) for immutable data containers and CLI args.
- HuggingFace `PretrainedConfig` subclass for model config (`IronCellConfig`).
- HuggingFace `PreTrainedModel` subclass for models (`IronCellModel`).
- `TrainArgs` uses `@dataclass(frozen=True)` with `HfArgumentParser` for CLI parsing.

### Tensor conventions
- **dtype**: BF16 (`torch.bfloat16`) for all model operations. Float32 for loss accumulators.
- **Shapes**: always document expected tensor shapes in comments — `# [B, L, H]`.
- Explicit `int()` / `float()` / `bool()` casts around config values to avoid type ambiguity.
- Use `torch.no_grad()` context for frozen module forward passes.
- Use `torch.autocast(device_type=..., dtype=torch.bfloat16)` for mixed precision.

### Error handling
- Validate config constraints with explicit `raise ValueError(...)` messages.
- Use `assert` only for internal shape invariants, not user-facing validation.
- `FileNotFoundError` for missing data/checkpoint paths.
- Guard distributed operations with `if torch.distributed.is_initialized()`.

### Model architecture patterns
- **Freeze strategy**: controlled per-phase via `set_phase()` — toggle `requires_grad`.
- **Monkey-patching**: `LlamaAttention.forward` is replaced at runtime for deep KV
  injection via a global `DEEP_KV_CONTEXT` dict (see `hack_llama_ddp.py` / `hack_llama_fsdp.py`).
- **Initialization**: custom init functions (`_init_eye_plus_noise_`, `_init_projector_`)
  for linear layers; `nn.init.normal_` for new parameters.
- **Special tokens**: `<soc>`, `<eoc>`, `<v_none>` added via `add_iron_cell_special_tokens`;
  embeddings smart-initialized from existing vocab then made trainable.

### Training patterns
- **Distributed**: supports `none`, `ddp`, `fsdp` via `--parallel` flag.
- **Gradient accumulation**: explicit micro-step loop with `no_sync()` context.
- **Gradient clipping**: `clip_grad_norm_` (or `FSDP.clip_grad_norm_` for FSDP).
- **Logging**: WandB for metrics; `print()` for console output on rank 0 only.
- **Checkpointing**: `save_pretrained` + `optimizer.pt` + `trainer_state.json`.

### Data format
- Training/eval data: JSONL files with `{"text": "..."}` per line.
- Adaptive chunking data: JSONL with `{"text": "...", "chunk_lens": [int, ...]}`.
- The `IronCellCollator` handles tokenization, chunking, zipper layout construction,
  staircase mask generation, and label masking (prefix positions set to -100).

### Key environment variables
- `CUDA_VISIBLE_DEVICES` — GPU selection (default `0,1,2,3,4,5,6,7`).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — used in training scripts.
- `NCCL_ASYNC_ERROR_HANDLING=1` — set in multi-phase scripts.
- Standard `RANK`, `LOCAL_RANK`, `WORLD_SIZE` for distributed training via `torchrun`.
