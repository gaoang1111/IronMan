# AGENTS.md — IronCell (Project Iron-Cell / SoulBone)

## Project Overview

IronCell is a PyTorch research project for autoregressive compressed memory using
homologous model differentiation. It pairs two clones of the same LLM (Llama 3.1 8B):
a **Compressor** and a **Generator**, bridged by a trainable cross-attention module
called **Javis**. The system compresses long-context KV caches by 8:1 via a
"zipper layout" embedding sequence with staircase attention masks.

## Research Background & Core Concepts

> **IMPORTANT**: Read this section to understand WHY the project exists and HOW decisions were made.
> For full details, see `docs/RESEARCH_OVERVIEW.md`.

### Core Thesis: Homologous Model Differentiation (同源模型分化)

The project's fundamental insight is that current AI models are "pre-built state machines" —
their capabilities are bounded by fixed parameters. Inspired by biological stem cell
differentiation, IronCell explores whether a pre-trained base model can be cloned and
"differentiated" into specialized functional modules (compressor, generator) that cooperate
without "rejection" because they share the same pre-training origin.

**Compression memory is the first validation scenario** for this paradigm — the ultimate
goal is a general framework for model capability expansion through homologous differentiation.

### Key Concepts

| Concept | What It Means |
|---------|--------------|
| **Homologous clones** | Compressor and Generator start from the same Llama 3.1 8B checkpoint. Same tokenizer, embedding space, and layer-wise semantic structure. Javis only needs to learn "how to compress", not "how to align two different models". |
| **Javis** | Cross-attention module with learnable queries, delta-q (content-adaptive bias), and deep KV projection. Compresses chunk hidden states into memory vectors. |
| **Deep KV Injection** | Compressed KV pairs are residually injected into 3 Generator layers (~15, ~23, 31) via monkey-patched attention. Later layers get higher injection coefficients (self-learned during training). |
| **Zipper Layout** | Sequence format: `[<soc>] V₁ [<eoc>] V₂ [<eoc>] ... raw_tokens`. Compressed vectors interleaved with raw tokens. |
| **Staircase Mask** | Chunk N's raw tokens can only see compressed info from chunks 1..N + own preceding raw tokens. Forces Generator to rely on compressed memory. |
| **Tail Bias** | Autoregressive models naturally concentrate attention on later tokens in each chunk. IronCell shifted from fighting this to exploiting it (→ Adaptive Chunking). |
| **Delta-Q** | `q = q_base + delta_q(chunk_mean)`: makes queries content-adaptive. With delta-q, attention shifts toward information-dense positions; without it, attention is uniform. |

### Version Evolution (Mark-1 → Mark-42 → Adaptive)

```
Mark-1 (Linear) → Mark-33 (CrossAttn) → Mark-42 (Deep KV) → Mark-42+ (Delta-Q) → Adaptive Seg
```

Each iteration solved a specific problem discovered in the previous version:
- Mark-1: loss stalled → needed better extraction
- Mark-33: gradient too weak → needed deep injection  
- Mark-42: loss plateau at 2.3, needle issues → needed content-adaptive queries
- Mark-42+: fixed chunking wastes tail bias → adaptive chunking

**See `docs/EVOLUTION.md` for the complete iteration history with decisions and findings.**

### Current Status (Quick Reference)

| Item | Value |
|------|-------|
| Branch | `ga/adaptive-seg` |
| Latest checkpoint | `phase-full_step_480` |
| Fixed eval loss | ~2.3 |
| Adaptive eval loss | ~2.6 (distribution shift, needs fine-tuning) |
| Compression ratio | 8:1 (16 tokens → 2 vectors) |
| Key blocker | Loss plateau; compression effectiveness unconfirmed by ablation |

### Active Research Challenges

1. **Loss plateau at ~2.3**: Information bottleneck? Need more data/steps? Or architectural limit?
2. **Compression effectiveness**: Streaming generation works, but is it the compressed info or model's own ability?
3. **Deep KV "shock"**: Needle test shows no-inject performs better; direct residual addition may disrupt attention patterns
4. **Next proposed solution**: Sparse attention decoupling — separate compressed KV attention from original attention, add to hidden state instead of KV

## Repository Layout

```
src/                          Core Python package (imported as `src.*`)
  models/                     Model definitions
    config.py                 IronCellConfig (HF PretrainedConfig subclass)
    javis.py                  Javis cross-attention module
    iron_cell.py              IronCellModel (compressor + generator)
  attention/                  Attention patching for deep KV injection
    kv_context.py             Global DEEP_KV_CONTEXT management
    patched_attention.py      Unified attention forward with residual injection
    train_step.py             TrainStepModule for DDP/FSDP training
  evaluation/                 Evaluation utilities
    evaluator.py              compute_ppl, run_eval_loop
  utils/                      Modular utilities (new)
    checkpoint.py             save/load checkpoint
    data.py                   JsonlDataset
    distributed.py            FSDP auto wrap policy
    model_loader.py           load_tokenizer, load_model
    javis_init.py             warmup_init_javis_query
  train.py                    Distributed training entry point
  eval.py                     Eval entry point (loss on eval set)
  train_utils.py              (shim) Re-exports from src.utils for backward compat
  token_utils.py              Special token handling (<soc>, <eoc>, <v_none>)
  data_processor/             Zipper layout builder (inheritance-based)
    batch.py                  ZipperBatch dataclass
    builder.py                ZipperBuilderBase + ZipperBuilder + AdaptiveZipperBuilder
    collator.py               IronCellCollatorBase + IronCellCollator + AdaptiveIronCellCollator
scripts/                      Organized shell scripts
  training/                   Training scripts
    run_phase_warmup.sh       Phase-warmup (Javis + special tokens only)
    run_phase_cmp.sh          Phase-cmp (unfreeze compressor + Javis)
    run_phase_full.sh         Phase-full (unfreeze everything)
  eval/                       Evaluation scripts
    eval_ppl.py               PPL evaluation (--chunking fixed|adaptive)
    run_ppl.sh                Wrapper script
    needle/                   Needle-in-haystack experiments
  data/                       Data processing scripts
    prepare_adaptive_chunks.py  Prepare adaptive chunking data
  deprecated/                 Old/unused scripts
eval/                         Standalone evaluation scripts
examples/                     Inference demo
docs/                         Documentation suite
  RESEARCH_OVERVIEW.md        Research motivation, core method, related work
  ARCHITECTURE.md             Technical architecture details
  EVOLUTION.md                Version history (Mark-1 → Mark-42 → Adaptive)
  EXPERIMENT_LOG.md           All experiments with hypotheses/results/conclusions
  KNOWN_ISSUES.md             Known problems, technical decisions, trade-offs
  ROADMAP.md                  Future directions, hypotheses to verify
  RESEARCH_STATUS.md          Current state snapshot (checkpoint, metrics, blockers)
  training_guide.md           Phase-1 training guide
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
  injection via a global `DEEP_KV_CONTEXT` dict (see `src/attention/` module).
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

## Research Context

**IMPORTANT**: See `docs/RESEARCH_STATUS.md` for:
- Current research status and experimental findings
- Known issues and technical decisions
- Next steps and experiment log

This document should be consulted at the start of each conversation to understand the current state of the project.

### Documentation Suite

The `docs/` directory contains a comprehensive documentation system. Read these documents
for full context on the project:

| Document | When to Read | Content |
|----------|-------------|---------|
| `docs/RESEARCH_OVERVIEW.md` | First time / new conversation | Research motivation, core idea (homologous model differentiation), method overview, related work comparison |
| `docs/ARCHITECTURE.md` | When working on code | Full technical architecture: model components, data flow, Deep KV injection, zipper layout, staircase mask |
| `docs/EVOLUTION.md` | To understand design decisions | Complete version history: Mark-1 → Mark-33 → Mark-42 → Mark-42+ → Adaptive. Why each change was made. |
| `docs/EXPERIMENT_LOG.md` | When planning experiments | All experiments with hypotheses, configs, results, conclusions. Includes failed experiments. |
| `docs/KNOWN_ISSUES.md` | Before making design changes | Known bugs (fixed and active), key technical decisions and their rationale, potential risks |
| `docs/ROADMAP.md` | When deciding what to do next | Future directions, unverified hypotheses, priority rankings |
| `docs/RESEARCH_STATUS.md` | Every conversation start | Current checkpoint, metrics, blockers, next actions |
