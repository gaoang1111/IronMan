# IronCell Technical Architecture

> This document describes the full technical architecture of IronCell, including model
> components, data flow, attention mechanisms, and training pipeline.

---

## 1. System Overview

```
Input Text
    |
    v
[Tokenizer + Chunking]  ──── Fixed (16 tokens) or Adaptive (delimiter-based)
    |
    v
[Compressor (Llama 3.1 8B)]  ──── frozen or trainable depending on phase
    |
    v  hidden states [B*C, L, H]
    |
[Javis Cross-Attention Module]
    |
    ├──── Group 0 output ──→ [build_inputs_embeds] ──→ embedding injection
    |                                                       |
    └──── All groups ──→ [get_all_layer_kv] ──→ Deep KV     |
                              |                              |
                              v                              v
                     [DEEP_KV_CONTEXT]              [Zipper Layout Sequence]
                              |                              |
                              v                              v
                     [Patched Attention x32] ←──── [Generator (Llama 3.1 8B)]
                              |
                              v
                     LM Loss + L2 + Ortho Penalty
```

**Three core components:**

| Component | Role | Source | Trainability |
|-----------|------|--------|-------------|
| Compressor | Reads text chunks, produces hidden states | Llama 3.1 8B clone | Frozen (phase-warmup/cmp) or Trainable (phase-full) |
| Javis | Cross-attention compression + deep KV projection | New module | Always trainable |
| Generator | Consumes zipper layout, produces next-token predictions | Llama 3.1 8B clone | Frozen (phase-warmup) or Trainable (phase-cmp/full) |

---

## 2. Compressor

**File**: `src/models/iron_cell.py` — `IronCellModel.compute_compressed_vectors()`

The compressor is a frozen (or trainable) Llama 3.1 8B clone. Its job is to read each text chunk and produce rich hidden states.

**Input/Output:**
- Input: `chunk_input_ids [B, C, L]` — B batches, C chunks, L tokens per chunk
- The chunks are flattened to `[B*C, L]` for efficient batched forward pass
- Output: `hidden_states [B*C, L, H]` — H=4096 for Llama 8B
- Hidden states are reshaped back to `[B, C, L, H]` and fed to Javis

**Freeze behavior:**
- When frozen: wrapped in `torch.no_grad()`, all params `requires_grad=False`
- When trainable: gradients flow back through compressor via Javis

---

## 3. Javis Cross-Attention Module

**File**: `src/models/javis.py` — `Javis(nn.Module)`

Javis is the central innovation — a cross-attention module that compresses chunk hidden states into memory vectors and projects them into per-layer KV pairs for deep injection.

### 3.1 Architecture

```
Compressor Hidden States [B, C, L, H]
         |
         v
    [in_proj]  ──── Linear (if H_cmp != H_gen) or Identity
         |
         v
    [ln_in]  ──── LayerNorm (optional)
         |
         v
    [Multi-Head Cross-Attention]
         |    Q = q_base + delta_q(chunk_mean)    ← content-adaptive
         |    K = wk(hidden)
         |    V = wv(hidden)
         |
         v
    [wo + group_ln_out]  ──── output projection + per-group LayerNorm
         |
         v
    out + 0.5 * shortcut  ──── shortcut = global mean of hidden states
         |
         v
    Javis Output [B, G, Q, H]
         |
         ├──── Group 0: [B, C, Q, H] → embedding injection
         └──── All groups → [kv_proj_weights] → 32 layers of (K, V) pairs
```

### 3.2 Key Design Elements

**Query Groups** (`q_base [G, Q, H]`):
- 32 transformer layers are divided into G groups (G = num_layers / query_group_size)
- Each group shares the same set of Q learnable queries
- Different groups learn to extract different aspects of information for different layer depths

**Delta-Q Mechanism** (`q_proj`):
- Static query: `q_base` — learned during training, same for all inputs
- Dynamic bias: `delta_q = q_proj(mean(chunk_hidden_states))`
- Final query: `q = q_base + delta_q`
- Purpose: makes queries content-adaptive rather than purely static
- Experimental finding: with delta-q, attention weights shift toward later tokens (utilizing tail bias); without it (delta-q=0), attention becomes uniform

**Shortcut Connection**:
- `final_out = cross_attn_out + 0.5 * global_mean(hidden_states)`
- Provides an information highway to prevent degradation
- The global mean acts as a coarse summary that is always available

**Orthogonality Constraint**:
- `penalty = relu(|cos_sim(q_i, q_j)| - 0.1)` for all query pairs
- Margin of 0.1: allows mild similarity, penalizes strong overlap
- Prevents queries from collapsing to extract the same information

### 3.3 Initialization

- **KV projections** (`wk`, `wv`, `wo`): Identity + Gaussian noise (`std=javis_init_noise_std`)
- **Layer gates**: Initialized to 0.07 (small, gentle injection at start)
- **kv_proj_weights**: `nn.init.normal_(std=0.01)`
- Rationale: near-identity initialization ensures stable training start; the model gradually learns to deviate

---

## 4. Deep KV Injection

**Files**: `src/attention/patched_attention.py`, `src/attention/kv_context.py`

### 4.1 Motivation

Early versions (Mark-1, Mark-33) only injected compressed information at the embedding layer. Problems:
- Gradient backpropagation was weak — the model had little incentive to improve the compressor
- Information injected at the bottom layers got "buried" as it passed through 32 transformer layers
- Inspired by cross-model ResNet: direct residual connections across model depths

### 4.2 Mechanism

Javis output `[B, G, Q, H]` is projected via `kv_proj_weights [L, H, kv_dim_per_layer]` into per-layer KV pairs:

```python
# For each of the selected injection layers:
k_javis, v_javis = javis.get_all_layer_kv(v_out_blocks)
# Returns: list of (K, V) tuples, each [B, num_kv_heads, Q, head_dim]
```

At each patched attention layer, for each batch element and chunk:
```python
key_states[b, :, memory_pos:memory_pos+Q, :] += k_javis[b, c]   # residual addition
value_states[b, :, memory_pos:memory_pos+Q, :] += v_javis[b, c]  # residual addition
```

### 4.3 Key Design Decisions

**Residual addition (not replacement)**:
- `orig_k + k_javis` preserves the original token semantics
- Direct overwrite caused gradient vanishing in experiments

**3-layer injection (not all 32)**:
- Initial experiment with all 32 layers: training caused injection coefficients to decrease toward zero
- Settled on 3 representative layers: ~layer 15, ~layer 23, layer 31
- Observation: later layer coefficients > middle > front
- Interpretation: aligns with Llama's architecture where later layers handle more global/fused information and benefit more from clear compressed information injection

**Layer Gates**:
- Per-layer learnable scalars controlling injection strength
- Initialized to 0.07
- During training, later layer gates tend to **increase** (e.g., from 0.07 upward)
- This suggests the model actively "wants" unfiltered compressed information at higher layers

### 4.4 Global Context (Monkey-Patch Pattern)

```
TrainStepModule.forward()
    |
    ├── set_kv_context(layer_kvs, memory_positions, num_queries)
    |       └── writes to global DEEP_KV_CONTEXT dict
    |
    ├── generator.forward()
    |       └── each LlamaAttention.forward (patched) reads DEEP_KV_CONTEXT
    |
    └── clear_kv_context()  (in finally block)
```

- `DEEP_KV_CONTEXT` is a module-level global dict in `src/attention/kv_context.py`
- `LlamaAttention.forward` is monkey-patched at `TrainStepModule.__init__` time
- This avoids modifying the HuggingFace source code
- Safe because PyTorch training is single-threaded per process in DDP/FSDP

---

## 5. Zipper Layout & Staircase Mask

**Files**: `src/data_processor/builder.py`, `src/data_processor/collator.py`

### 5.1 Zipper Layout

The generator input sequence interleaves compressed vectors with raw tokens:

```
[<soc>] [V₁] [<eoc>] [V₂] [<eoc>] ... [Vₙ] [<eoc>] [raw_tokens_chunk_1] [raw_tokens_chunk_2] ...
```

Where:
- `<soc>` (start-of-chunk): marks beginning of compressed memory section
- `Vᵢ`: compressed vector slots (replaced by Javis output in `build_inputs_embeds`)
- `<eoc>` (end-of-chunk): delimiter between compressed chunks
- `raw_tokens`: original text tokens, grouped by chunk

### 5.2 Staircase Mask

The attention mask enforces a critical constraint: **each chunk can only see its own preceding raw tokens and all previous chunks' compressed information**.

```
            V₁  eoc  V₂  eoc  raw₁_t1  raw₁_t2  raw₂_t1  raw₂_t2
V₁          ✓
eoc₁        ✓   ✓
V₂          ✓   ✓    ✓
eoc₂        ✓   ✓    ✓   ✓
raw₁_t1     ✓   ✓                ✓
raw₁_t2     ✓   ✓                ✓       ✓
raw₂_t1     ✓   ✓    ✓   ✓                        ✓
raw₂_t2     ✓   ✓    ✓   ✓                        ✓       ✓
```

**Key properties:**
- `raw₂` tokens can see V₁, eoc₁, V₂, eoc₂ (all preceding compressed info)
- `raw₂` tokens **cannot** see raw₁ tokens (forces reliance on compressed memory)
- `raw₂` tokens can see preceding tokens within their own chunk
- This prevents the model from "cheating" by reading raw text of previous chunks

**Design purpose:**
- Forces the Generator to depend on compressed memory, not raw previous context
- Pushes the Compressor + Javis to learn meaningful compression
- Without this constraint, the Generator could ignore compressed vectors entirely

### 5.3 `<eoc>` as Generation Anchor

In the generation scenario:
- The first token of each chunk is predicted solely from preceding compressed information + `<eoc>`
- `<eoc>` is used as the prediction target instead of the next raw token
- This prevents compressed information from collapsing toward simple next-token prediction
- The fact that coherent generation is possible from `<eoc>` + compressed vectors provides evidence that compression is working

---

## 6. Special Tokens

**File**: `src/token_utils.py`

| Token | Semantic Init | Purpose |
|-------|--------------|---------|
| `<soc>` | Mean embedding of "Summary" | Signals start of compressed memory section |
| `<eoc>` | Mean embedding of ":" | Delimiter between chunks; generation anchor |
| `<v_none>` | Mean embedding of "none" | Placeholder for empty/padding memory slots |

**Initialization strategy:**
- Smart init from semantically meaningful tokens (not random)
- Small Gaussian noise (std=1e-3) added for symmetry breaking
- Maintained as separate `nn.Embedding` during training (not modifying the main embedding table)
- Injected via mask-and-fill in `build_inputs_embeds`

---

## 7. Training Pipeline

**File**: `src/attention/train_step.py` — `TrainStepModule`

### 7.1 Forward Pass Orchestration

```python
def forward(batch, return_metrics=False):
    # 1. Compress: chunk_input_ids → compressed vectors + deep KV
    memory_vectors, deep_kv, metrics = iron.compute_compressed_vectors(...)
    
    # 2. Set global context for monkey-patched attention
    set_kv_context(deep_kv, memory_positions, num_queries)
    
    try:
        # 3. Build generator inputs with memory injection
        inputs_embeds = iron.build_inputs_embeds(zipper_ids, memory_vectors, positions)
        
        # 4. Generator forward (deep KV injection happens inside patched attention)
        output = iron.forward(inputs_embeds, attention_mask, position_ids, labels)
        gen_loss = output.loss
    finally:
        # 5. Always clear context (GPU memory safety)
        clear_kv_context()
    
    # 6. Compute auxiliary losses
    total_loss = gen_loss + l2_coeff * l2_loss + q_cos_coeff * ortho_penalty
    return total_loss, metrics
```

### 7.2 Three-Phase Training

| Phase | Unfrozen Components | Frozen Components | Purpose |
|-------|-------------------|------------------|---------|
| **warmup** | Javis + special token embeddings | Compressor + Generator backbone | Train Javis to compress; lowest risk |
| **cmp** | Compressor + Javis | Generator backbone | Allow Compressor to adapt for compression |
| **full** | Everything | Nothing | End-to-end fine-tuning |

**Training scripts**: `scripts/training/run_phase_{warmup,cmp,full}.sh`

### 7.3 Loss Components

| Loss | Formula | Active Phase | Purpose |
|------|---------|-------------|---------|
| **Generation loss** | Standard causal LM cross-entropy (labels=-100 for prefix) | All | Main objective |
| **L2 regularization** | `mean(memory_vectors.norm(dim=-1))` | phase-cmp, phase-full | Prevent memory vector norm explosion |
| **Orthogonality penalty** | `relu(\|cos_sim(q_i, q_j)\| - 0.1)` | All | Prevent query collapse |

### 7.4 Distributed Training

- Supports: `none` (single GPU), `ddp`, `fsdp`
- Gradient accumulation with `no_sync()` context for micro-steps
- Gradient clipping via `clip_grad_norm_`
- Default: 8x GPU via `torchrun --nproc_per_node 8`

---

## 8. Data Processing

**Files**: `src/data_processor/`

### 8.1 Fixed Chunking (`ZipperBuilder`)

- Tokenizes text, splits into equal chunks of `chunk_size` tokens (default: 16)
- Pads last chunk if needed
- Constructs zipper layout + staircase mask

### 8.2 Adaptive Chunking (`AdaptiveZipperBuilder`)

- Splits by strong delimiters (`.` `?` `!` newline)
- Merges short segments if boundary surprisal is low
- Splits long segments by weak punctuation or surprisal valleys
- Data format: JSONL with `{"text": "...", "chunk_lens": [int, ...]}`
- Chunk lengths are pre-computed offline (`scripts/data/prepare_adaptive_chunks.py`)

### 8.3 Collator Pipeline

```
Raw JSONL → Tokenize → Chunk → Build Zipper Layout → Staircase Mask → Label Masking → ZipperBatch
```

`ZipperBatch` (dataclass) contains:
- `zipper_input_ids [B, S]`: full zipper sequence
- `labels [B, S]`: -100 for prefix positions, token ids for raw positions
- `attention_mask_2d [B, S, S]`: staircase mask
- `position_ids [B, S]`: position assignments
- `chunk_input_ids [B, C, L]`: chunks for compressor
- `chunk_attention_mask [B, C, L]`: attention mask for compressor
- `memory_positions [B, C]`: where compressed vectors sit in the zipper sequence
- `prefix_lens [B]`: length of the prefix (compressed) section
- `valid_lens [B]`: total valid sequence length

---

## 9. Evaluation

**Files**: `src/evaluation/evaluator.py`, `scripts/eval/eval_ppl.py`

### 9.1 Perplexity Evaluation

- Uses same `TrainStepModule` for forward pass (ensures consistency with training)
- Computes average loss across all samples, then `PPL = exp(loss)`
- Supports both fixed and adaptive chunking via `--chunking` flag

### 9.2 Needle-in-Haystack

- Tests recall ability at different positions within compressed context
- Scripts in `scripts/eval/needle/`
- Key finding: recall is significantly better when the needle falls near chunk boundaries (tail bias)

### 9.3 Streaming Generation

- Uses staircase mask pattern for autoregressive generation
- Each new chunk's first token is predicted from compressed history + `<eoc>`
- Validates that compressed information is meaningful enough for coherent generation

---

## 10. Key File Reference

| Component | Primary File | Entry Point |
|-----------|-------------|-------------|
| Model config | `src/models/config.py` | `IronCellConfig` |
| Javis module | `src/models/javis.py` | `Javis` |
| Main model | `src/models/iron_cell.py` | `IronCellModel` |
| Deep KV context | `src/attention/kv_context.py` | `set_kv_context()` / `clear_kv_context()` |
| Patched attention | `src/attention/patched_attention.py` | `smart_hybrid_attention_forward()` |
| Training step | `src/attention/train_step.py` | `TrainStepModule` |
| Special tokens | `src/token_utils.py` | `add_iron_cell_special_tokens()` |
| Data builder | `src/data_processor/builder.py` | `ZipperBuilder` / `AdaptiveZipperBuilder` |
| Data collator | `src/data_processor/collator.py` | `IronCellCollator` / `AdaptiveIronCellCollator` |
| Training entry | `src/train.py` | `main()` |
| Eval entry | `scripts/eval/eval_ppl.py` | `main()` |
