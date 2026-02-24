from __future__ import annotations

from collections.abc import Iterator, Sequence
import types
import torch
import torch.nn.functional as F


def _infer_token_id(tokenizer, candidates: Sequence[str]) -> int:
    unk = getattr(tokenizer, "unk_token_id", None)
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is None:
            continue
        tid_i = int(tid)
        if unk is not None and tid_i == int(unk):
            continue
        if tid_i >= 0:
            return tid_i
    raise ValueError(f"Special token not found in tokenizer vocab: {list(candidates)}")


def _embed_with_special_overrides(iron_model, input_ids: torch.LongTensor) -> torch.Tensor:
    embed = iron_model.generator.get_input_embeddings()
    inputs_embeds = embed(input_ids)
    special_ids = getattr(iron_model, "special_token_ids", None)
    special_embeds = getattr(iron_model, "special_token_embeddings", None)
    if special_ids is None or special_embeds is None:
        return inputs_embeds
    if not isinstance(special_ids, torch.Tensor) or int(special_ids.numel()) == 0:
        return inputs_embeds

    special_ids = special_ids.to(device=inputs_embeds.device)
    for i in range(int(special_ids.numel())):
        tid = int(special_ids[i].item())
        mask = input_ids == tid
        if mask.any():
            inputs_embeds[mask] = special_embeds.weight[i].to(inputs_embeds.dtype)
    return inputs_embeds


def _mark42_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    javis_all_layer_kvs = kwargs.pop("javis_all_layer_kvs", None)
    javis_meta = kwargs.pop("javis_meta", None)

    input_shape = hidden_states.shape[:-1]
    bsz, q_len = int(input_shape[0]), int(input_shape[1])
    head_dim = int(self.head_dim)
    hidden_shape = (*input_shape, -1, head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    if javis_all_layer_kvs is not None and javis_meta is not None:
        target_layers = getattr(self, "_mark42_target_layers", None)
        if target_layers is None or int(getattr(self, "layer_idx", -1)) in target_layers:
            mem_pos_abs, num_q = javis_meta
            layer_idx = int(getattr(self, "layer_idx", -1))
            layer_kv = None
            if isinstance(javis_all_layer_kvs, (list, tuple)) and 0 <= layer_idx < len(javis_all_layer_kvs):
                layer_kv = javis_all_layer_kvs[layer_idx]
            if layer_kv is not None:
                k_javis, v_javis = layer_kv
                if isinstance(mem_pos_abs, torch.Tensor) and mem_pos_abs.dim() == 1:
                    mem_pos_abs = mem_pos_abs.unsqueeze(0)
                if isinstance(k_javis, torch.Tensor) and k_javis.dim() == 4:
                    k_javis = k_javis.unsqueeze(1)
                    v_javis = v_javis.unsqueeze(1)

                past_len = int(past_key_values.get_seq_length()) if past_key_values is not None else 0
                mem_pos_abs = mem_pos_abs.to(device=hidden_states.device)
                local_pos = mem_pos_abs - past_len

                key_states = key_states.clone()
                value_states = value_states.clone()
                for b in range(bsz):
                    for c in range(int(local_pos.size(1))):
                        start = int(local_pos[b, c].item())
                        if 0 <= start and start + int(num_q) <= q_len:
                            key_states[b, :, start : start + int(num_q), :] = (
                                key_states[b, :, start : start + int(num_q), :]
                                + k_javis[b, c].to(dtype=key_states.dtype, device=key_states.device)
                            )
                            value_states[b, :, start : start + int(num_q), :] = (
                                value_states[b, :, start : start + int(num_q), :]
                                + v_javis[b, c].to(dtype=value_states.dtype, device=value_states.device)
                            )

    from transformers.models.llama.modeling_llama import ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb, eager_attention_forward

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    attention_interface = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


class Mark42StreamingEngine:
    def __init__(
        self,
        iron_model,
        tokenizer,
        *,
        chunk_size: int = 16,
        buffer_num: int = 2,
        q_num: int | None = None,
        target_layers: Sequence[int] = (15, 23, 31),
    ) -> None:
        self.model = iron_model
        self.tokenizer = tokenizer
        self.chunk_size = int(chunk_size)
        self.buffer_num = int(buffer_num)
        self.target_layers = tuple(int(x) for x in target_layers)

        inferred_q = getattr(getattr(self.model, "javis", None), "num_queries", None)
        self.q_num = int(inferred_q) if q_num is None and inferred_q is not None else int(q_num or 2)

        self.device = self.model.device

        bos_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_id is None:
            raise ValueError("tokenizer.bos_token_id is required for inference.")
        self.bos_id = int(bos_id)
        self.soc_id = _infer_token_id(self.tokenizer, ("<soc>", "<SOC>"))
        self.eoc_id = _infer_token_id(self.tokenizer, ("<eoc>", "<EOC>"))
        self.v_none_id = _infer_token_id(self.tokenizer, ("<v_none>", "<V-1>", "<v-1>", "<V_NONE>"))

        self._patch_generator_attention()

    def _patch_generator_attention(self) -> None:
        gen = getattr(self.model, "generator", None)
        layers = getattr(getattr(gen, "model", None), "layers", None)
        if layers is None:
            raise ValueError("iron_model.generator.model.layers not found; cannot patch attention for inference.")
        for layer in layers:
            attn = layer.self_attn
            if getattr(attn, "_mark42_patched", False):
                attn._mark42_target_layers = self.target_layers
                continue
            attn._mark42_target_layers = self.target_layers
            attn._mark42_original_forward = attn.forward
            attn.forward = types.MethodType(_mark42_llama_attention_forward, attn)
            attn._mark42_patched = True

    def stream_generate(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        eos_token_id: int | None = None,
        
    ) -> Iterator[int]:
        if eos_token_id is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
            eos_token_id = int(eos) if eos is not None else None

        self.model.eval()
        self.model.generator.eval()

        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if self.buffer_num <= 0:
            raise ValueError(f"buffer_num must be > 0, got {self.buffer_num}")
        if self.q_num <= 0:
            raise ValueError(f"q_num must be > 0, got {self.q_num}")

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        raw_queue: list[int] = [int(x) for x in prompt_ids]
        full_context: list[int] = [int(x) for x in prompt_ids]

        embed_dtype = self.model.generator.get_input_embeddings().weight.dtype

        prefix_ids = torch.tensor(
            [[self.bos_id, self.soc_id] + [self.v_none_id] * self.q_num],
            device=self.device,
            dtype=torch.long,
        )
        prefix_embeds = _embed_with_special_overrides(self.model, prefix_ids).to(dtype=embed_dtype)
        with torch.inference_mode():
            out_prefix = self.model.generator(inputs_embeds=prefix_embeds, use_cache=True)
        mem_cache = out_prefix.past_key_values

        raw_queue, mem_cache = self._compress_batch_if_needed(raw_queue, mem_cache, embed_dtype)
        decode_cache, logits_last = self._prefill_suffix(raw_queue, mem_cache, embed_dtype)

        max_new_tokens = int(max_new_tokens)
        for _ in range(max_new_tokens):
            next_token = self._sample_next_token(logits_last, full_context=full_context, temperature=temperature, repetition_penalty=repetition_penalty)
            raw_queue.append(next_token)
            full_context.append(next_token)
            yield next_token

            if eos_token_id is not None and int(next_token) == int(eos_token_id):
                break

            raw_queue, mem_cache, rebuilt = self._compress_batch_if_needed(raw_queue, mem_cache, embed_dtype, return_rebuilt=True)
            if rebuilt:
                decode_cache, logits_last = self._prefill_suffix(raw_queue, mem_cache, embed_dtype)
                continue

            token_ids = torch.tensor([[int(next_token)]], device=self.device, dtype=torch.long)
            token_embeds = _embed_with_special_overrides(self.model, token_ids).to(dtype=embed_dtype)
            with torch.inference_mode():
                out = self.model.generator(inputs_embeds=token_embeds, past_key_values=decode_cache, use_cache=True)
            decode_cache = out.past_key_values
            logits_last = out.logits[:, -1, :]

    def _calc_cknum(self, raw_len: int) -> int:
        full_chunks = int(raw_len) // int(self.chunk_size)
        return max(0, full_chunks - int(self.buffer_num) + 1)

    def _compress_batch_if_needed(
        self,
        raw_queue: list[int],
        mem_cache,
        embed_dtype: torch.dtype,
        *,
        return_rebuilt: bool = False,
    ):
        cknum = self._calc_cknum(len(raw_queue))
        if cknum <= 0:
            return (raw_queue, mem_cache, False) if return_rebuilt else (raw_queue, mem_cache)

        take = int(cknum) * int(self.chunk_size)
        ids = torch.tensor(raw_queue[:take], device=self.device, dtype=torch.long).view(1, cknum, self.chunk_size)
        mask = torch.ones((1, cknum, self.chunk_size), device=self.device, dtype=torch.long)

        with torch.inference_mode():
            _, v_vecs, deep_layer_kvs, _ = self.model.compute_compressed_vectors(
                chunk_input_ids=ids,
                chunk_attention_mask=mask,
                return_metrics=False,
            )

        v_flat = v_vecs.reshape(1, cknum * self.q_num, -1).to(dtype=embed_dtype, device=self.device)
        past_len = int(mem_cache.get_seq_length()) if mem_cache is not None else 0
        mem_pos_abs = (
            past_len
            + torch.arange(0, cknum * self.q_num, step=self.q_num, device=self.device, dtype=torch.long).unsqueeze(0)
        )

        with torch.inference_mode():
            out = self.model.generator(
                inputs_embeds=v_flat,
                past_key_values=mem_cache,
                use_cache=True,
                javis_all_layer_kvs=deep_layer_kvs,
                javis_meta=(mem_pos_abs, self.q_num),
            )
        mem_cache = out.past_key_values
        raw_queue = raw_queue[take:]
        return (raw_queue, mem_cache, True) if return_rebuilt else (raw_queue, mem_cache)

    def _prefill_suffix(self, raw_queue: list[int], mem_cache, embed_dtype: torch.dtype):
        suffix_ids = torch.tensor([[self.eoc_id] + [int(x) for x in raw_queue]], device=self.device, dtype=torch.long)
        suffix_embeds = _embed_with_special_overrides(self.model, suffix_ids).to(dtype=embed_dtype)
        with torch.inference_mode():
            out = self.model.generator(inputs_embeds=suffix_embeds, past_key_values=mem_cache, use_cache=True)
        return out.past_key_values, out.logits[:, -1, :]

    # def _sample_next_token(self, logits: torch.Tensor, *, temperature: float, repetition_penalty: float) -> int:
    #     if float(temperature) > 0.0:
    #         probs = F.softmax(logits / float(temperature), dim=-1)
    #         return int(torch.multinomial(probs, num_samples=1).item())
    #     return int(torch.argmax(logits, dim=-1).item())

    def _sample_next_token(
        self, 
        logits: torch.Tensor, 
        full_context: list[int], 
        *, 
        temperature: float = 0.0, 
        repetition_penalty: float = 1.0
    ) -> int:
        processed_logits = logits.clone()

        if repetition_penalty > 1.0:
            history_tokens = set(full_context)
            for token_id in history_tokens:
                score = processed_logits[0, token_id]
                # HuggingFace 标准的惩罚逻辑：负数乘以惩罚系数使概率更低，正数除以惩罚系数
                if score < 0:
                    processed_logits[0, token_id] = score * repetition_penalty
                else:
                    processed_logits[0, token_id] = score / repetition_penalty

        if float(temperature) > 0.0:
            probs = F.softmax(processed_logits / float(temperature), dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())
        return int(torch.argmax(processed_logits, dim=-1).item())

    def generate(
        self,
        prompt_text: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        eos_token_id: int | None = None,
    ) -> str:
        token_ids: list[int] = []
        for tid in self.stream_generate(
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
        ):
            token_ids.append(int(tid))
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)
