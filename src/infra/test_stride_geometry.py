import unittest
import torch

from .infra import Mark42StreamingEngine


class _DummyCache:
    def __init__(self, seq_len: int = 0) -> None:
        self._seq_len = int(seq_len)

    def get_seq_length(self) -> int:
        return int(self._seq_len)

    def advance(self, n: int) -> None:
        self._seq_len += int(n)


class _DummyGenerator:
    def __call__(self, *, inputs_embeds, past_key_values=None, use_cache=True, **kwargs):
        cache = past_key_values if past_key_values is not None else _DummyCache(0)
        cache.advance(int(inputs_embeds.shape[1]))
        return type("Out", (), {"past_key_values": cache})


class _DummyModel:
    def __init__(self, q_num: int, hidden_size: int = 4) -> None:
        self.q_num = int(q_num)
        self.hidden_size = int(hidden_size)
        self.generator = _DummyGenerator()
        self.last_chunk_input_ids = None

    def compute_compressed_vectors(self, *, chunk_input_ids, chunk_attention_mask, return_metrics=False):
        self.last_chunk_input_ids = chunk_input_ids.detach().cpu().tolist()
        bsz, cknum, _ = chunk_input_ids.shape
        v_vecs = torch.zeros((bsz, cknum, self.q_num, self.hidden_size), dtype=torch.float32, device=chunk_input_ids.device)
        deep_layer_kvs = None
        return None, v_vecs, deep_layer_kvs, None


class TestStrideGeometry(unittest.TestCase):
    def _make_engine(self, *, chunk_size: int, chunk_stride: int, buffer_num: int, q_num: int):
        engine = Mark42StreamingEngine.__new__(Mark42StreamingEngine)
        engine.device = torch.device("cpu")
        engine.chunk_size = int(chunk_size)
        engine.chunk_stride = int(chunk_stride)
        engine.buffer_num = int(buffer_num)
        engine.q_num = int(q_num)
        engine.model = _DummyModel(q_num=q_num)
        return engine

    def test_cknum_slide_window(self):
        engine = self._make_engine(chunk_size=8, chunk_stride=4, buffer_num=2, q_num=2)
        self.assertEqual(engine._calc_cknum(7), 0)
        self.assertEqual(engine._calc_cknum(8), 0)
        self.assertEqual(engine._calc_cknum(11), 0)
        self.assertEqual(engine._calc_cknum(12), 1)
        self.assertEqual(engine._calc_cknum(16), 2)

    def test_compress_uses_overlapped_slices_and_advances_by_stride(self):
        engine = self._make_engine(chunk_size=8, chunk_stride=4, buffer_num=2, q_num=2)
        raw_queue = list(range(16))
        cache = _DummyCache(0)
        embed_dtype = torch.float16

        new_queue, new_cache, rebuilt = engine._compress_batch_if_needed(raw_queue, cache, embed_dtype, return_rebuilt=True)
        self.assertTrue(rebuilt)
        self.assertEqual(new_queue, list(range(8, 16)))
        self.assertEqual(new_cache.get_seq_length(), 4)
        self.assertEqual(
            engine.model.last_chunk_input_ids,
            [[list(range(0, 8)), list(range(4, 12))]],
        )

    def test_stride_equal_chunk_size_matches_old_advance(self):
        engine = self._make_engine(chunk_size=8, chunk_stride=8, buffer_num=2, q_num=2)
        raw_queue = list(range(24))
        cache = _DummyCache(0)
        embed_dtype = torch.float16

        new_queue, new_cache, rebuilt = engine._compress_batch_if_needed(raw_queue, cache, embed_dtype, return_rebuilt=True)
        self.assertTrue(rebuilt)
        self.assertEqual(new_queue, list(range(16, 24)))
        self.assertEqual(new_cache.get_seq_length(), 4)
        self.assertEqual(
            engine.model.last_chunk_input_ids,
            [[list(range(0, 8)), list(range(8, 16))]],
        )


if __name__ == "__main__":
    unittest.main()
