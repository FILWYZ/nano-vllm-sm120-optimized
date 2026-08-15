import unittest

import torch
import torch.nn.functional as F

from nanovllm.layers.attention import (
    Attention, set_attention_backend, store_kvcache,
)
from nanovllm.layers.flashinfer_backend import (
    flashinfer_available,
    reset_flashinfer_runtime,
)
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available() and flashinfer_available(),
                     "CUDA and FlashInfer are required")
class TestFlashInferAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        set_attention_backend("flashinfer")
        reset_flashinfer_runtime()
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.scale = self.head_dim ** -0.5
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scale, self.num_kv_heads
        ).to(self.device)

    def tearDown(self):
        reset_context()
        reset_flashinfer_runtime()

    def reference(self, q, k, v, is_causal, attn_mask=None):
        return F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=True,
        ).squeeze(0).transpose(0, 1)

    def random_qkv(self, q_len, kv_len=None):
        kv_len = q_len if kv_len is None else kv_len
        q = torch.randn(q_len, self.num_heads, self.head_dim,
                        device=self.device, dtype=self.dtype)
        k = torch.randn(kv_len, self.num_kv_heads, self.head_dim,
                        device=self.device, dtype=self.dtype)
        v = torch.randn_like(k)
        return q, k, v

    def test_kv_store_ignores_padded_slots(self):
        page_size = 2
        key = torch.randn(3, self.num_kv_heads, self.head_dim,
                          device=self.device, dtype=self.dtype)
        value = torch.randn_like(key)
        k_cache = torch.full(
            (2, page_size, self.num_kv_heads, self.head_dim), 7.0,
            device=self.device, dtype=self.dtype,
        )
        v_cache = torch.full_like(k_cache, 11.0)
        slots = torch.tensor([0, -1, 3], device=self.device, dtype=torch.int32)

        store_kvcache(key, value, k_cache, v_cache, slots)

        torch.testing.assert_close(k_cache.view(-1, self.num_kv_heads, self.head_dim)[0], key[0])
        torch.testing.assert_close(k_cache.view(-1, self.num_kv_heads, self.head_dim)[3], key[2])
        torch.testing.assert_close(v_cache.view(-1, self.num_kv_heads, self.head_dim)[0], value[0])
        torch.testing.assert_close(v_cache.view(-1, self.num_kv_heads, self.head_dim)[3], value[2])
        self.assertTrue(torch.all(k_cache.view(-1, self.num_kv_heads, self.head_dim)[1:3] == 7))
        self.assertTrue(torch.all(v_cache.view(-1, self.num_kv_heads, self.head_dim)[1:3] == 11))

    def test_ragged_prefill(self):
        q1, k1, v1 = self.random_qkv(17)
        q2, k2, v2 = self.random_qkv(29)
        q, k, v = map(lambda xs: torch.cat(xs), ((q1, q2), (k1, k2), (v1, v2)))
        cu = torch.tensor([0, 17, 46], device=self.device, dtype=torch.int32)
        set_context(True, cu, cu, 29, 29,
                    torch.empty(0, device=self.device, dtype=torch.int32))

        actual = self.attn(q, k, v)
        expected = torch.cat((self.reference(q1, k1, v1, True),
                              self.reference(q2, k2, v2, True)))
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_paged_prefix_prefill(self):
        page_size = 16
        self.attn.k_cache = torch.zeros(2, page_size, self.num_kv_heads,
                                        self.head_dim, device=self.device,
                                        dtype=self.dtype)
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)
        prefix_k = torch.randn(14, self.num_kv_heads, self.head_dim,
                               device=self.device, dtype=self.dtype)
        prefix_v = torch.randn_like(prefix_k)
        self.attn.k_cache[0, :14] = prefix_k
        self.attn.v_cache[0, :14] = prefix_v
        q, k, v = self.random_qkv(4)
        cu_q = torch.tensor([0, 4], device=self.device, dtype=torch.int32)
        cu_k = torch.tensor([0, 18], device=self.device, dtype=torch.int32)
        slots = torch.tensor([14, 15, 16, 17], device=self.device, dtype=torch.int32)
        blocks = torch.tensor([[0, 1]], device=self.device, dtype=torch.int32)
        set_context(True, cu_q, cu_k, 4, 18, slots, None, blocks)

        actual = self.attn(q, k, v)
        full_k, full_v = torch.cat((prefix_k, k)), torch.cat((prefix_v, v))
        q_pos = torch.arange(4, device=self.device) + 14
        k_pos = torch.arange(18, device=self.device)
        mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        expected = self.reference(q, full_k, full_v, False, mask)
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_paged_decode(self):
        page_size = 16
        self.attn.k_cache = torch.zeros(3, page_size, self.num_kv_heads,
                                        self.head_dim, device=self.device,
                                        dtype=self.dtype)
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)
        old_k1 = torch.randn(14, self.num_kv_heads, self.head_dim,
                             device=self.device, dtype=self.dtype)
        old_v1 = torch.randn_like(old_k1)
        old_k2 = torch.randn(16, self.num_kv_heads, self.head_dim,
                             device=self.device, dtype=self.dtype)
        old_v2 = torch.randn_like(old_k2)
        self.attn.k_cache[0, :14], self.attn.v_cache[0, :14] = old_k1, old_v1
        self.attn.k_cache[1, :16], self.attn.v_cache[1, :16] = old_k2, old_v2
        q, k, v = self.random_qkv(2)
        slots = torch.tensor([14, 32], device=self.device, dtype=torch.int32)
        lengths = torch.tensor([15, 17], device=self.device, dtype=torch.int32)
        blocks = torch.tensor([[0, -1], [1, 2]], device=self.device,
                              dtype=torch.int32)
        set_context(False, slot_mapping=slots, context_lens=lengths,
                    block_tables=blocks)

        actual = self.attn(q, k, v)
        expected1 = self.reference(q[:1], torch.cat((old_k1, k[:1])),
                                   torch.cat((old_v1, v[:1])), False)
        expected2 = self.reference(q[1:], torch.cat((old_k2, k[1:])),
                                   torch.cat((old_v2, v[1:])), False)
        expected = torch.cat((expected1, expected2))
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)

    def test_mixed_decode_and_chunked_prefill(self):
        page_size = 16
        self.attn.k_cache = torch.zeros(
            3, page_size, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)
        old_decode_k = torch.randn(
            5, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        old_decode_v = torch.randn_like(old_decode_k)
        prefix_k = torch.randn(
            14, self.num_kv_heads, self.head_dim,
            device=self.device, dtype=self.dtype,
        )
        prefix_v = torch.randn_like(prefix_k)
        self.attn.k_cache[0, :5] = old_decode_k
        self.attn.v_cache[0, :5] = old_decode_v
        self.attn.k_cache[1, :14] = prefix_k
        self.attn.v_cache[1, :14] = prefix_v
        decode_q, decode_k, decode_v = self.random_qkv(1)
        prefill_q, prefill_k, prefill_v = self.random_qkv(4)
        q = torch.cat((decode_q, prefill_q))
        k = torch.cat((decode_k, prefill_k))
        v = torch.cat((decode_v, prefill_v))
        cu_q = torch.tensor([0, 1, 5], device=self.device, dtype=torch.int32)
        cu_k = torch.tensor([0, 6, 24], device=self.device, dtype=torch.int32)
        slots = torch.tensor(
            [5, 30, 31, 32, 33], device=self.device, dtype=torch.int32
        )
        blocks = torch.tensor(
            [[0, -1], [1, 2]], device=self.device, dtype=torch.int32
        )
        set_context(True, cu_q, cu_k, 4, 18, slots, None, blocks)

        actual = self.attn(q, k, v)
        expected_decode = self.reference(
            decode_q, torch.cat((old_decode_k, decode_k)),
            torch.cat((old_decode_v, decode_v)), False,
        )
        full_k = torch.cat((prefix_k, prefill_k))
        full_v = torch.cat((prefix_v, prefill_v))
        q_positions = torch.arange(4, device=self.device) + 14
        k_positions = torch.arange(18, device=self.device)
        mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
        expected_prefill = self.reference(
            prefill_q, full_k, full_v, False, mask
        )
        torch.testing.assert_close(
            actual, torch.cat((expected_decode, expected_prefill)),
            rtol=2e-3, atol=2e-3,
        )
