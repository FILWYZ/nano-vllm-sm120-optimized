import unittest

import torch
import torch.nn.functional as F

from nanovllm.layers.attention import Attention, set_attention_backend
from nanovllm.utils.context import reset_context, set_context


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSDPAAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        set_attention_backend("sdpa")
        self.device = torch.device("cuda")
        self.dtype = torch.float16
        self.num_heads = 4
        self.num_kv_heads = 2
        self.head_dim = 16
        self.scale = self.head_dim ** -0.5
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scale, self.num_kv_heads
        ).to(self.device)

    def tearDown(self):
        reset_context()

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

    def test_packed_prefill(self):
        q1, k1, v1 = self.random_qkv(3)
        q2, k2, v2 = self.random_qkv(5)
        q, k, v = map(lambda xs: torch.cat(xs), ((q1, q2), (k1, k2), (v1, v2)))
        cu = torch.tensor([0, 3, 8], device=self.device, dtype=torch.int32)
        set_context(True, cu, cu, 5, 5, torch.empty(0, device=self.device,
                                                    dtype=torch.int32))

        actual = self.attn(q, k, v)
        expected = torch.cat((self.reference(q1, k1, v1, True),
                              self.reference(q2, k2, v2, True)))
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_prefix_cache_prefill(self):
        block_size = 4
        self.attn.k_cache = torch.zeros(2, block_size, self.num_kv_heads,
                                        self.head_dim, device=self.device,
                                        dtype=self.dtype)
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)
        prefix_k = torch.randn(2, self.num_kv_heads, self.head_dim,
                               device=self.device, dtype=self.dtype)
        prefix_v = torch.randn_like(prefix_k)
        self.attn.k_cache[0, :2] = prefix_k
        self.attn.v_cache[0, :2] = prefix_v
        q, k, v = self.random_qkv(2)
        cu_q = torch.tensor([0, 2], device=self.device, dtype=torch.int32)
        cu_k = torch.tensor([0, 4], device=self.device, dtype=torch.int32)
        slots = torch.tensor([2, 3], device=self.device, dtype=torch.int32)
        blocks = torch.tensor([[0]], device=self.device, dtype=torch.int32)
        set_context(True, cu_q, cu_k, 2, 4, slots, None, blocks)

        actual = self.attn(q, k, v)
        full_k, full_v = torch.cat((prefix_k, k)), torch.cat((prefix_v, v))
        mask = torch.tensor([[True, True, True, False],
                             [True, True, True, True]], device=self.device)
        expected = self.reference(q, full_k, full_v, False, mask)
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

    def test_paged_decode(self):
        block_size = 4
        self.attn.k_cache = torch.zeros(3, block_size, self.num_kv_heads,
                                        self.head_dim, device=self.device,
                                        dtype=self.dtype)
        self.attn.v_cache = torch.zeros_like(self.attn.k_cache)
        old_k1 = torch.randn(2, self.num_kv_heads, self.head_dim,
                             device=self.device, dtype=self.dtype)
        old_v1 = torch.randn_like(old_k1)
        old_k2 = torch.randn(4, self.num_kv_heads, self.head_dim,
                             device=self.device, dtype=self.dtype)
        old_v2 = torch.randn_like(old_k2)
        self.attn.k_cache[0, :2], self.attn.v_cache[0, :2] = old_k1, old_v1
        self.attn.k_cache[1, :4], self.attn.v_cache[1, :4] = old_k2, old_v2
        q, k, v = self.random_qkv(2)
        slots = torch.tensor([2, 8], device=self.device, dtype=torch.int32)
        lengths = torch.tensor([3, 5], device=self.device, dtype=torch.int32)
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
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
