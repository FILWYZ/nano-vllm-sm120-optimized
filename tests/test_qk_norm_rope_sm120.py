import pytest
import torch

from nanovllm.layers.qk_norm_rope import (
    apply_sm120_qk_norm_rope,
    set_qk_norm_rope_backend,
    use_sm120_qk_norm_rope,
)
from nanovllm.layers.rotary_embedding import RotaryEmbedding
from nanovllm.layers.layernorm import RMSNorm


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 16, 64, 256])
def test_sm120_qk_norm_rope_matches_torch(dtype: torch.dtype, tokens: int) -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU is required")
    torch.manual_seed(29)
    q_norm = RMSNorm(128).cuda().to(dtype)
    k_norm = RMSNorm(128).cuda().to(dtype)
    rope = RotaryEmbedding(128, 128, 512, 1_000_000.0).cuda()
    query = torch.randn((tokens, 16, 128), device="cuda", dtype=dtype)
    key = torch.randn((tokens, 8, 128), device="cuda", dtype=dtype)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)

    expected_q = q_norm(query.clone())
    expected_k = k_norm(key.clone())
    expected_q, expected_k = rope(positions, expected_q, expected_k)

    actual_q, actual_k = query.clone(), key.clone()
    set_qk_norm_rope_backend("sm120")
    try:
        assert use_sm120_qk_norm_rope(
            actual_q, actual_k, positions, rope.cos_sin_cache
        )
        apply_sm120_qk_norm_rope(
            actual_q,
            actual_k,
            q_norm.weight,
            k_norm.weight,
            positions,
            rope.cos_sin_cache,
            q_norm.eps,
        )
    finally:
        set_qk_norm_rope_backend("torch")

    atol = 4e-2 if dtype == torch.bfloat16 else 6e-3
    rtol = 4e-2 if dtype == torch.bfloat16 else 6e-3
    torch.testing.assert_close(actual_q, expected_q, atol=atol, rtol=rtol)
    torch.testing.assert_close(actual_k, expected_k, atol=atol, rtol=rtol)


def test_qk_backend_falls_back_for_unsupported_head_dim() -> None:
    query = torch.empty((1, 16, 64), device="cuda", dtype=torch.float16)
    key = torch.empty((1, 8, 64), device="cuda", dtype=torch.float16)
    positions = torch.zeros(1, device="cuda", dtype=torch.int64)
    cache = torch.empty((1, 1, 64), device="cuda", dtype=torch.float32)
    set_qk_norm_rope_backend("sm120")
    try:
        assert not use_sm120_qk_norm_rope(query, key, positions, cache)
    finally:
        set_qk_norm_rope_backend("torch")
