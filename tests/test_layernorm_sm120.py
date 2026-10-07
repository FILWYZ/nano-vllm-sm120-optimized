import pytest
import torch

from nanovllm.layers.layernorm import RMSNorm, set_rmsnorm_backend


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("tokens", [1, 16, 64, 256])
def test_sm120_fused_add_matches_torch(dtype: torch.dtype, tokens: int) -> None:
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 GPU is required")
    torch.manual_seed(23)
    module = RMSNorm(1024).cuda().to(dtype)
    x = torch.randn((tokens, 1024), device="cuda", dtype=dtype)
    residual = torch.randn_like(x)

    set_rmsnorm_backend("torch")
    expected, expected_residual = module(x.clone(), residual.clone())
    set_rmsnorm_backend("sm120")
    try:
        actual, actual_residual = module(x.clone(), residual.clone())
    finally:
        set_rmsnorm_backend("torch")

    atol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    rtol = 3e-2 if dtype == torch.bfloat16 else 5e-3
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        actual_residual, expected_residual, atol=atol, rtol=rtol
    )


def test_auto_backend_resolves_on_sm120() -> None:
    expected = "sm120" if torch.cuda.get_device_capability() == (12, 0) else "torch"
    try:
        assert set_rmsnorm_backend("auto") == expected
    finally:
        set_rmsnorm_backend("torch")
