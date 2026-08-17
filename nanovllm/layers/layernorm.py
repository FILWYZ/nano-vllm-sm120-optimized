import torch
from torch import nn

try:
    from sm120_rmsnorm import fused_add_rms_norm_inplace
except ImportError:
    fused_add_rms_norm_inplace = None


_RMSNORM_BACKEND = "torch"


def set_rmsnorm_backend(backend: str) -> str:
    global _RMSNORM_BACKEND
    if backend == "auto":
        is_sm120 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)
        backend = "sm120" if is_sm120 and fused_add_rms_norm_inplace is not None else "torch"
    if backend not in {"torch", "sm120"}:
        raise ValueError(f"unsupported RMSNorm backend: {backend}")
    if backend == "sm120" and fused_add_rms_norm_inplace is None:
        raise RuntimeError(
            "SM120 RMSNorm backend requested, but sm120_rmsnorm is not installed"
        )
    _RMSNORM_BACKEND = backend
    return backend


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        if (
            _RMSNORM_BACKEND == "sm120"
            and x.is_cuda
            and x.is_contiguous()
            and residual.is_contiguous()
            and self.weight.is_contiguous()
        ):
            fused_add_rms_norm_inplace(x, residual, self.weight, self.eps)
            return x, residual
        return self.add_rms_forward(x, residual)
