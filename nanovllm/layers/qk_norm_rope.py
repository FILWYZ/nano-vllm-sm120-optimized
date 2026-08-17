import torch

try:
    from sm120_rmsnorm import (
        fused_qk_rms_norm_rope,
        fused_qk_rms_norm_rope_kv,
    )
except ImportError:
    fused_qk_rms_norm_rope = None
    fused_qk_rms_norm_rope_kv = None


_QK_NORM_ROPE_BACKEND = "torch"


def set_qk_norm_rope_backend(backend: str) -> str:
    global _QK_NORM_ROPE_BACKEND
    if backend == "auto":
        is_sm120 = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability() == (12, 0)
        )
        backend = "sm120" if is_sm120 and fused_qk_rms_norm_rope is not None else "torch"
    if backend not in {"torch", "sm120"}:
        raise ValueError(f"unsupported QK Norm+RoPE backend: {backend}")
    if backend == "sm120" and fused_qk_rms_norm_rope is None:
        raise RuntimeError(
            "SM120 QK Norm+RoPE backend requested, but sm120_rmsnorm is not installed"
        )
    _QK_NORM_ROPE_BACKEND = backend
    return backend


def use_sm120_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> bool:
    return (
        _QK_NORM_ROPE_BACKEND == "sm120"
        and fused_qk_rms_norm_rope is not None
        and query.is_cuda
        and key.is_cuda
        and query.dtype in {torch.float16, torch.bfloat16, torch.float32}
        and key.dtype == query.dtype
        and query.ndim == 3
        and key.ndim == 3
        and query.shape[0] == key.shape[0]
        and query.shape[-1] == 128
        and key.shape[-1] == 128
        and query.is_contiguous()
        and key.is_contiguous()
        and positions.is_cuda
        and positions.dtype == torch.int64
        and positions.is_contiguous()
        and cos_sin_cache.is_cuda
        and cos_sin_cache.dtype == torch.float32
        and cos_sin_cache.is_contiguous()
    )


def apply_sm120_qk_norm_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_qk_rms_norm_rope(
        query,
        key,
        query_weight,
        key_weight,
        positions,
        cos_sin_cache,
        epsilon,
    )
    return query, key


def apply_sm120_qk_norm_rope_kv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    fused_qk_rms_norm_rope_kv(
        query,
        key,
        value,
        query_weight,
        key_weight,
        positions,
        cos_sin_cache,
        key_cache,
        value_cache,
        slot_mapping,
        epsilon,
    )
    return query, key
