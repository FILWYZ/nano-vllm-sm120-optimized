import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = flash_attn_with_kvcache = None

from nanovllm.layers.flashinfer_backend import (
    flashinfer_available,
    flashinfer_forward,
)
from nanovllm.utils.context import get_context

_ATTENTION_BACKEND = "auto"


def resolve_attention_backend(backend: str) -> str:
    if backend not in {"auto", "flash", "flashinfer", "sdpa"}:
        raise ValueError(f"Unsupported attention backend: {backend}")
    if backend == "flash":
        if flash_attn_varlen_func is None:
            raise RuntimeError(
                "attention_backend='flash' requires the optional flash-attn package"
            )
        return "flash"
    if backend == "flashinfer":
        if not flashinfer_available():
            raise RuntimeError(
                "attention_backend='flashinfer' requires the flashinfer optional dependency"
            )
        return "flashinfer"
    if backend == "sdpa":
        return "sdpa"

    # FlashAttention wheels frequently lag new GPU architectures. Prefer the
    # PyTorch backend on Blackwell until a wheel explicitly validated for the
    # installed PyTorch/CUDA combination is available.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
        return "flashinfer" if flashinfer_available() else "sdpa"
    return "flash" if flash_attn_varlen_func is not None else "sdpa"


def set_attention_backend(backend: str) -> str:
    global _ATTENTION_BACKEND
    _ATTENTION_BACKEND = resolve_attention_backend(backend)
    return _ATTENTION_BACKEND


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    if slot_mapping.numel() != N:
        context = get_context()
        raise AssertionError(
            f"KV slot mismatch: slots={slot_mapping.numel()} tokens={N} "
            f"is_prefill={context.is_prefill} "
            f"cu_q={context.cu_seqlens_q.tolist() if context.cu_seqlens_q is not None else None} "
            f"cu_k={context.cu_seqlens_k.tolist() if context.cu_seqlens_k is not None else None}"
        )
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_torch(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
                       v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    slots = slot_mapping.to(dtype=torch.long)
    valid = slots >= 0
    if not torch.any(valid):
        return
    flat_k_cache = k_cache.view(-1, *k_cache.shape[-2:])
    flat_v_cache = v_cache.view(-1, *v_cache.shape[-2:])
    flat_k_cache.index_copy_(0, slots[valid], key[valid])
    flat_v_cache.index_copy_(0, slots[valid], value[valid])


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.backend = resolve_attention_backend(_ATTENTION_BACKEND)

    @staticmethod
    def _paged_kv(cache: torch.Tensor, block_table: torch.Tensor, length: int):
        block_size = cache.size(1)
        num_blocks = (length + block_size - 1) // block_size
        blocks = block_table[:num_blocks].to(dtype=torch.long)
        return cache.index_select(0, blocks).flatten(0, 1)[:length]

    def _sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              is_causal: bool, attn_mask: torch.Tensor | None = None):
        output = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0),
            k.transpose(0, 1).unsqueeze(0),
            v.transpose(0, 1).unsqueeze(0),
            attn_mask=attn_mask,
            is_causal=is_causal,
            scale=self.scale,
            enable_gqa=self.num_heads != self.num_kv_heads,
        )
        return output.squeeze(0).transpose(0, 1)

    def _forward_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      context):
        outputs = []
        if context.is_prefill:
            q_offsets = context.cu_seqlens_q.tolist()
            k_offsets = context.cu_seqlens_k.tolist()
            for i in range(len(q_offsets) - 1):
                q_i = q[q_offsets[i]:q_offsets[i + 1]]
                if context.block_tables is None:
                    k_i = k[k_offsets[i]:k_offsets[i + 1]]
                    v_i = v[k_offsets[i]:k_offsets[i + 1]]
                    outputs.append(self._sdpa(q_i, k_i, v_i, is_causal=True))
                    continue

                k_len = k_offsets[i + 1] - k_offsets[i]
                k_i = self._paged_kv(self.k_cache, context.block_tables[i], k_len)
                v_i = self._paged_kv(self.v_cache, context.block_tables[i], k_len)
                q_len = q_i.size(0)
                prefix_len = k_len - q_len
                q_positions = torch.arange(q_len, device=q.device) + prefix_len
                k_positions = torch.arange(k_len, device=q.device)
                causal_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
                outputs.append(self._sdpa(q_i, k_i, v_i, False, causal_mask))
        else:
            for i, context_len in enumerate(context.context_lens.tolist()):
                k_i = self._paged_kv(self.k_cache, context.block_tables[i], context_len)
                v_i = self._paged_kv(self.v_cache, context.block_tables[i], context_len)
                outputs.append(self._sdpa(q[i:i + 1], k_i, v_i, is_causal=False))
        return torch.cat(outputs, dim=0)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_already_stored: bool = False,
    ):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel() and not kv_already_stored:
            if self.backend in {"flash", "flashinfer"}:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            else:
                store_kvcache_torch(k, v, k_cache, v_cache, context.slot_mapping)
        if self.backend == "flashinfer":
            return flashinfer_forward(
                q, k, v, k_cache, v_cache, context,
                self.num_heads, self.num_kv_heads, self.head_dim, self.scale,
            )
        if self.backend == "sdpa":
            return self._forward_sdpa(q, k, v, context)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o
