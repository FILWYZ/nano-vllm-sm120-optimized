"""Small FlashInfer adapter kept separate from the readable SDPA reference."""

import os
import sys
from pathlib import Path

import torch

try:
    import flashinfer
except ImportError:
    flashinfer = None


def flashinfer_available() -> bool:
    return flashinfer is not None


def _ensure_build_tools_on_path():
    # FlashInfer invokes the `ninja` executable during first-use JIT. Running a
    # venv's Python by absolute path does not automatically prepend its bin dir.
    venv_bin = str(Path(sys.executable).parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if venv_bin not in path_entries:
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


class FlashInferRuntime:
    """Owns shared wrappers so one batch plan is reused by every model layer."""

    def __init__(self):
        self.workspace = None
        self.ragged_prefill = None
        self.paged_prefill = None
        self.decode = None
        self.planned_context = None
        self.planned_mode = None
        self.planned_signature = None

    def _workspace(self, device):
        if self.workspace is None or self.workspace.device != device:
            _ensure_build_tools_on_path()
            self.workspace = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            self.ragged_prefill = None
            self.paged_prefill = None
            self.decode = None
        return self.workspace

    @staticmethod
    def _page_metadata(block_tables, seq_lens, page_size):
        seq_lens = seq_lens.to(dtype=torch.int32)
        num_pages = torch.div(seq_lens + page_size - 1, page_size, rounding_mode="floor")
        page_columns = torch.arange(block_tables.size(1), device=block_tables.device)
        page_mask = page_columns.unsqueeze(0) < num_pages.unsqueeze(1)
        indices = block_tables[page_mask].to(dtype=torch.int32)
        indptr = torch.cat((
            torch.zeros(1, dtype=torch.int32, device=block_tables.device),
            num_pages.cumsum(0, dtype=torch.int32),
        ))
        last_page_len = (seq_lens - 1).remainder(page_size) + 1
        return indptr, indices, last_page_len

    def _plan(self, context, q, k, k_cache, num_heads, num_kv_heads, head_dim, scale):
        mode = "ragged_prefill" if context.is_prefill and context.block_tables is None else (
            "paged_prefill" if context.is_prefill else "decode"
        )
        signature = (mode, num_heads, num_kv_heads, head_dim, q.dtype)
        if self.planned_context is context and self.planned_signature == signature:
            return mode

        workspace = self._workspace(q.device)
        if mode == "ragged_prefill":
            if self.ragged_prefill is None:
                self.ragged_prefill = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
                    workspace, "NHD", backend="auto"
                )
            self.ragged_prefill.plan(
                context.cu_seqlens_q,
                context.cu_seqlens_k,
                num_heads,
                num_kv_heads,
                head_dim,
                causal=True,
                sm_scale=scale,
                q_data_type=q.dtype,
                kv_data_type=k.dtype,
            )
        else:
            page_size = k_cache.size(1)
            if context.is_prefill:
                seq_lens = context.cu_seqlens_k[1:] - context.cu_seqlens_k[:-1]
            else:
                seq_lens = context.context_lens
            indptr, indices, last_page_len = self._page_metadata(
                context.block_tables, seq_lens, page_size
            )
            if mode == "paged_prefill":
                if self.paged_prefill is None:
                    self.paged_prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                        workspace, "NHD", backend="auto"
                    )
                self.paged_prefill.plan(
                    context.cu_seqlens_q,
                    indptr,
                    indices,
                    last_page_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    page_size,
                    causal=True,
                    sm_scale=scale,
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                )
            else:
                if self.decode is None:
                    self.decode = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                        workspace, "NHD", backend="auto"
                    )
                self.decode.plan(
                    indptr,
                    indices,
                    last_page_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    page_size,
                    sm_scale=scale,
                    q_data_type=q.dtype,
                    kv_data_type=k_cache.dtype,
                )
        self.planned_context = context
        self.planned_mode = mode
        self.planned_signature = signature
        return mode

    def forward(self, q, k, v, k_cache, v_cache, context,
                num_heads, num_kv_heads, head_dim, scale):
        mode = self._plan(
            context, q, k, k_cache, num_heads, num_kv_heads, head_dim, scale
        )
        if mode == "ragged_prefill":
            return self.ragged_prefill.run(q, k, v)
        if mode == "paged_prefill":
            return self.paged_prefill.run(q, (k_cache, v_cache))
        return self.decode.run(q, (k_cache, v_cache))

    def reset(self):
        self.planned_context = None
        self.planned_mode = None
        self.planned_signature = None


_RUNTIME = FlashInferRuntime()


def flashinfer_forward(*args, **kwargs):
    return _RUNTIME.forward(*args, **kwargs)


def reset_flashinfer_runtime():
    _RUNTIME.reset()
