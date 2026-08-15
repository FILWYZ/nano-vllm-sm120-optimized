# M1: SM120 Batch-Paged Attention with FlashInfer

## Why this optimization

M0 showed that the correctness-first SDPA backend reconstructed contiguous KV
for every request and every layer. An 8-request x 16-token decode produced 6,736
`index_select` calls and 3.28 GiB of cumulative temporary CUDA allocation.
Single-layer paged decode took about 1.14 ms at batch 1 and 1.22 ms at batch 8.

FlashInfer 0.6.6 was selected because it supports SM120 without replacing the
validated PyTorch 2.11.0+cu128 and Triton 3.6 environment. FlashInfer 0.6.17 was
rejected for this milestone because its dependency solution replaced the full
runtime with PyTorch 2.13 and CUDA 13 packages.

## Implementation

- Added an explicit `flashinfer` attention backend and made `auto` prefer it on
  compute capability 12.x when installed.
- Added a shared `FlashInferRuntime` with one 128 MiB workspace per GPU process.
- Reused one batch plan across all Transformer layers in a model iteration.
- Used `BatchPrefillWithRaggedKVCacheWrapper` for ordinary packed prefill.
- Used `BatchPrefillWithPagedKVCacheWrapper` for prefix-cache and chunked
  prefill.
- Used `BatchDecodeWithPagedKVCacheWrapper` for decode without reconstructing
  contiguous K/V tensors.
- Kept PyTorch SDPA as the reference backend and original FlashAttention as an
  optional compatibility path.
- Added a pinned optional dependency group:
  `flashinfer-python==0.6.6` plus Ninja.

The adapter is isolated in `nanovllm/layers/flashinfer_backend.py`; the model and
scheduler remain readable and backend-independent.

## Correctness evidence

An independent SM120 probe matched PyTorch SDPA with maximum absolute error
0.000244. Three backend tests then validated:

1. ragged variable-length prefill;
2. paged prefix-cache prefill with bottom-right causal masking;
3. multi-request paged decode with GQA.

All use FP16 and compare against PyTorch SDPA with `rtol=atol=2e-3`.

## Performance evidence

Same M0 Qwen3-0.6B matrix, one warmup and two measured repeats:

| Requests | Input | M0 SDPA tok/s | M1 FlashInfer tok/s | Gain |
|---:|---:|---:|---:|---:|
| 1 | 64 | 27.9 | 30.7 | +10.0% |
| 4 | 64 | 88.2 | 115.9 | +31.4% |
| 8 | 64 | 143.5 | 235.6 | +64.2% |
| 1 | 256 | 27.3 | 30.2 | +10.6% |
| 4 | 256 | 86.5 | 115.7 | +33.8% |
| 8 | 256 | 137.2 | 226.9 | +65.4% |

Attention CUDA-event p50 comparison:

| Phase | Batch | Length | M0 SDPA | M1 FlashInfer | Change |
|---|---:|---:|---:|---:|---:|
| Prefill | 1 | 128 | 0.154 ms | 0.032 ms | -79.2% |
| Decode | 1 | 128 | 1.139 ms | 0.533 ms | -53.2% |
| Decode | 8 | 128 | 1.224 ms | 0.521 ms | -57.4% |

The M1 profile reduced:

- `index_select` from 6,736 calls / 3.28 GiB cumulative allocation to 16 calls /
  2.23 MiB;
- copy calls from 7,757 to 638;
- total profiled self CPU time from 987 ms to 557 ms (-43.6%).

Peak allocated memory for the largest measured workload was 4.724 GiB versus
roughly 4.69 GiB in M0. The shared workspace therefore adds little end-to-end
peak pressure because the removed temporary KV tensors offset most of its cost.

## Why the gain is not yet 2x end-to-end

The attention kernel itself is more than 2x faster in decode, but the full model
still performs per-layer KV updates and synchronization. The M1 profile now
shows the next dominant avoidable costs:

- 1,807 `nonzero` calls taking 156 ms self CPU time;
- 449 scalar extractions taking 34.4 ms;
- 448 `torch.any`/boolean-indexing KV-update paths;
- FlashInfer planning still transfers small metadata to the host once per model
  iteration.

These measurements define M2: replace boolean-index KV writes, remove scalar
synchronization, preallocate outputs/metadata, and make the hot path GPU-native.

## Reproduction

```bash
source .venv/bin/activate
python -m unittest tests.test_attention_flashinfer -v
python -m benchmarks.attention_micro \
  --backend flashinfer \
  --output benchmarks/results/m1_attention_flashinfer.json
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B \
  --output benchmarks/results/m1_flashinfer_end_to_end.json
```
