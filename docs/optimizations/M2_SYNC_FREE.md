# M2: Synchronization-Free KV Append and Buffer Reuse

## Why this optimization

M1 removed per-request KV gathering, exposing a different bottleneck in the
correctness-first KV write path. Every Transformer layer converted slot indices
to `int64`, built a validity mask, called `torch.any`, and then used boolean
indexing twice. On the 8-request x 16-token profile this caused 1,807 `nonzero`
calls, 449 scalar extractions, and repeated small allocations.

This is particularly expensive for a small model: GPU kernels are short, so a
host-visible scalar synchronization can cost more than the actual KV copy.

## Implementation

- FlashInfer and FlashAttention now use the existing Triton KV append kernel.
  It consumes the `int32` slot map directly and ignores `-1` padded slots in
  the kernel, without host synchronization or boolean-index temporaries.
- SDPA retains the explicit PyTorch implementation as the readable reference
  path.
- FlashInfer output tensors are cached by device, dtype, and shape and supplied
  through its `out=` API. The same buffer can be reused across layers because
  dependent projection and attention kernels execute in order on one CUDA
  stream.
- The page-column tensor used to form FlashInfer metadata is cached by device
  and page-table width instead of recreating `arange` each iteration.
- A regression test verifies that real slots are written exactly and a `-1`
  graph-padding slot leaves cache contents untouched.

The optimization deliberately does not cache request-dependent page indices or
lengths. Reusing stale metadata would be incorrect when decoding crosses a KV
page boundary.

## Correctness gates

Seven SM120 GPU tests pass:

1. SDPA packed prefill;
2. SDPA prefix-cache prefill;
3. SDPA paged decode;
4. FlashInfer ragged prefill;
5. FlashInfer paged prefix prefill;
6. FlashInfer paged GQA decode;
7. Triton KV append with a `-1` padded slot.

Attention outputs remain checked against PyTorch SDPA at FP16
`rtol=atol=2e-3`.

## Performance evidence

Same Qwen3-0.6B matrix, one shape warmup and two measured repeats:

| Requests | Input | M1 tok/s | M2 tok/s | Gain |
|---:|---:|---:|---:|---:|
| 1 | 64 | 30.7 | 60.7 | +97.7% |
| 4 | 64 | 115.9 | 222.8 | +92.2% |
| 8 | 64 | 235.6 | 445.1 | +88.9% |
| 1 | 256 | 30.2 | 59.0 | +95.4% |
| 4 | 256 | 115.7 | 214.0 | +85.0% |
| 8 | 256 | 226.9 | 408.7 | +80.1% |

For the warmed 8-request x 16-token profile:

| Signal | M1 | M2 | Interpretation |
|---|---:|---:|---|
| `nonzero` calls | 1,807 | 15 | request metadata only |
| scalar extractions | 449 | 1 | hot-path synchronization removed |
| copy calls | 638 | 190 | fewer dtype/mask temporaries |
| profiler self CPU | 557 ms | 280 ms | -49.7% |
| KV-related `index_select` allocation | 2.23 MiB | 0 | remaining 2.23 MiB is token embedding |

CUPTI remains unavailable in this WSL driver path, so CUDA activity columns in
the profiler are incomplete. End-to-end CUDA synchronization and CUDA-event
microbenchmarks remain the authoritative performance measurements.

## Why these metrics demonstrate the optimization

The intended causal chain is visible at three levels: the exact operations
targeted by the change nearly disappear from the trace, CPU dispatch time is
halved, and every fixed end-to-end workload improves by at least 80%. The
separate padded-slot test prevents gaining speed by silently corrupting graph
padding or KV placement.

## Evidence for M3

After M2, the profile is dominated by model dispatch rather than KV metadata:

- 2,720 TorchDynamo cache lookups;
- 2,720 compiled-region prologues;
- 1,808 matrix-multiply launches;
- 448 invocations for several per-layer compiled regions.

The next optimization should therefore capture stable decode shapes in CUDA
Graphs and bucket request counts. M3 must keep FlashInfer planning outside the
captured graph, use stable buffers, and prove parity for padded batch buckets.

## Reproduction

```bash
source .venv/bin/activate
python -m unittest tests.test_attention_sdpa tests.test_attention_flashinfer -v
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B --backend flashinfer \
  --output benchmarks/results/m2_sync_free_end_to_end.json
python -m benchmarks.profile_decode \
  --model /path/to/Qwen3-0.6B \
  --summary benchmarks/results/m2_profile.txt
```
