# M0: Measurement Before Optimization

## Why this milestone exists

The original benchmark reported only aggregate output throughput. That number
cannot distinguish attention cost, prefill throughput, decode latency, Python
overhead, or GPU memory pressure. Optimizing against it alone risks improving a
microbenchmark while regressing end-to-end behavior.

M0 deliberately changes observability, not inference algorithms.

## Implementation

- `LLMEngine.last_metrics` records prefill/decode tokens, iterations, elapsed
  time, batch TTFT, mean TPOT, and output throughput after each `generate` call.
- `benchmarks/e2e/end_to_end.py` runs a fixed matrix over batch size and prompt
  length and writes machine-readable JSON.
- `benchmarks/micro/attention_micro.py` isolates SDPA prefill and paged decode using
  CUDA events and reports p50/p95 latency.
- `benchmarks/micro/profile_decode.py` exports a PyTorch CPU/CUDA Chrome trace and a
  top-operator text summary.
- Every JSON result embeds the software, GPU, compute capability, memory, and Git
  commit used to produce it.
- Every end-to-end shape has an unmeasured warmup before measured repeats so
  `torch.compile` and allocator cold-start cost do not contaminate steady-state
  throughput.

## Reproduction

```bash
cd /home/asus/projects/nano-vllm-baseline
MODEL=/mnt/c/Users/ASUS/Documents/ChatGPT/算子/models/Qwen3-0.6B

.venv/bin/python -m benchmarks.attention_micro \
  --output benchmarks/results/m0_attention_sdpa.json

.venv/bin/python -m benchmarks.end_to_end \
  --model "$MODEL" \
  --output benchmarks/results/m0_end_to_end.json

.venv/bin/python -m benchmarks.profile_decode \
  --model "$MODEL"
```

## Correctness gates

```bash
.venv/bin/python -m unittest tests.test_attention_sdpa -v
uv pip check --python .venv/bin/python
```

The benchmark is accepted only after packed prefill, prefix-cache prefill, and
paged decode match the PyTorch SDPA reference.

## Metrics and interpretation

- Prefill tokens/s reveals compute-bound prompt processing.
- Decode tokens/s and TPOT reveal memory traffic and launch overhead.
- Attention p50/p95 separates stable kernel cost from intermittent host stalls.
- Peak memory bounds the KV-cache capacity available for larger workloads.
- The CPU/CUDA trace identifies synchronization, allocation, and kernel launch
  overhead that aggregate throughput hides.

Chrome traces can exceed 100 MB and are intentionally ignored by Git. Compact
JSON results and the profiler text summary remain versioned. CUPTI CUDA activity
collection is unavailable in the current WSL driver path, so CUDA-event
microbenchmarks are the authoritative kernel latency measurement.

## Baseline result

Steady-state Qwen3-0.6B results, averaged over two measured repeats after one
shape-matched warmup:

| Requests | Input | Output | Prefill tok/s | Decode tok/s | E2E output tok/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 32 | 1,717.9 | 27.9 | 27.9 |
| 4 | 64 | 32 | 6,631.9 | 87.8 | 88.2 |
| 8 | 64 | 32 | 11,333.1 | 142.7 | 143.5 |
| 1 | 256 | 32 | 7,039.3 | 27.3 | 27.3 |
| 4 | 256 | 32 | 18,982.0 | 87.1 | 86.5 |
| 8 | 256 | 32 | 24,330.5 | 139.3 | 137.2 |

Attention-only CUDA-event p50 latency is roughly 0.15 ms for batch-1 prefill
and 1.14 ms for batch-1 paged decode. Decode remains around 1.22-1.26 ms at
batch 8, showing that fixed overhead dominates this small model.

The warmed CPU profile of 8 requests x 16 output tokens reports:

- 6,736 `index_select` calls and 3.28 GiB of cumulative temporary CUDA
  allocation attributed to paged-KV gathering;
- 1,792 `nonzero` calls consuming 157 ms self CPU time;
- 448 scalar extractions consuming 33.8 ms self CPU time;
- 7,757 copies consuming 92.1 ms self CPU time.

## Evidence for M1

The current SDPA decode implementation loops over sequences, copies paged KV
blocks into temporary contiguous tensors, calls SDPA once per request per layer,
and synchronizes metadata through `.tolist()`. M1 targets this path with a
batch-paged SM120 backend while retaining SDPA as the correctness oracle.
