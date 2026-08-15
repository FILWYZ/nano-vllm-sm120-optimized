# Local Blackwell Baseline

This branch preserves upstream `main` at commit
`bb823b3e06983d71485a8e1f23715ebd87d98ef8` and adapts nano-vLLM to the local
RTX 5060 Laptop GPU without requiring an unvalidated FlashAttention wheel.

## Validated environment

- Windows + WSL2 Ubuntu 24.04
- NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB, compute capability 12.0
- NVIDIA driver 610.74
- Python 3.12.3 managed by `uv`
- PyTorch 2.11.0+cu128
- Triton 3.6.0
- Qwen3-0.6B

## Compatibility changes

- `attention_backend="auto"` chooses PyTorch SDPA on compute capability 12.x.
- `attention_backend="flash"` remains available when the optional `flash-attn`
  package is installed and validated for the exact PyTorch/CUDA/GPU stack.
- SDPA supports packed prefill, prefix-cache prefill, paged-KV decode, and GQA.
- The SDPA path uses eager execution for now; CUDA graph capture remains on the
  original FlashAttention path.
- Engine shutdown is idempotent, so explicit cleanup and the `atexit` hook can
  safely coexist.

## Correctness gates

```bash
cd /home/asus/projects/nano-vllm-baseline
uv pip check --python .venv/bin/python
.venv/bin/python -m unittest tests.test_attention_sdpa -v
```

The three GPU tests compare nano-vLLM against PyTorch SDPA for:

1. packed variable-length prefill;
2. prefill with a paged prefix cache;
3. multi-sequence paged decode.

## Reproducible performance baseline

```bash
.venv/bin/python bench_local.py \
  --model /mnt/c/Users/ASUS/Documents/ChatGPT/算子/models/Qwen3-0.6B
```

Workload: 8 requests, random 64-128 token prompts, 32 output tokens per request,
512-token maximum context, and 75% GPU memory utilization.

| Backend | Output tokens | Time | Throughput | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|
| PyTorch SDPA | 256 | 4.608 s | 55.56 tok/s | 4.763 GiB | 4.832 GiB |

This is a correctness-first compatibility baseline, not a performance target.
Compare every future optimization against both the correctness tests and this
fixed workload.

## Recommended optimization order

1. Replace Python per-sequence paged-KV gathering with a Blackwell-safe Triton
   kernel while retaining SDPA as the reference implementation.
2. Make decode shapes static and restore CUDA graph capture.
3. Benchmark native FlashAttention only after an official wheel supports the
   exact PyTorch/CUDA/SM120 combination.
4. Add continuous benchmarks for prefill throughput, decode throughput, latency,
   and peak memory at several batch sizes and context lengths.
