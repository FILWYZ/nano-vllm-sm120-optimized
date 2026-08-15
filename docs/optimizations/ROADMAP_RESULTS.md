# nano-vLLM Local SM120 Optimization Results

## Scope

This branch remains an offline, teaching-oriented inference engine. It targets
the local NVIDIA GeForce RTX 5060 Laptop GPU (compute capability 12.0), WSL2,
PyTorch 2.11.0+cu128, Triton 3.6, FlashInfer 0.6.6, and Qwen3-0.6B FP16.
Results should not be generalized to other models, GPUs, context lengths, or
online traffic without rerunning the versioned benchmark suite.

## Reproducible version index

| Stage | Git commit | Tag | Primary result |
|---|---|---|---|
| V0 | `7aa1f25` | `baseline-blackwell-v0` | SM120 SDPA baseline runs |
| M0 | `acafa11` | `m0-measurement` | fixed metrics/profiler baseline |
| M1 | `88b92d1` | `m1-flashinfer` | batch-paged FlashInfer attention |
| M2 | `7d52dd8` | `m2-sync-free` | synchronization-free KV append |
| M3 | `a95c408` | `m3-cudagraph` | bucketed decode CUDA Graphs |
| M4 | `677d73c` | `m4-kv-pages` | selected 16-token KV pages |
| M5 | `6e5441e` | `m5-continuous-batching` | mixed continuous batching |
| M6 | `3bcfe31` | `m6-prefix-cache` | observable LRU + offline reorder |
| M7 | recorded by tag `m7-final` | `m7-final` | greedy/buffered sampling |

Every stage is a rollback point. Machine-readable JSON and compact profiler
summaries live under `benchmarks/results`; large Chrome traces are ignored.

## Headline fixed-matrix result

M0 SDPA versus final M7 stochastic sampling (`temperature=0.6`):

| Requests | Input | M0 tok/s | M7 tok/s | Overall speedup |
|---:|---:|---:|---:|---:|
| 1 | 64 | 27.9 | 198.3 | 7.1x |
| 4 | 64 | 88.2 | 745.4 | 8.5x |
| 8 | 64 | 143.5 | 1,440.2 | 10.0x |
| 1 | 256 | 27.3 | 188.6 | 6.9x |
| 4 | 256 | 86.5 | 683.9 | 7.9x |
| 8 | 256 | 137.2 | 1,107.4 | 8.1x |

This is not one optimization's effect. It is the cumulative change from a
correctness-first eager SDPA path to paged attention, synchronization removal,
graph replay, smaller KV pages, and scheduler/cache/sampling work.

Final greedy mode reaches 1,543.3 tok/s at 8 requests/input 64, but it is listed
separately because greedy and stochastic sampling have different semantics.

## What each milestone proved

### M0: measure before changing

Separated prefill/decode throughput, TTFT, TPOT, peak memory, attention latency,
and profiler evidence. It identified 6,736 paged-KV gathers and 3.28 GiB of
cumulative temporary allocation.

### M1: use a native paged-attention backend

FlashInfer removed per-request contiguous KV reconstruction. Batch-8 throughput
rose 64-65%, while attention decode p50 fell 53-57%.

### M2: remove synchronization around KV writes

The Triton append kernel reduced `nonzero` calls from 1,807 to 15 and scalar
extractions from 449 to 1. Profiler self CPU fell 49.7%.

### M3: replay stable decode work

Bucketed CUDA Graphs cut visible TorchDynamo lookups and matrix-multiply calls
by over 92%. Batch-8/input-64 throughput reached 1,518.5 tok/s, with 48/48 exact
tokens across padded buckets and a page boundary.

### M4: trade less than 5% speed for usable KV capacity

A five-size isolated sweep selected 16-token pages. They reduce expected tail
waste 94.1% and make prefix reuse 16x finer while staying within the declared
throughput budget.

### M5: protect decode when prompts arrive

Mixed decode+prefill batches reduced the deliberate online maximum inter-token
gap from 82.33 ms to 18.95 ms (-77%) without offline throughput regression.

### M6: make cache behavior explicit

LRU/FIFO metrics expose hits, collisions, and evictions. Stable offline prefix
clustering reduced prefill tokens 42.9%, raised hit rate 60.0% to 92.3%, and
preserved the original output order exactly.

### M7: avoid random-sampling work when deterministic output is requested

Greedy sampler p50 is 65-74% lower than stochastic sampling at Qwen3 vocabulary
size and improves selected end-to-end workloads by up to 7.2%.

## Correctness gates

The final suite contains 21 passing CPU/GPU tests covering:

- SDPA packed/prefix/decode reference paths;
- FlashInfer ragged, paged, mixed, and GQA attention;
- Triton KV writes and padded slots;
- graph/eager output parity and page boundaries;
- variable page allocation and prefix hash reuse;
- mixed scheduler fairness and ablation behavior;
- LRU/FIFO/collision semantics;
- greedy, stochastic-buffer, and mixed sampling behavior.

`uv pip check` reports all 63 installed packages compatible.

## How to interpret the metrics

- Output tok/s is the user-visible offline throughput, but cannot diagnose a
  bottleneck alone.
- Decode tok/s and TPOT identify memory/launch sensitivity.
- TTFT and prefill tokens expose prompt and cache effects.
- Maximum inter-token gap is the fairness signal for staggered arrivals.
- Cache token hits prove skipped model work; hit rate alone can hide request
  size differences.
- Peak allocated/reserved memory bounds capacity, while expected tail waste
  estimates how much of that capacity is usable.
- Operator counts and CPU self time establish the causal mechanism behind a
  speedup.

## Standard reproduction

```bash
cd /home/asus/projects/nano-vllm-baseline
source .venv/bin/activate
MODEL=/mnt/c/Users/ASUS/Documents/ChatGPT/算子/models/Qwen3-0.6B

python -m unittest discover -s tests -v
uv pip check
python -m benchmarks.end_to_end \
  --model "$MODEL" --backend flashinfer --block-size 16 \
  --output benchmarks/results/reproduction.json
```

For a specific claim, use the reproduction command in that milestone's
document. Do not compare cold-start runs with shape-warmed JSON.

## Honest remaining limits

- Results cover one small Qwen model and one 8 GiB laptop GPU.
- CUPTI CUDA activities are unavailable in the current WSL driver path; CUDA
  events and synchronized end-to-end timings are authoritative.
- FlashInfer planning retains small host/device metadata copies.
- Prompt KV is not lazily allocated per chunk.
- No FP8 KV, speculative decoding, tensor parallel, top-k/top-p, or multi-model
  evaluation is claimed.

These are the measured next directions, not hidden completion criteria for the
M0-M7 baseline.
