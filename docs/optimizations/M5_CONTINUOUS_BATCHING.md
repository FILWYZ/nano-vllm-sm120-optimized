# M5: Chunked Prefill and Mixed Continuous Batching

## Why this optimization

The baseline scheduler had a limited chunked-prefill path: only the first
waiting request could consume a partial token budget, and any schedulable
prefill batch prevented decode from running in that step. When a long prompt
arrived while another request was decoding, all prompt chunks ran first and
created a visible inter-token latency spike.

FlashInfer paged prefill can represent both operations in one packed batch:
a decode request has query length 1 and an existing KV prefix, while a prompt
chunk has query length greater than 1. This allows true mixed execution without
adding a second model or attention implementation.

## Implementation

- Added `max_prefill_chunk_size` (default 512) as an explicit compute cap.
- Added `enable_mixed_batching` for controlled ablation; it defaults on.
- Schedules active decode requests first, then fills remaining sequence and
  token capacity with round-robin prompt chunks.
- Rotates partially processed prompts through the waiting queue instead of
  allowing the head request to monopolize a step.
- Executes mixed batches through the existing paged-prefill preparation path.
  The LM head already selects `cu_seqlens_q[1:] - 1`, so it produces exactly
  one sample per decode request or prompt chunk.
- Preserves decode-only CUDA Graph replay when no waiting prompt is present.
- Adds counters for prefill chunks, mixed/prefill-only/decode-only batches, and
  preemptions, with per-generation deltas in `LLM.last_metrics`.
- Separately accounts mixed-batch time and decode tokens so TPOT metrics do not
  silently omit decode work performed alongside prefill.

## Correctness evidence

Thirteen tests pass. New gates cover:

1. decode-first mixed scheduling;
2. the prefill-first ablation behavior;
3. round-robin chunk caps across waiting requests;
4. postprocessing a decode token and an incomplete prompt chunk in one batch;
5. FlashInfer numerical parity for one decode request plus one chunked-prefix
   request in the same paged-prefill call.

The mixed attention output matches independent PyTorch SDPA references at FP16
`rtol=atol=2e-3`. Existing graph, page-boundary, prefix, and KV-write tests also
remain green.

## Online arrival experiment

`benchmarks/online_batching.py` starts a 64-token/24-output request, lets it
produce four tokens, then injects a 512-token request split into 128-token
chunks. Both policies receive a shape-matched unmeasured warmup in isolated
processes.

| Metric | Prefill-first | Mixed | Change |
|---|---:|---:|---:|
| mean short-request inter-token gap | 8.471 ms | 7.383 ms | -12.8% |
| maximum inter-token gap | 82.330 ms | 18.948 ms | -77.0% |
| scheduler steps after arrival | 24 | 20 | -16.7% |
| mixed batches | 0 | 4 | expected |
| prefill-only batches | 5 | 1 | -80.0% |

The baseline's p95 is not used for the decision because this 20-gap workload
has one deliberate arrival stall and the nearest-rank percentile excludes that
single maximum. Maximum gap directly measures the head-of-line blocking event.

The long request still completes with one output token under both policies, and
neither run preempts a request.

## Offline throughput regression

The fixed M4 matrix was rerun with one warmup and two measured repeats:

| Requests | Input | M4 tok/s | M5 tok/s | Change |
|---:|---:|---:|---:|---:|
| 1 | 64 | 197.8 | 199.3 | +0.8% |
| 4 | 64 | 767.1 | 778.3 | +1.5% |
| 8 | 64 | 1,470.5 | 1,501.2 | +2.1% |
| 1 | 256 | 184.9 | 195.7 | +5.8% |
| 4 | 256 | 663.5 | 693.7 | +4.6% |
| 8 | 256 | 1,116.9 | 1,134.5 | +1.6% |

The fixed batch has no staggered arrivals, so the new policy should be neutral;
the small positive changes are ordinary run-to-run variance rather than an
algorithmic throughput claim.

## Why these metrics demonstrate the optimization

Scheduler counters show the exact four long-prompt chunks overlapping four
decode steps. That removes four standalone steps and cuts the intentional
arrival stall by 77%, while the fixed offline matrix does not regress. The
mixed SDPA parity test demonstrates that decode KV history and chunked causal
masking remain correct in the unified batch.

## Known boundary

Prompt KV blocks are still reserved for the full prompt at admission. M5 chunks
compute and improves fairness, but does not yet allocate prompt KV pages lazily.
That is an intentional teaching simplification; changing it requires admission
control and partial-prefix eviction semantics together.

## Evidence for M6

M4 made prefix blocks 16-token granular, but the cache currently uses a free
deque rather than an explicit LRU policy and exposes no hit/eviction metrics.
M6 should add recency tracking, collision-safe lookup statistics, deterministic
offline request reordering, and a repeated-prefix benchmark. The expected
proof is lower prefill tokens/TTFT at a known hit rate without changing output.

## Reproduction

```bash
python -m benchmarks.online_batching \
  --model /path/to/Qwen3-0.6B --policy mixed \
  --output benchmarks/results/m5_online_mixed.json
python -m benchmarks.online_batching \
  --model /path/to/Qwen3-0.6B --policy prefill-first \
  --output benchmarks/results/m5_online_prefill_first.json
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B --backend flashinfer --block-size 16 \
  --output benchmarks/results/m5_end_to_end.json
```
