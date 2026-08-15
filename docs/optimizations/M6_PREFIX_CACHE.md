# M6: Observable LRU Prefix Cache and Offline Reordering

## Why this optimization

The baseline retained hashes on freed KV blocks, but eviction was an implicit
property of deque order. It had no named policy, no hit/collision/eviction
counters, and no way for an offline workload to cluster requests that share a
prefix. As a result, cache behavior could not be evaluated or explained.

M4's 16-token pages make fine-grained reuse possible. M6 turns that mechanism
into an explicit, measurable teaching subsystem.

## Implementation

- Added `prefix_cache_policy={lru,fifo}` with LRU as the default.
- A successful lookup of a free cached block moves it to the MRU end under LRU;
  FIFO preserves the original order for ablation.
- Counts prefix queries, block/token hits, misses, collision rejections, and
  cached-block evictions.
- Retains collision-safe verification: a hash match is accepted only when the
  stored token block is exactly equal.
- Exposes per-`generate` prefix-cache deltas in `LLM.last_metrics`.
- Honors a positive `num_kvcache_blocks` as an upper bound, enabling controlled
  cache-pressure experiments while still clamping it to available GPU memory.
- Added optional stable offline reordering to `generate`. Requests are sorted by
  a configurable token-prefix key before admission, while an original-index to
  sequence-ID map restores the caller's output order.
- Reordering is opt-in because an online service cannot generally delay and
  globally reorder interactive arrivals.

## Correctness evidence

Sixteen tests pass. New cache-specific tests verify:

1. an LRU hit changes the next eviction victim;
2. FIFO lookup leaves eviction order unchanged;
3. an injected hash collision is counted and rejected;
4. all prior scheduler, attention, graph, page, and KV behaviors remain valid.

The FIFO, LRU, and LRU+reorder GPU runs also produced exactly the same 12 output
tokens in original request order. Thus clustering changes admission order and
compute reuse, not the API's result order.

## Cache-pressure experiment

The benchmark uses 12 requests from three interleaved prefix families. Each
request contains a 64-token shared family prefix and a 16-token unique suffix.
The cache is capped at 12 pages of 16 tokens, so it cannot retain every family's
working set indefinitely.

| Variant | Prefill tokens | Block hits | Hit rate | Cached evictions | Generate time |
|---|---:|---:|---:|---:|---:|
| FIFO, interleaved | 672 | 18 | 60.0% | 30 | 0.846 s |
| LRU, interleaved | 672 | 18 | 60.0% | 30 | 0.800 s |
| LRU, prefix-reordered | 384 | 36 | 92.3% | 12 | 0.790 s |

LRU and FIFO intentionally have the same hit count in this cyclic,
equal-frequency workload: no recency policy can retain three equal working sets
in a cache sized for roughly two. The small timing difference between them is
not treated as an LRU performance claim.

Offline clustering provides the actual measured gain:

- prefill tokens fall by 42.9%;
- block hits double and cached token hits rise from 288 to 576;
- cached evictions fall by 60%;
- hit rate rises from 60.0% to 92.3%;
- generation time falls by 6.7% versus FIFO in this small-model test.

The time reduction is smaller than the token reduction because Python,
FlashInfer planning, sampling, and 12 decode steps remain fixed costs.

## Offline throughput regression

Unique random prompts should not benefit from the cache. The fixed matrix with
default LRU remains within -3.2% to +0.3% of M5:

| Requests | Input | M5 tok/s | M6 tok/s | Change |
|---:|---:|---:|---:|---:|
| 1 | 64 | 199.3 | 198.5 | -0.4% |
| 4 | 64 | 778.3 | 780.3 | +0.3% |
| 8 | 64 | 1,501.2 | 1,481.5 | -1.3% |
| 1 | 256 | 195.7 | 193.1 | -1.3% |
| 4 | 256 | 693.7 | 671.7 | -3.2% |
| 8 | 256 | 1,134.5 | 1,128.4 | -0.5% |

## Why these metrics demonstrate the optimization

Token hits prove work was skipped; eviction counts explain why a miss occurred;
collision counters protect correctness; and exact output comparison verifies
stable API ordering. The bounded-cache ablation also shows the limit of LRU
honestly, while the reordered run establishes when offline knowledge changes
the result materially.

## Known boundaries

- Reordering uses a stable lexicographic prefix key, not a full radix tree or
  semantic similarity model.
- Cache metadata is process-local and not persisted between engine instances.
- Full prompt blocks are still reserved at admission, as documented in M5.

## Evidence for M7

After attention, KV updates, and model dispatch are optimized, the M3 trace and
M5 online runs show fixed overhead in sampling, LM-head work, input metadata
copies, and repeated small compiled regions. M7 should be a measured final
polish rather than another architectural rewrite:

- provide greedy sampling and avoid softmax/exponential allocation when valid;
- keep stochastic sampling as a separate, tested path;
- reduce repeated sampling buffers/casts;
- run final ablations, exact-output gates, and an end-to-end version index.

## Reproduction

```bash
python -m benchmarks.prefix_cache \
  --model /path/to/Qwen3-0.6B --policy fifo \
  --output benchmarks/results/m6_prefix_fifo.json
python -m benchmarks.prefix_cache \
  --model /path/to/Qwen3-0.6B --policy lru \
  --output benchmarks/results/m6_prefix_lru.json
python -m benchmarks.prefix_cache \
  --model /path/to/Qwen3-0.6B --policy lru --reorder \
  --output benchmarks/results/m6_prefix_lru_reordered.json
python -m benchmarks.compare_output_tokens \
  benchmarks/results/m6_prefix_*.json
```
