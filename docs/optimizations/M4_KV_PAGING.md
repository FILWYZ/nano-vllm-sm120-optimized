# M4: Empirical KV Page Size and Memory Layout

## Why this optimization

The original 256-token block was inherited from the teaching project and was
also enforced by an assertion. A large page keeps page tables small but wastes
the unused tail of every live request, only caches prefixes at 256-token
boundaries, and delays freeing/reusing memory at fine granularity.

FlashInfer supports smaller power-of-two pages, but more page indices can raise
planning and attention overhead. M4 therefore selects a page size from measured
throughput and capacity rather than assuming smaller is always better.

## Implementation

- Generalized the configuration and scheduler to 16, 32, 64, 128, and 256
  token power-of-two pages.
- Retained the 256-token restriction when explicitly selecting the original
  FlashAttention backend.
- Made the local FlashInfer/SDPA default 16 tokens and synchronized the
  `Sequence` page arithmetic from `Config` at engine construction.
- Added page size, physical block count, and raw KV token capacity to benchmark
  JSON.
- Added an isolated-process sweep so CUDA allocator peaks and graph pools from
  one candidate cannot contaminate the next.
- Added CPU regressions covering allocation, append boundaries, hash reuse, and
  cached-token accounting at every supported size.

## Selection rule

Before running the sweep, the rule was fixed as:

> Select the smallest page whose geometric-mean output throughput is within 5%
> of the fastest candidate; use capacity as a secondary check.

This prevents choosing a statistically small throughput winner while ignoring
fragmentation and prefix-cache granularity.

## Sweep results

One shape warmup and one measured repeat per workload, with every candidate in
a fresh process:

| Page | Geomean output tok/s | Raw KV tokens | Mean tail waste/seq | Prefix granularity |
|---:|---:|---:|---:|---:|
| 16 | 568.30 | 31,648 | 7.5 | 16 |
| 32 | 570.25 | 31,648 | 15.5 | 32 |
| 64 | 572.86 | 31,616 | 31.5 | 64 |
| 128 | 569.38 | 31,616 | 63.5 | 128 |
| 256 | 571.92 | 31,488 | 127.5 | 256 |

Page 64 is the raw throughput winner, but page 16 is only 0.8% slower and is
therefore selected by the declared rule. Relative to 256 it:

- reduces expected live tail waste by 94.1% (127.5 to 7.5 tokens/sequence);
- makes prefix reuse granularity 16x finer;
- raises raw token capacity by 160 tokens (+0.51%);
- raises estimated usable capacity with eight active sequences from 30,468 to
  31,588 tokens (+3.68%) after expected tail waste.

## Final two-repeat result

The selected page was rerun with M3's one-warmup/two-repeat protocol:

| Requests | Input | M3 page-256 tok/s | M4 page-16 tok/s | Change |
|---:|---:|---:|---:|---:|
| 1 | 64 | 194.1 | 197.8 | +1.9% |
| 4 | 64 | 783.5 | 767.1 | -2.1% |
| 8 | 64 | 1,518.5 | 1,470.5 | -3.2% |
| 1 | 256 | 192.7 | 184.9 | -4.0% |
| 4 | 256 | 689.0 | 663.5 | -3.7% |
| 8 | 256 | 1,129.9 | 1,116.9 | -1.2% |

Every workload remains within the 5% throughput budget. M4 is consequently a
capacity/cache-granularity optimization, not a claim of higher raw kernel
speed.

## Correctness evidence

Nine tests pass: the prior seven GPU attention/KV tests plus variable-page
allocation and boundary-append tests. The full graph/eager validation was also
rerun at page 16 with 255-token prompts. It crossed sixteen page boundaries per
request and all 48 completion tokens matched exactly.

## Why these metrics demonstrate the optimization

Raw block count alone is misleading because smaller blocks contain fewer
tokens. The report therefore compares raw token capacity, expected unusable tail
tokens per live request, prefix granularity, and end-to-end throughput. Page 16
materially improves the first three while respecting the predeclared 5% speed
budget and preserving exact outputs.

## Evidence for M5

Smaller pages let long prompts allocate incrementally, but the scheduler still
has two limitations:

- one large first request may consume the entire prefill token budget;
- any schedulable waiting prefill batch prevents running decode in that step,
  creating decode latency spikes under request arrivals.

M5 should make chunk size explicit, expose scheduler counters, and improve
continuous batching/fairness. Because the model context currently represents a
single phase per step, mixed prefill+decode must either use a unified paged
prefill representation or a bounded phase policy with measured latency.

## Reproduction

```bash
python -m benchmarks.block_size_sweep --model /path/to/Qwen3-0.6B
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B --backend flashinfer --block-size 16 \
  --output benchmarks/results/m4_block16_end_to_end.json
python -m benchmarks.validate_graph \
  --model /path/to/Qwen3-0.6B \
  --output benchmarks/results/m4_block16_graph_parity.json
```
