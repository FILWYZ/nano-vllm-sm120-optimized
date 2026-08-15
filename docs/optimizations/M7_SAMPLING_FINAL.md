# M7: Sampling Fast Paths and Final Ablation

## Why this optimization

After M1-M6, decode model execution is graph-replayed and attention is no
longer the dominant Python path. The original sampler still always converted
the full vocabulary logits to FP32, computed softmax, allocated and filled an
exponential-noise tensor, divided probabilities, and reduced with argmax. It
also prohibited temperature zero, so deterministic inference paid the full
random-sampling cost.

For Qwen3's 151,936-token vocabulary this work is measurable, especially at
larger batches.

## Implementation

- `temperature=0` now explicitly selects greedy decoding; negative values are
  rejected.
- Added a compiled greedy path that directly reduces logits with `argmax`,
  avoiding FP32 logits, softmax, exponential RNG, division, and noise storage.
- Retained the stochastic Gumbel/exponential path for positive temperatures.
- Cached FP32 stochastic noise buffers by device and logits shape, eliminating
  one full-vocabulary allocation per sampling step after warmup.
- Supports mixed batches containing greedy and stochastic rows: stochastic work
  is performed for the batch and greedy rows are overwritten with exact argmax.
- ModelRunner derives `all_greedy` and `has_greedy` from CPU request metadata,
  avoiding a GPU scalar synchronization to choose the fast path.
- Added sampler CUDA-event microbenchmarks and temperature selection to the
  fixed end-to-end benchmark.

This is deliberately not a speculative custom fused softmax/RNG kernel. The
compiled implementation is short, readable, and testable, preserving the
project's teaching goal.

## Correctness evidence

Twenty-one tests pass. New sampling tests verify:

1. greedy output equals `logits.argmax` exactly;
2. zero temperature is accepted and negative temperature rejected;
3. a greedy row remains exact inside a mixed sampling batch;
4. stochastic noise storage retains the same data pointer across calls.

The full greedy graph/eager validator also uses three requests in a padded
bucket with 255-token prompts and 16-token KV pages. All 48 completion tokens
match exactly across eager and CUDA Graph execution.

## Sampler microbenchmark

CUDA-event latency at vocabulary size 151,936, after 10 warmups and over 100
measured iterations:

| Batch | Stochastic p50 | Greedy p50 | Reduction | Greedy p95 |
|---:|---:|---:|---:|---:|
| 1 | 0.1684 ms | 0.0595 ms | -64.7% | 0.0782 ms |
| 8 | 0.1821 ms | 0.0480 ms | -73.6% | 0.0538 ms |

The batch-8 greedy timing being slightly lower than batch 1 reflects kernel
selection and measurement noise; the conclusion is the large separation from
the stochastic path, not cross-batch monotonicity.

## End-to-end greedy benefit

Both modes use the final M7 code and differ only in temperature:

| Requests | Input | Stochastic tok/s | Greedy tok/s | Gain |
|---:|---:|---:|---:|---:|
| 1 | 64 | 198.3 | 198.1 | -0.1% |
| 4 | 64 | 745.4 | 799.2 | +7.2% |
| 8 | 64 | 1,440.2 | 1,543.3 | +7.2% |
| 1 | 256 | 188.6 | 198.2 | +5.1% |
| 4 | 256 | 683.9 | 697.5 | +2.0% |
| 8 | 256 | 1,107.4 | 1,139.5 | +2.9% |

At batch 1/input 64, model and host overhead hide the 0.11 ms sampler saving.
At larger batches, avoiding full-vocabulary stochastic work produces a clear
end-to-end gain.

## Stochastic compatibility regression

Relative to M6, M7 stochastic output throughput changes by -4.5% to +1.8%
across the fixed matrix, inside the 5% steady-state tolerance. Buffer reuse is
primarily an allocation-stability improvement; the benchmark does not claim it
makes the stochastic kernel materially faster.

## Why these metrics demonstrate the optimization

The microbenchmark isolates exactly the removed full-vocabulary operations,
while the mode-controlled full-model matrix shows when that saving becomes
user-visible. Exact argmax tests and full graph/eager token parity prove that
the shortcut implements deterministic decoding rather than approximating the
stochastic path.

## Further optimization directions

M7 closes this local M0-M7 baseline rather than claiming a universal endpoint.
The next evidence-driven experiments should be separate branches:

- CUDA/Triton top-k/top-p kernels with per-request RNG state;
- lazy KV allocation for chunked prompts and admission control;
- FlashInfer planning metadata built directly from CPU scheduler arrays to
  reduce the remaining copy/synchronization cost;
- FP8 KV cache with accuracy/perplexity validation on SM120;
- speculative decoding with acceptance-rate and wall-time measurements;
- tensor parallelism tests when a second compatible GPU is available;
- broader model and context-length matrices before calling any result general.

## Reproduction

```bash
python -m benchmarks.sampler_micro \
  --output benchmarks/results/m7_sampler_micro.json
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B --temperature 0 \
  --output benchmarks/results/m7_greedy_end_to_end.json
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B --temperature 0.6 \
  --output benchmarks/results/m7_stochastic_end_to_end.json
python -m benchmarks.validate_graph \
  --model /path/to/Qwen3-0.6B --temperature 0 \
  --output benchmarks/results/m7_greedy_graph_parity.json
```
