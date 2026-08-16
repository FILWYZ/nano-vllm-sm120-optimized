# M3: Bucketed Decode CUDA Graphs

## Why this optimization

After M2 removed KV-write synchronization, an 8-request x 16-token trace still
contained 2,720 TorchDynamo cache lookups/prologues and 1,808 visible matrix
multiply launches. Qwen3-0.6B has short kernels, so repeatedly dispatching every
layer from Python limits decode more than attention arithmetic.

CUDA Graphs record a stable decode execution once and replay it with one host
launch. They are appropriate for decode, where each active request contributes
one query token; prefill remains dynamic and eager.

## Implementation

- Enabled CUDA Graphs for FlashInfer while retaining eager SDPA as the readable
  compatibility backend.
- Captured decode graphs for bounded batch buckets. Bucket candidates never
  exceed `max_num_seqs`, and a non-power-of-two maximum is added explicitly.
- Gave every FlashInfer bucket its own wrapper and stable GPU buffers for page
  indptr, page indices, and last-page lengths.
- Kept FlashInfer `plan()` outside graph capture/replay. It refreshes the stable
  metadata buffers before replay as sequence lengths and block tables change.
- Padded unused bucket rows with `context_len=1`, block 0, and `slot=-1`.
  Padded computation is discarded and the Triton KV append kernel cannot write
  it to cache.
- Cleared all shared graph input buffers before copying a smaller live batch,
  preventing stale requests or stale page-table columns from a prior replay.
- Added an `--enforce-eager` switch to benchmark tools for controlled ablation.

The model weights, scheduler semantics, sampling algorithm, and KV dtype remain
unchanged.

## Correctness evidence

The existing seven SDPA/FlashInfer/KV tests pass. In addition,
`benchmarks/analysis/validate_graph.py` compares two full model runs:

- 3 real requests executed in a padded graph bucket;
- 255 prompt tokens plus 16 output tokens, forcing allocation and access across
  the 256-token KV-page boundary;
- identical prompt tokens, sampling seed, and temperature;
- eager FlashInfer versus graph-replayed FlashInfer.

All 48 completion tokens matched exactly (`48/48`). This catches stale page
metadata, bucket-padding writes, and graph/eager numerical divergence at the
user-visible output level.

## Performance evidence

Same Qwen3-0.6B fixed matrix, one warmup and two measured repeats:

| Requests | Input | M2 eager tok/s | M3 graph tok/s | Gain |
|---:|---:|---:|---:|---:|
| 1 | 64 | 60.7 | 194.1 | +219.8% |
| 4 | 64 | 222.8 | 783.5 | +251.7% |
| 8 | 64 | 445.1 | 1,518.5 | +241.2% |
| 1 | 256 | 59.0 | 192.7 | +226.6% |
| 4 | 256 | 214.0 | 689.0 | +222.0% |
| 8 | 256 | 408.7 | 1,129.9 | +176.4% |

Profile comparison for 8 requests x 16 output tokens:

| Signal | M2 eager | M3 graph | Change |
|---|---:|---:|---:|
| profiler self CPU | 280 ms | 103 ms | -63.1% |
| TorchDynamo cache lookups | 2,720 | 185 | -93.2% |
| visible `aten::mm` calls | 1,808 | 128 | -92.9% |
| visible `nonzero` calls | 15 | 15 | graph-external planning only |

Largest-workload peak allocation is 4.758 GiB and peak reservation is
4.861 GiB. This is a small increase over M2 because captured graphs retain
private pools and each bucket owns stable FlashInfer metadata.

CUPTI CUDA activities remain unavailable in this WSL setup. The end-to-end
timers include synchronization through token sampling and are therefore the
authoritative graph-speed measurement.

## Why these metrics demonstrate the optimization

The dispatch operations targeted by graph capture fall by more than 92%, CPU
profile time falls by 63%, and all fixed workloads improve substantially. Exact
token parity across a page boundary shows the speedup did not come from stale or
truncated KV state.

## Evidence for M4

Graph replay shifts the visible host bottleneck to metadata copies: `aten::copy_`
accounts for 75 ms in the profiler, mostly FlashInfer planning and stable-buffer
updates. The current 256-token KV page is also inherited from the teaching
baseline rather than selected for this GPU. It coarsens prefix-cache reuse and
wastes up to 255 token slots per active sequence.

M4 should benchmark smaller supported page sizes, measure KV capacity and
fragmentation, and choose a local default. Smaller pages increase page-table
metadata and planning copies, so the choice must be empirical rather than
assuming the smallest page is fastest.

## Reproduction

```bash
source .venv/bin/activate
python -m benchmarks.validate_graph --model /path/to/Qwen3-0.6B
python -m benchmarks.end_to_end \
  --model /path/to/Qwen3-0.6B --backend flashinfer \
  --output benchmarks/results/m3_cudagraph_end_to_end.json
python -m benchmarks.profile_decode \
  --model /path/to/Qwen3-0.6B \
  --summary benchmarks/results/m3_profile.txt
```
