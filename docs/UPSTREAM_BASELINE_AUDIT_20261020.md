# Upstream baseline audit — 2026-10-20

## Conclusion

This run compares the optimized nano-vLLM checkout with a runnable, minimally adapted SM120 compatibility baseline. It is a same-machine, same-model, fixed-workload system comparison. It is not a direct run of unmodified upstream nano-vLLM, it does not establish token-for-token equivalence, and it does not isolate the contribution of any one optimization.

Across three interleaved independent-process rounds, median throughput was 216.54 tok/s for the compatibility baseline and 1,765.69 tok/s for the optimized configuration (8.15×). Median wall time fell from 618.67 s to 75.87 s (87.74% lower). Median peak PyTorch allocated memory rose from 4.76 GiB to 5.09 GiB (+6.93%). Each run generated 133,966 output tokens.

## Comparison subjects

| Subject | Source | Status and valid interpretation |
|---|---|---|
| Unmodified upstream | [GeeeekExplorer/nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm), upstream commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8` | Source snapshot is retained locally. Importing the original source in the current environment fails because `flash_attn` is not installed; this subject has no local performance result. We do not substitute the upstream README's RTX 4070 Laptop result for a local RTX 5060 measurement. |
| Minimal SM120 compatibility baseline | Tag `baseline-blackwell-v0`, commit `7aa1f251cf1e041ded3371bab41f90e6736111ac`, derived from the upstream snapshot with the minimum SDPA/SM120 compatibility changes | Runs locally with the tested PyTorch/CUDA stack. This is the measured baseline below; it is not unmodified upstream. |
| Optimized project | Integrated candidate commit `11d0aa58a2c68d7132dd0d919e1029afea6a6aa2` | Measured with the optimized M8 offline configuration and its engine/scheduler features. |

The [upstream README](https://github.com/GeeeekExplorer/nano-vllm) documents results on RTX 4070 Laptop. Those public numbers are not comparable to this RTX 5060 Laptop run because the GPU and potentially the software stack differ.

## Matched workload and environment

- RTX 5060 Laptop, Blackwell SM120, 8 GiB, WSL2 Ubuntu.
- Python 3.12.3; PyTorch 2.14.1+cu130; CUDA Runtime 13.0; Triton 3.8.0; FlashInfer 0.6.6; NVIDIA-SMI 610.57.01.
- Qwen3-0.6B, BF16 weights, model revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- 256 requests, random prompt/output lengths from 100 to 1024 tokens, seed 0, temperature 0.6, EOS ignored, max model length 4096, max sequences 256, max batched tokens 16384, GPU memory utilization 0.75.
- Each run produced 142,827 input tokens and exactly 133,966 output tokens. The three rounds used independent processes, interleaved baseline-first / optimized-first / baseline-first.

The request lengths, sampling parameters and model are held constant. The implementation settings intentionally differ as part of the system comparison:

| Setting | Compatibility baseline | Optimized configuration |
|---|---|---|
| Attention | PyTorch SDPA | FlashInfer Paged Attention |
| KV page | 256 tokens | 64 tokens |
| RMSNorm | PyTorch | SM120 extension |
| Q/K Norm + RoPE | PyTorch | PyTorch (same path) |
| CUDA Graph | Eager execution | Enabled |
| Output KV admission | Baseline behavior | Output KV reservation enabled |

Therefore the result measures the cumulative engine configuration, including the attention backend, KV page size, execution mode and scheduling/admission path. It does not assign the end-to-end gain to a single kernel. The output SHA-256 was stable within each implementation across all three rounds, but differed between implementations. Equal output length does not demonstrate equivalent generated content or quality; no token/logit parity claim is made.

## Results

| Round | Compatibility tok/s | Optimized tok/s | Paired ratio | Compatibility time | Optimized time | Peak allocated memory, compatibility → optimized |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 218.36 | 1,749.87 | 8.014× | 613.52 s | 76.56 s | 4.76 → 5.09 GiB |
| 2 | 215.64 | 1,765.69 | 8.188× | 621.24 s | 75.87 s | 4.76 → 5.09 GiB |
| 3 | 216.54 | 1,768.32 | 8.166× | 618.67 s | 75.76 s | 4.76 → 5.09 GiB |
| **Median** | **216.54** | **1,765.69** | **8.154×** | **618.67 s** | **75.87 s** | **4.76 → 5.09 GiB** |

The paired ratios above are round-by-round ratios. The headline 8.154× is the ratio of the two throughput medians. The median memory change is +6.93% (about +0.33 GiB). These are PyTorch peak allocated bytes; they do not include all driver or system memory.

## Artifacts and reproduction

Raw per-run JSON and logs, including resolved configuration, environment, output counts, output hashes, elapsed time and peak memory, are in [`benchmarks/results/upstream_compat_ab_20261020/`](../benchmarks/results/upstream_compat_ab_20261020/). The runner is [`run_upstream_compat_ab.py`](../benchmarks/e2e/run_upstream_compat_ab.py); an example invocation is:

```bash
python -m benchmarks.e2e.run_upstream_compat_ab \
  --compat-repo /path/to/clean-baseline-blackwell-v0 \
  --optimized-repo /path/to/nano-vllm-sm120-optimized \
  --model /path/to/Qwen3-0.6B \
  --output-dir benchmarks/results/upstream_compat_ab_YYYYMMDD \
  --runs 3 --max-model-len 4096 --max-num-seqs 256 \
  --max-num-batched-tokens 16384 --gpu-memory-utilization 0.75
```

## Historical numbers

The earlier 7.66× result and 8.23× six-workload geometric mean remain historical results from their recorded commits and conditions. This fresh 2026-10-20 three-round run is a separate measurement with the integrated candidate and should be cited as 8.15× only with the configuration and limitations above. It does not replace official vLLM comparisons, which are separate engine-level comparisons with a different software stack.
