import argparse

import torch

from benchmarks.common import environment, percentile, write_json
from nanovllm.layers.attention import Attention, set_attention_backend
from nanovllm.utils.context import reset_context, set_context


def parse_args():
    parser = argparse.ArgumentParser(description="Attention-only CUDA microbenchmark")
    parser.add_argument("--output", default="benchmarks/results/attention_latest.json")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--backend", default="sdpa",
                        choices=["sdpa", "flashinfer"])
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def time_cuda(fn, warmup: int, iterations: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return {
        "mean_ms": sum(samples) / len(samples),
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
    }


def prefill_case(batch_size: int, seq_len: int, warmup: int, iterations: int):
    heads, kv_heads, dim = 4, 2, 128
    total = batch_size * seq_len
    q = torch.randn(total, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(total, kv_heads, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    cu = torch.arange(0, total + 1, seq_len, device="cuda", dtype=torch.int32)
    slots = torch.empty(0, device="cuda", dtype=torch.int32)
    attn = Attention(heads, dim, dim ** -0.5, kv_heads).cuda()
    set_context(True, cu, cu, seq_len, seq_len, slots)
    timing = time_cuda(lambda: attn(q, k, v), warmup, iterations)
    reset_context()
    return timing


def decode_case(batch_size: int, seq_len: int, warmup: int, iterations: int):
    heads, kv_heads, dim, block_size = 4, 2, 128, 256
    blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_seq
    attn = Attention(heads, dim, dim ** -0.5, kv_heads).cuda()
    attn.k_cache = torch.randn(
        num_blocks, block_size, kv_heads, dim, device="cuda", dtype=torch.float16
    )
    attn.v_cache = torch.randn_like(attn.k_cache)
    q = torch.randn(batch_size, heads, dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch_size, kv_heads, dim, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    block_tables = torch.arange(num_blocks, device="cuda", dtype=torch.int32).view(
        batch_size, blocks_per_seq
    )
    slots = block_tables[:, -1] * block_size + (seq_len - 1) % block_size
    lengths = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    set_context(False, slot_mapping=slots, context_lens=lengths, block_tables=block_tables)
    timing = time_cuda(lambda: attn(q, k, v), warmup, iterations)
    reset_context()
    return timing


def main():
    args = parse_args()
    set_attention_backend(args.backend)
    cases = []
    for phase in ("prefill", "decode"):
        for batch_size in (1, 4, 8):
            for seq_len in (128, 512):
                fn = prefill_case if phase == "prefill" else decode_case
                timing = fn(batch_size, seq_len, args.warmup, args.iterations)
                result = {
                    "phase": phase,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    **timing,
                }
                cases.append(result)
                print(
                    f"phase={phase} batch={batch_size} seq={seq_len} "
                    f"p50={timing['p50_ms']:.3f}ms p95={timing['p95_ms']:.3f}ms"
                )
    write_json(args.output, {
        "schema_version": 1,
        "environment": environment(),
        "backend": args.backend,
        "cases": cases,
    })
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
