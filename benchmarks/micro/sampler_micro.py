import argparse

import torch

from benchmarks.common import environment, percentile, write_json
from nanovllm.layers.sampler import Sampler


def parse_args():
    parser = argparse.ArgumentParser(description="Sampler CUDA-event benchmark")
    parser.add_argument(
        "--output", default="benchmarks/results/m7_sampler_micro.json"
    )
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--iterations", type=int, default=100)
    return parser.parse_args()


def measure(sampler, logits, temperatures, greedy, iterations):
    for _ in range(10):
        sampler(logits, temperatures, greedy, greedy)
    torch.cuda.synchronize()
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        sampler(logits, temperatures, greedy, greedy)
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end))
    return {
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "mean_ms": sum(latencies) / len(latencies),
    }


def main():
    args = parse_args()
    torch.manual_seed(2026)
    sampler = Sampler().cuda()
    results = []
    for batch_size in (1, 8):
        logits = torch.randn(
            batch_size, args.vocab_size, device="cuda", dtype=torch.float16
        )
        for mode, temperature in (("greedy", 0.0), ("stochastic", 0.6)):
            temperatures = torch.full(
                (batch_size,), temperature, device="cuda", dtype=torch.float32
            )
            metrics = measure(
                sampler, logits, temperatures, mode == "greedy", args.iterations
            )
            result = {
                "mode": mode,
                "batch_size": batch_size,
                "vocab_size": args.vocab_size,
                **metrics,
            }
            results.append(result)
            print(
                f"mode={mode} batch={batch_size} "
                f"p50={metrics['p50_ms']:.4f}ms p95={metrics['p95_ms']:.4f}ms"
            )
    payload = {
        "schema_version": 1,
        "environment": environment(),
        "config": {
            "vocab_size": args.vocab_size,
            "iterations": args.iterations,
            "warmup_iterations": 10,
            "timing": "CUDA events with per-iteration synchronization",
        },
        "results": results,
        "noise_buffer_shapes": [
            list(key[1]) for key in sampler.noise_buffers
        ],
    }
    write_json(args.output, payload)
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
