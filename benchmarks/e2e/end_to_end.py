import argparse
import random

import torch

from benchmarks.common import environment, write_json
from nanovllm import LLM, SamplingParams


DEFAULT_MATRIX = [
    {"requests": 1, "input_len": 64, "output_len": 32},
    {"requests": 4, "input_len": 64, "output_len": 32},
    {"requests": 8, "input_len": 64, "output_len": 32},
    {"requests": 1, "input_len": 256, "output_len": 32},
    {"requests": 4, "input_len": 256, "output_len": 32},
    {"requests": 8, "input_len": 256, "output_len": 32},
]


def parse_args():
    parser = argparse.ArgumentParser(description="nano-vLLM end-to-end benchmark matrix")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="benchmarks/results/latest.json")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "sdpa", "flashinfer", "flash"])
    parser.add_argument("--rmsnorm-backend", default="torch",
                        choices=["auto", "torch", "sm120"])
    parser.add_argument("--qk-norm-rope-backend", default="torch",
                        choices=["auto", "torch", "sm120"])
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--temperature", type=float, default=0.6)
    return parser.parse_args()


def make_prompts(requests: int, input_len: int, seed: int):
    rng = random.Random(seed)
    return [[rng.randint(10, 10000) for _ in range(input_len)] for _ in range(requests)]


def main():
    args = parse_args()
    max_requests = max(case["requests"] for case in DEFAULT_MATRIX)
    llm = LLM(
        args.model,
        attention_backend=args.backend,
        rmsnorm_backend=args.rmsnorm_backend,
        qk_norm_rope_backend=args.qk_norm_rope_backend,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(2048, args.max_model_len * max_requests),
        max_num_seqs=max_requests,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=args.block_size,
    )
    results = []
    for case_index, case in enumerate(DEFAULT_MATRIX):
        params = SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=case["output_len"],
        )
        for warmup in range(args.warmup_repeats):
            prompts = make_prompts(
                case["requests"], case["input_len"],
                args.seed + case_index * 100 + 10_000 + warmup,
            )
            llm.generate(prompts, params, use_tqdm=False)
        repeats = []
        for repeat in range(args.repeats):
            prompts = make_prompts(
                case["requests"], case["input_len"],
                args.seed + case_index * 100 + repeat,
            )
            torch.manual_seed(args.seed + repeat)
            torch.cuda.reset_peak_memory_stats()
            llm.generate(prompts, params, use_tqdm=False)
            metrics = dict(llm.last_metrics)
            metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            repeats.append(metrics)
        aggregate = {
            key: sum(item[key] for item in repeats) / len(repeats)
            for key in (
                "prefill_tokens_per_second", "decode_tokens_per_second",
                "output_tokens_per_second", "batch_ttft_seconds",
                "mean_tpot_seconds", "peak_allocated_gib", "peak_reserved_gib",
            )
        }
        result = {**case, "repeats": repeats, "mean": aggregate}
        results.append(result)
        print(
            f"requests={case['requests']} input={case['input_len']} "
            f"output={case['output_len']} prefill={aggregate['prefill_tokens_per_second']:.1f}tok/s "
            f"decode={aggregate['decode_tokens_per_second']:.1f}tok/s "
            f"e2e={aggregate['output_tokens_per_second']:.1f}tok/s"
        )
    payload = {
        "schema_version": 1,
        "environment": environment(),
        "config": {
            "model": args.model,
            "requested_backend": args.backend,
            "resolved_backend": llm.model_runner.config.attention_backend,
            "requested_rmsnorm_backend": args.rmsnorm_backend,
            "resolved_rmsnorm_backend": llm.model_runner.config.rmsnorm_backend,
            "requested_qk_norm_rope_backend": args.qk_norm_rope_backend,
            "resolved_qk_norm_rope_backend": (
                llm.model_runner.config.qk_norm_rope_backend
            ),
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "warmup_repeats": args.warmup_repeats,
            "repeats": args.repeats,
            "enforce_eager": args.enforce_eager,
            "kvcache_block_size": args.block_size,
            "num_kvcache_blocks": llm.model_runner.config.num_kvcache_blocks,
            "kvcache_capacity_tokens": (
                llm.model_runner.config.num_kvcache_blocks * args.block_size
            ),
            "temperature": args.temperature,
        },
        "workloads": results,
    }
    write_json(args.output, payload)
    print(f"wrote={args.output}")
    llm.exit()


if __name__ == "__main__":
    main()
