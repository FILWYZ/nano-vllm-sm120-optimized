import argparse
import random

import torch

from benchmarks.common import environment, write_json
from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure synthetic-model context/concurrency capacity"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-lens", default="512,1024,2048,3072")
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--warmup-repeats", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--backend", default="flashinfer", choices=["sdpa", "flashinfer", "flash"])
    parser.add_argument("--rmsnorm-backend", default="sm120", choices=["torch", "sm120"])
    return parser.parse_args()


def make_prompts(batch, prompt_len, seed):
    rng = random.Random(seed)
    return [[rng.randint(10, 10000) for _ in range(prompt_len)] for _ in range(batch)]


def main():
    args = parse_args()
    prompt_lens = [int(value) for value in args.prompt_lens.split(",")]
    batches = [int(value) for value in args.batches.split(",")]
    max_batch = max(batches)
    if max(prompt_lens) + args.output_len > args.max_model_len:
        raise ValueError("max prompt length plus output length exceeds max_model_len")

    llm = LLM(
        args.model,
        attention_backend=args.backend,
        rmsnorm_backend=args.rmsnorm_backend,
        qk_norm_rope_backend="torch",
        enforce_eager=False,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(2048, args.max_model_len * max_batch),
        max_num_seqs=max_batch,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=16,
    )
    params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
    results = []

    for case_index, (batch, prompt_len) in enumerate(
        (batch, length) for length in prompt_lens for batch in batches
    ):
        for warmup in range(args.warmup_repeats):
            llm.generate(
                make_prompts(batch, prompt_len, args.seed + 10_000 + case_index * 10 + warmup),
                params,
                use_tqdm=False,
            )
        repeats = []
        for repeat in range(args.repeats):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            prompts = make_prompts(batch, prompt_len, args.seed + case_index * 100 + repeat)
            try:
                llm.generate(prompts, params, use_tqdm=False)
                torch.cuda.synchronize()
                repeats.append({
                    "status": "ok",
                    **dict(llm.last_metrics),
                    "current_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "current_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                })
            except Exception as exc:
                torch.cuda.synchronize()
                repeats.append({
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                })
        results.append({
            "requests": batch,
            "input_len": prompt_len,
            "output_len": args.output_len,
            "repeats": repeats,
        })
        successful = [item for item in repeats if item["status"] == "ok"]
        if successful:
            mean_tps = sum(item["output_tokens_per_second"] for item in successful) / len(successful)
            print(f"requests={batch} input={prompt_len} status=ok e2e={mean_tps:.1f}tok/s")
        else:
            print(f"requests={batch} input={prompt_len} status=failed")

    payload = {
        "schema_version": 1,
        "environment": environment(),
        "model_kind": "synthetic_random_initialized_qwen3_architecture",
        "config": {
            "model": args.model,
            "requested_backend": args.backend,
            "resolved_backend": llm.model_runner.config.attention_backend,
            "requested_rmsnorm_backend": args.rmsnorm_backend,
            "resolved_rmsnorm_backend": llm.model_runner.config.rmsnorm_backend,
            "max_model_len": args.max_model_len,
            "max_num_seqs": max_batch,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "num_kvcache_blocks": llm.model_runner.config.num_kvcache_blocks,
            "kvcache_block_size": llm.model_runner.config.kvcache_block_size,
            "kvcache_capacity_tokens": (
                llm.model_runner.config.num_kvcache_blocks
                * llm.model_runner.config.kvcache_block_size
            ),
            "warmup_repeats": args.warmup_repeats,
            "repeats": args.repeats,
        },
        "workloads": results,
    }
    write_json(args.output, payload)
    print(f"wrote={args.output}")
    llm.exit()


if __name__ == "__main__":
    main()
