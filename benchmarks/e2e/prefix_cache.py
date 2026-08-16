import argparse
import random

import torch

from benchmarks.common import environment, write_json
from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Prefix-cache pressure test")
    parser.add_argument("--model", required=True)
    parser.add_argument("--policy", choices=["fifo", "lru"], required=True)
    parser.add_argument("--reorder", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def make_prompts():
    families = []
    for family in range(3):
        rng = random.Random(100 + family)
        families.append([rng.randint(10, 10000) for _ in range(64)])
    prompts = []
    for repeat in range(4):
        for family, prefix in enumerate(families):
            suffix = [20_000 + family * 100 + repeat * 16 + i for i in range(16)]
            prompts.append(prefix + suffix)
    return prompts


def main():
    args = parse_args()
    llm = LLM(
        args.model,
        attention_backend="flashinfer",
        enforce_eager=False,
        max_model_len=128,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        max_prefill_chunk_size=128,
        enable_mixed_batching=True,
        kvcache_block_size=16,
        num_kvcache_blocks=12,
        prefix_cache_policy=args.policy,
        gpu_memory_utilization=0.75,
    )
    prompts = make_prompts()
    torch.manual_seed(2026)
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=1e-6, ignore_eos=True, max_tokens=1),
        use_tqdm=False,
        reorder_by_prefix=args.reorder,
        prefix_reorder_tokens=64,
    )
    metrics = dict(llm.last_metrics)
    cache = metrics["prefix_cache"]
    payload = {
        "schema_version": 1,
        "environment": environment(),
        "policy": args.policy,
        "reorder": args.reorder,
        "workload": {
            "prefix_families": 3,
            "repeats_per_family": 4,
            "shared_prefix_tokens": 64,
            "unique_suffix_tokens": 16,
            "block_size": 16,
            "num_kvcache_blocks": 12,
        },
        "prefill_tokens": metrics["prefill_tokens"],
        "prefill_seconds": metrics["prefill_seconds"],
        "total_seconds": metrics["total_seconds"],
        "output_tokens_per_second": metrics["output_tokens_per_second"],
        "cache": cache,
        "block_hit_rate": (
            cache["block_hits"]
            / (cache["block_hits"] + cache["misses"])
            if cache["block_hits"] + cache["misses"] else 0.0
        ),
        "output_tokens": [output["token_ids"] for output in outputs],
    }
    write_json(args.output, payload)
    print(
        f"policy={args.policy} reorder={args.reorder} "
        f"prefill_tokens={metrics['prefill_tokens']} "
        f"hits={cache['block_hits']} evictions={cache['cached_evictions']} "
        f"total={metrics['total_seconds']:.4f}s"
    )
    llm.exit()


if __name__ == "__main__":
    main()
