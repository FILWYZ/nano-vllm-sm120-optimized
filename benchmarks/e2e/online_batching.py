import argparse
import random
from time import perf_counter

import torch

from benchmarks.common import environment, percentile, write_json
from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure decode jitter when a long prompt arrives"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy", choices=["mixed", "prefill-first"], required=True
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def tokens(seed, length):
    rng = random.Random(seed)
    return [rng.randint(10, 10000) for _ in range(length)]


def warm_online_shapes(llm):
    short_params = SamplingParams(
        temperature=0.1, ignore_eos=True, max_tokens=8
    )
    long_params = SamplingParams(
        temperature=0.1, ignore_eos=True, max_tokens=1
    )
    llm.add_request(tokens(10, 64), short_params)
    short = llm.scheduler.waiting[-1]
    while short.num_completion_tokens < 2:
        llm.step()
    llm.add_request(tokens(20, 512), long_params)
    long_seq = llm.scheduler.waiting[-1]
    while not short.is_finished or not long_seq.is_finished:
        llm.step()


def main():
    args = parse_args()
    mixed = args.policy == "mixed"
    llm = LLM(
        args.model,
        attention_backend="flashinfer",
        enforce_eager=False,
        max_model_len=768,
        max_num_batched_tokens=512,
        max_num_seqs=4,
        max_prefill_chunk_size=128,
        enable_mixed_batching=mixed,
        kvcache_block_size=16,
        gpu_memory_utilization=0.75,
    )
    warm_online_shapes(llm)
    scheduler_before = dict(llm.scheduler.metrics)
    torch.manual_seed(2026)
    short_params = SamplingParams(
        temperature=0.1, ignore_eos=True, max_tokens=24
    )
    long_params = SamplingParams(
        temperature=0.1, ignore_eos=True, max_tokens=1
    )
    llm.add_request(tokens(1, 64), short_params)
    short = llm.scheduler.waiting[-1]

    completion_times = []
    previous_count = 0
    while short.num_completion_tokens < 4:
        llm.step()
        if short.num_completion_tokens > previous_count:
            completion_times.append(perf_counter())
            previous_count = short.num_completion_tokens

    arrival_time = perf_counter()
    llm.add_request(tokens(2, 512), long_params)
    long_seq = llm.scheduler.waiting[-1]
    step_latencies = []
    while not short.is_finished:
        started = perf_counter()
        llm.step()
        step_latencies.append(perf_counter() - started)
        if short.num_completion_tokens > previous_count:
            completion_times.append(perf_counter())
            previous_count = short.num_completion_tokens

    gaps = [
        right - left
        for left, right in zip(completion_times, completion_times[1:])
        if right >= arrival_time
    ]
    long_completion_tokens = long_seq.num_completion_tokens
    scheduler_metrics = {
        key: value - scheduler_before[key]
        for key, value in llm.scheduler.metrics.items()
    }
    payload = {
        "schema_version": 1,
        "environment": environment(),
        "policy": args.policy,
        "workload": {
            "short_prompt_tokens": 64,
            "short_output_tokens": 24,
            "long_prompt_tokens": 512,
            "long_output_tokens": 1,
            "arrival_after_short_output_tokens": 4,
            "prefill_chunk_size": 128,
        },
        "short_decode": {
            "observed_gaps": len(gaps),
            "mean_inter_token_gap_seconds": sum(gaps) / len(gaps),
            "p95_inter_token_gap_seconds": percentile(gaps, 0.95),
            "max_inter_token_gap_seconds": max(gaps),
        },
        "steps_after_arrival": {
            "count": len(step_latencies),
            "mean_seconds": sum(step_latencies) / len(step_latencies),
            "p95_seconds": percentile(step_latencies, 0.95),
        },
        "long_request_finished": long_seq.is_finished,
        "long_completion_tokens": long_completion_tokens,
        "scheduler": scheduler_metrics,
    }
    write_json(args.output, payload)
    print(
        f"policy={args.policy} mean_gap={payload['short_decode']['mean_inter_token_gap_seconds']:.6f}s "
        f"p95_gap={payload['short_decode']['p95_inter_token_gap_seconds']:.6f}s "
        f"max_gap={payload['short_decode']['max_inter_token_gap_seconds']:.6f}s"
    )
    llm.exit()


if __name__ == "__main__":
    main()
