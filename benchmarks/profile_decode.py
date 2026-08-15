import argparse
import random
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Export a nano-vLLM CPU/CUDA trace")
    parser.add_argument("--model", required=True)
    parser.add_argument("--trace", default="benchmarks/results/m0_decode_trace.json")
    parser.add_argument("--summary", default="benchmarks/results/m0_profile.txt")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=16)
    return parser.parse_args()


def main():
    args = parse_args()
    llm = LLM(
        args.model,
        attention_backend="auto",
        enforce_eager=True,
        max_model_len=512,
        max_num_batched_tokens=2048,
        max_num_seqs=args.requests,
        gpu_memory_utilization=0.75,
    )
    rng = random.Random(2026)
    prompts = [
        [rng.randint(10, 10000) for _ in range(args.input_len)]
        for _ in range(args.requests)
    ]
    params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=args.output_len)
    llm.generate(prompts, params, use_tqdm=False)
    trace_path = Path(args.trace)
    summary_path = Path(args.summary)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        llm.generate(prompts, params, use_tqdm=False)
    profiler.export_chrome_trace(str(trace_path))
    cpu_summary = profiler.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=40
    )
    cuda_summary = profiler.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=40
    )
    summary = "CPU HOTSPOTS\n" + cpu_summary + "\n\nCUDA HOTSPOTS\n" + cuda_summary
    summary_path.write_text(summary + "\n")
    print(summary)
    print(f"trace={trace_path}")
    print(f"summary={summary_path}")
    llm.exit()


if __name__ == "__main__":
    main()
