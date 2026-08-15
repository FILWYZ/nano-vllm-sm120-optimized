import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one isolated nano-vLLM upstream-vs-M7 measurement."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--suite", choices=["fixed", "github"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--block-size", type=int)
    return parser.parse_args()


def git_commit():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def make_fixed(args):
    rng = random.Random(args.seed)
    prompts = [
        [rng.randint(10, 10_000) for _ in range(args.input_len)]
        for _ in range(args.requests)
    ]
    params = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    warm_rng = random.Random(args.seed + 1_000_000)
    warm_prompts = [
        [warm_rng.randint(10, 10_000) for _ in range(args.input_len)]
        for _ in range(args.requests)
    ]
    return prompts, params, warm_prompts, params


def make_github(args):
    # Exact request distribution from upstream commit bb823b3 bench.py.
    rng = random.Random(0)
    prompts = [
        [rng.randint(0, 10_000) for _ in range(rng.randint(100, 1024))]
        for _ in range(256)
    ]
    params = [
        SamplingParams(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=rng.randint(100, 1024),
        )
        for _ in range(256)
    ]
    return prompts, params, ["Benchmark: "], SamplingParams()


def token_digest(outputs):
    digest = hashlib.sha256()
    for output in outputs:
        for token in output["token_ids"]:
            digest.update(int(token).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.suite == "fixed":
        prompts, params, warm_prompts, warm_params = make_fixed(args)
        default_max_len = max(512, args.input_len + args.output_len)
    else:
        prompts, params, warm_prompts, warm_params = make_github(args)
        default_max_len = 4096

    kwargs = {
        "enforce_eager": False,
        "max_model_len": args.max_model_len or default_max_len,
    }
    if args.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = args.gpu_memory_utilization
    if args.block_size is not None:
        kwargs["kvcache_block_size"] = args.block_size

    init_started = time.perf_counter()
    llm = LLM(args.model, **kwargs)
    init_seconds = time.perf_counter() - init_started
    try:
        torch.manual_seed(args.seed + 10_000)
        llm.generate(warm_prompts, warm_params, use_tqdm=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        torch.manual_seed(args.seed)
        started = time.perf_counter()
        outputs = llm.generate(prompts, params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        actual_output_tokens = sum(len(item["token_ids"]) for item in outputs)
        expected_output_tokens = (
            sum(item.max_tokens for item in params)
            if isinstance(params, list)
            else len(prompts) * params.max_tokens
        )
        props = torch.cuda.get_device_properties(0)
        payload = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "variant": args.variant,
            "suite": args.suite,
            "git_commit": git_commit(),
            "working_directory": os.getcwd(),
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "device": props.name,
                "compute_capability": list(torch.cuda.get_device_capability()),
                "total_memory_gib": props.total_memory / 2**30,
            },
            "config": {
                "model": args.model,
                "seed": args.seed,
                "temperature": args.temperature if args.suite == "fixed" else 0.6,
                "max_model_len": kwargs["max_model_len"],
                "gpu_memory_utilization": getattr(
                    llm.model_runner.config, "gpu_memory_utilization", None
                ),
                "kvcache_block_size": getattr(
                    llm.model_runner.config, "kvcache_block_size", None
                ),
                "attention_backend": getattr(
                    llm.model_runner.config, "attention_backend", "upstream-flash"
                ),
                "enforce_eager_resolved": llm.model_runner.enforce_eager,
            },
            "workload": {
                "requests": len(prompts),
                "input_tokens": sum(
                    len(item) if not isinstance(item, str) else len(llm.tokenizer.encode(item))
                    for item in prompts
                ),
                "expected_output_tokens": expected_output_tokens,
                "actual_output_tokens": actual_output_tokens,
            },
            "result": {
                "init_seconds": init_seconds,
                "elapsed_seconds": elapsed,
                "output_tokens_per_second": actual_output_tokens / elapsed,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                "output_sha256": token_digest(outputs),
            },
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps(payload["result"], indent=2), flush=True)
        print(f"wrote={output}", flush=True)
    finally:
        llm.exit()


if __name__ == "__main__":
    main()
