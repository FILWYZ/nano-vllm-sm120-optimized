"""Capture one warm Qwen3 decode request with Nsight Systems' CUDA API range."""

from __future__ import annotations

import argparse
import json
import random
import time

import torch

from nanovllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--backend", choices=["flashinfer", "sdpa", "auto"], default="flashinfer")
    parser.add_argument("--rmsnorm-backend", choices=["sm120", "torch", "auto"], default="sm120")
    parser.add_argument("--qk-norm-rope-backend", choices=["torch", "sm120", "auto"], default="torch")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    prompts = [
        [rng.randint(10, 10000) for _ in range(args.input_len)]
        for _ in range(args.requests)
    ]
    params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=args.output_len,
    )
    llm = LLM(
        args.model,
        attention_backend=args.backend,
        rmsnorm_backend=args.rmsnorm_backend,
        qk_norm_rope_backend=args.qk_norm_rope_backend,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(2048, args.max_model_len * args.requests),
        max_num_seqs=args.requests,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kvcache_block_size=16,
    )

    # Trigger model kernels, compilation, CUDA Graph capture, and allocator setup
    # outside the profile range. Keep the measured trace focused on one warm call.
    llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    status = torch.cuda.cudart().cudaProfilerStart()
    if status != 0:
        raise RuntimeError(f"cudaProfilerStart failed: {status}")
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    stop_status = torch.cuda.cudart().cudaProfilerStop()
    if stop_status != 0:
        raise RuntimeError(f"cudaProfilerStop failed: {stop_status}")

    token_count = sum(len(item["token_ids"]) for item in outputs)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": args.model,
        "requests": args.requests,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "requested_backend": args.backend,
        "resolved_backend": llm.model_runner.config.attention_backend,
        "rmsnorm_backend": llm.model_runner.config.rmsnorm_backend,
        "qk_norm_rope_backend": llm.model_runner.config.qk_norm_rope_backend,
        "elapsed_seconds_including_sync": elapsed,
        "generated_tokens": token_count,
        "output_tokens_per_second": token_count / elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }
    print(json.dumps(result, indent=2))
    llm.exit()


if __name__ == "__main__":
    main()
