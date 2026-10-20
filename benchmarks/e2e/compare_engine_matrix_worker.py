"""Run the six fixed workloads in one isolated engine process."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


SHAPES = [(n, p, 32) for p in (64, 256) for n in (1, 4, 8)]


def git_commit(path):
    if not path:
        return None
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["nano", "compat", "upstream", "vllm"], required=True)
    parser.add_argument("--repo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--shape-offset", type=int, default=0)
    return parser.parse_args()


def prompts(seed, count, length):
    rng = random.Random(seed)
    return [[rng.randint(10, 10_000) for _ in range(length)] for _ in range(count)]


def digest(output_tokens):
    h = hashlib.sha256()
    for row in output_tokens:
        for token in row:
            h.update(int(token).to_bytes(4, "little", signed=False))
    return h.hexdigest()


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    i = (len(values) - 1) * q
    lo, hi = int(i), min(int(i) + 1, len(values) - 1)
    return values[lo] * (hi - i) + values[hi] * (i - lo)


def nvidia_smi_used_gib():
    value = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
        timeout=5,
    ).strip().splitlines()[0]
    return float(value) / 1024.0


def main():
    args = parse_args()
    result_path = Path(args.output)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "repo": args.repo,
        "git_commit": git_commit(args.repo),
        "status": "failed",
        "results": {},
    }
    try:
        if args.engine in {"nano", "compat", "upstream"}:
            if not args.repo:
                raise ValueError("--repo is required for nano engines")
            sys.path.insert(0, str(Path(args.repo).resolve()))
            from nanovllm import LLM, SamplingParams
        else:
            from vllm import LLM, SamplingParams
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        props = torch.cuda.get_device_properties(0)
        result["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": props.name,
            "compute_capability": list(torch.cuda.get_device_capability()),
            "total_memory_gib": props.total_memory / 2**30,
        }
        if args.engine == "upstream":
            import flash_attn
            result["environment"]["flash_attn"] = getattr(flash_attn, "__version__", "installed")
        if args.engine == "nano":
            try:
                import sm120_rmsnorm
                result["environment"]["sm120_rmsnorm_path"] = sm120_rmsnorm.__file__
            except Exception as exc:
                result["environment"]["sm120_rmsnorm_error"] = f"{type(exc).__name__}: {exc}"
        common = {
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if args.engine == "nano":
            engine_kwargs = {
                **common,
                "enforce_eager": False,
                "kvcache_block_size": 64,
                "attention_backend": "flashinfer",
                "rmsnorm_backend": "sm120",
                "qk_norm_rope_backend": "auto",
                "reserve_decode_kv": True,
            }
        elif args.engine == "compat":
            engine_kwargs = {
                **common,
                "enforce_eager": False,
                "kvcache_block_size": 256,
                "attention_backend": "sdpa",
            }
        elif args.engine == "upstream":
            engine_kwargs = {**common, "enforce_eager": False, "kvcache_block_size": 256}
        else:
            engine_kwargs = {
                **common,
                "dtype": "bfloat16",
                "seed": args.seed,
                "enforce_eager": False,
                "enable_prefix_caching": False,
                "disable_log_stats": True,
            }
        init_start = time.perf_counter()
        llm = LLM(args.model, **engine_kwargs)
        result["init_seconds"] = time.perf_counter() - init_start
        result["config"] = engine_kwargs
        result["vllm_version"] = getattr(sys.modules.get("vllm"), "__version__", None)
        result["gpu_memory_sampling"] = "nvidia-smi snapshots immediately before and after each measured batch; transient peaks may be missed"

        completion_times = {}
        if args.engine != "vllm":
            original_step = llm.step

            def timed_step(*step_args, **step_kwargs):
                batch = original_step(*step_args, **step_kwargs)
                finished_at = time.perf_counter()
                for sequence_id, _ in batch[0]:
                    completion_times[sequence_id] = finished_at
                return batch

            llm.step = timed_step

        ordered_shapes = SHAPES[args.shape_offset % len(SHAPES):] + SHAPES[:args.shape_offset % len(SHAPES)]
        for count, input_len, output_len in ordered_shapes:
            name = f"n{count}_p{input_len}_o{output_len}"
            params = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=output_len)
            warm = prompts(args.seed + 1_000_000 + input_len * 100 + count, count, input_len)
            inputs = prompts(args.seed + input_len * 100 + count, count, input_len)
            if args.engine == "vllm":
                warm = [{"prompt_token_ids": row} for row in warm]
                inputs = [{"prompt_token_ids": row} for row in inputs]
            llm.generate(warm, params, use_tqdm=False)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            completion_times.clear()
            total_before = nvidia_smi_used_gib()
            torch.manual_seed(args.seed + input_len * 100 + count)
            started = time.perf_counter()
            outputs = llm.generate(inputs, params, use_tqdm=False)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            total_peak = max(total_before, nvidia_smi_used_gib())

            if args.engine == "vllm":
                token_rows = [list(item.outputs[0].token_ids) for item in outputs]
                request_latencies = []
                for item in outputs:
                    metrics = getattr(item, "metrics", None)
                    arrival = getattr(metrics, "arrival_time", None)
                    finished = getattr(metrics, "finished_time", None)
                    if arrival is not None and finished is not None:
                        request_latencies.append(finished - arrival)
            else:
                token_rows = [list(item["token_ids"]) for item in outputs]
                request_latencies = [finished - started for finished in completion_times.values()]
            output_tokens = sum(map(len, token_rows))
            if len(outputs) != count or output_tokens != count * output_len:
                raise RuntimeError(
                    f"{name} output mismatch: {len(outputs)}/{count} requests, "
                    f"{output_tokens}/{count * output_len} tokens"
                )
            result["results"][name] = {
                "status": "passed",
                "requests": count,
                "input_tokens_per_request": input_len,
                "output_tokens_per_request": output_len,
                "output_tokens": output_tokens,
                "elapsed_seconds": elapsed,
                "requests_per_second": count / elapsed,
                "output_tokens_per_second": output_tokens / elapsed,
                "request_latency_seconds": {
                    "count": len(request_latencies),
                    "p50": pct(request_latencies, 0.50),
                    "p95": pct(request_latencies, 0.95),
                    "definition": "arrival to completion within one offline batch; vLLM exposes batch latency when per-request timestamps are unavailable",
                },
                "memory_gib": {
                    "gpu_total_used_baseline": total_before,
                    "gpu_total_used_max_sample": total_peak,
                    "gpu_total_used_delta": total_peak - total_before if total_peak is not None and total_before is not None else None,
                    "pytorch_peak_allocated": torch.cuda.max_memory_allocated() / 2**30,
                    "pytorch_peak_reserved": torch.cuda.max_memory_reserved() / 2**30,
                },
                "output_sha256": digest(token_rows),
            }
            print(
                f"{args.engine} {name}: {result['results'][name]['output_tokens_per_second']:.2f} tok/s, "
                f"{elapsed:.3f}s, gpu_peak={total_peak}", flush=True
            )
        result["status"] = "passed"
        if args.engine != "vllm":
            llm.exit()
    except BaseException as exc:
        result["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        active = result.get("results", {})
        result["failed_shape"] = next((s for s in [f"n{n}_p{p}_o{o}" for p in (64, 256) for n in (1, 4, 8)] if s not in active), None)
    finally:
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"engine": args.engine, "status": result.get("status"), "failed_shape": result.get("failed_shape"), "error": result.get("error")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
