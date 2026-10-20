"""Summarize the archived 2026-10-20 fixed-workload engine comparisons."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path


SHAPES = [f"n{n}_p{p}_o32" for p in (64, 256) for n in (1, 4, 8)]


def median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def read_rounds(directory: Path, engine: str):
    rows = []
    for index in range(1, 4):
        path = directory / f"matrix_round_{index:02d}_{engine}.json"
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def summarize_engine(rows):
    successful = sum(row.get("status") == "passed" for row in rows)
    summary = {
        "successful_rounds": successful,
        "total_rounds": len(rows),
        "environment": rows[0].get("environment", {}),
        "git_commit": rows[0].get("git_commit"),
        "vllm_version": rows[0].get("vllm_version"),
        "gpu_memory_sampling": rows[0].get("gpu_memory_sampling"),
        "shapes": {},
    }
    for shape in SHAPES:
        items = [row.get("results", {}).get(shape, {}) for row in rows]
        good = [item for item in items if item.get("status") == "passed"]
        summary["shapes"][shape] = {
            "successful_runs": len(good),
            "failures": [
                item.get("error") or {"status": item.get("status")}
                for item in items
                if item.get("status") != "passed"
            ],
            "median_output_tokens_per_second": median(
                [item.get("output_tokens_per_second") for item in good]
            ),
            "median_batch_latency_seconds": median(
                [item.get("elapsed_seconds") for item in good]
            ),
            "median_request_latency_p50_seconds": median(
                [item.get("request_latency_seconds", {}).get("p50") for item in good]
            ),
            "median_request_latency_p95_seconds": median(
                [item.get("request_latency_seconds", {}).get("p95") for item in good]
            ),
            "median_gpu_total_used_baseline_gib": median(
                [item.get("memory_gib", {}).get("gpu_total_used_baseline") for item in good]
            ),
            "median_gpu_total_used_max_sample_gib": median(
                [item.get("memory_gib", {}).get("gpu_total_used_max_sample") for item in good]
            ),
            "median_gpu_total_used_delta_gib": median(
                [item.get("memory_gib", {}).get("gpu_total_used_delta") for item in good]
            ),
            "median_pytorch_peak_allocated_gib": median(
                [item.get("memory_gib", {}).get("pytorch_peak_allocated") for item in good]
            ),
            "median_pytorch_peak_reserved_gib": median(
                [item.get("memory_gib", {}).get("pytorch_peak_reserved") for item in good]
            ),
            "runs": items,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    compat_dir = args.results_root / "current_engine_ab_20261020_formal"
    latest_dir = args.results_root / "current_vs_vllm032_20261020"
    rows = {
        "nano_current": read_rounds(latest_dir, "nano"),
        "compat_baseline": read_rounds(compat_dir, "compat"),
        "vllm_0_31_0": read_rounds(compat_dir, "vllm"),
        "vllm_0_32_0": read_rounds(latest_dir, "vllm"),
    }
    engines = {name: summarize_engine(items) for name, items in rows.items()}

    comparisons = {}
    for reference in ("compat_baseline", "vllm_0_31_0", "vllm_0_32_0"):
        per_shape = {}
        ratios = []
        for shape in SHAPES:
            numerator = engines["nano_current"]["shapes"][shape]["median_output_tokens_per_second"]
            denominator = engines[reference]["shapes"][shape]["median_output_tokens_per_second"]
            ratio = numerator / denominator if numerator and denominator else None
            per_shape[shape] = ratio
            if ratio is not None:
                ratios.append(ratio)
        per_shape["geometric_mean"] = (
            math.exp(sum(math.log(value) for value in ratios) / len(ratios)) if ratios else None
        )
        comparisons[f"nano_current_over_{reference}"] = per_shape

    first = rows["nano_current"][0]
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "three interleaved rounds; each engine process runs the same six fixed offline token-ID workloads after per-shape warmup",
        "model": {
            "name": "Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "model_safetensors_sha256": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        },
        "workload_matrix": [
            {"requests": n, "input_tokens_per_request": p, "output_tokens_per_request": 32}
            for p in (64, 256)
            for n in (1, 4, 8)
        ],
        "controls": {
            "precision": "BF16",
            "seed": 20261020,
            "temperature": 0.6,
            "ignore_eos": True,
            "max_model_len": 512,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 2048,
            "gpu_memory_utilization": 0.70,
            "enable_prefix_caching": False,
            "workload_input_ids": "deterministically generated by the worker from seed and input length; identical across engines",
        },
        "hardware": first.get("environment", {}),
        "engines": engines,
        "nano_over_reference_throughput": comparisons,
        "limitations": [
            "GPU total memory is sampled immediately before and after each measured batch; transient peaks may be missed.",
            "vLLM runs generation in a separate EngineCore process, so PyTorch allocator peaks from the outer process are zero and should not be compared.",
            "vLLM 0.32.0 resolved PyTorch 2.13.0+cu132, while nano-vLLM and the compatibility baseline use PyTorch 2.14.1+cu130.",
            "The original unmodified upstream nano-vLLM did not produce a successful local run; compatibility baseline results are a substitute comparison.",
            "This is an engine-level comparison. Attention backend, KV block size, cache policy and framework stack differ by engine.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
