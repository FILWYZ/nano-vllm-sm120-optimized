import glob
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "results"
AB = ROOT / "m8_ab"
SHAPES = [(1, 64), (4, 64), (8, 64), (1, 256), (4, 256), (8, 256)]


def load(pattern):
    paths = sorted(glob.glob(str(pattern)))
    if not paths:
        raise RuntimeError(f"no files matched {pattern}")
    return [json.loads(Path(path).read_text()) for path in paths]


def stats(values):
    return {
        "values": values,
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def validate_tokens(items):
    for item in items:
        workload = item["workload"]
        if workload["actual_output_tokens"] != workload["expected_output_tokens"]:
            raise AssertionError(f"incomplete output in {item}")


def main():
    fixed = []
    fixed_speedups = []
    for requests, input_len in SHAPES:
        upstream = load(AB / f"upstream_compat_r[123]_q{requests}_i{input_len}.json")
        optimized = load(AB / f"optimized_m7_r[123]_q{requests}_i{input_len}.json")
        validate_tokens(upstream + optimized)
        upstream_stats = stats([x["result"]["output_tokens_per_second"] for x in upstream])
        optimized_stats = stats([x["result"]["output_tokens_per_second"] for x in optimized])
        speedup = optimized_stats["median"] / upstream_stats["median"]
        fixed_speedups.append(speedup)
        fixed.append({
            "requests": requests,
            "input_len": input_len,
            "output_len": 32,
            "upstream_compat_tok_s": upstream_stats,
            "optimized_m7_tok_s": optimized_stats,
            "median_speedup": speedup,
        })

    upstream = load(AB / "upstream_compat_github_r[123].json")
    optimized = load(AB / "optimized_m8_offline_github_r[123].json")
    validate_tokens(upstream + optimized)
    upstream_tps = stats([x["result"]["output_tokens_per_second"] for x in upstream])
    optimized_tps = stats([x["result"]["output_tokens_per_second"] for x in optimized])
    upstream_time = stats([x["result"]["elapsed_seconds"] for x in upstream])
    optimized_time = stats([x["result"]["elapsed_seconds"] for x in optimized])
    upstream_memory = stats([x["result"]["peak_allocated_gib"] for x in upstream])
    optimized_memory = stats([x["result"]["peak_allocated_gib"] for x in optimized])
    speedup = optimized_tps["median"] / upstream_tps["median"]

    payload = {
        "schema_version": 1,
        "comparison_scope": {
            "upstream_github_commit": "bb823b3e06983d71485a8e1f23715ebd87d98ef8",
            "upstream_sm120_compat_commit": "7aa1f251cf1e041ded3371bab41f90e6736111ac",
            "optimized_m7_code_commit": "56fc37163d572aae5c95e1408f0d723f90d58fe5",
            "m8_offline_admission_commit": "2133478",
            "device": optimized[0]["environment"]["device"],
            "model": optimized[0]["config"]["model"],
        },
        "fixed_matrix": {
            "repeats_per_shape": 3,
            "workloads": fixed,
            "geomean_median_speedup": math.exp(
                sum(math.log(x) for x in fixed_speedups) / len(fixed_speedups)
            ),
        },
        "github_readme_workload": {
            "repeats": 3,
            "requests": 256,
            "input_tokens": optimized[0]["workload"]["input_tokens"],
            "output_tokens": optimized[0]["workload"]["actual_output_tokens"],
            "upstream_compat_tok_s": upstream_tps,
            "optimized_m8_tok_s": optimized_tps,
            "upstream_compat_seconds": upstream_time,
            "optimized_m8_seconds": optimized_time,
            "upstream_peak_allocated_gib": upstream_memory,
            "optimized_peak_allocated_gib": optimized_memory,
            "median_speedup": speedup,
            "median_throughput_gain_percent": (speedup - 1) * 100,
            "median_time_reduction_percent": (
                1 - optimized_time["median"] / upstream_time["median"]
            ) * 100,
            "median_peak_memory_change_percent": (
                optimized_memory["median"] / upstream_memory["median"] - 1
            ) * 100,
            "upstream_output_hash_stable": len({x["result"]["output_sha256"] for x in upstream}) == 1,
            "optimized_output_hash_stable": len({x["result"]["output_sha256"] for x in optimized}) == 1,
            "optimized_scheduler": optimized[0]["engine_metrics"]["scheduler"],
            "optimized_prefill_tokens": optimized[0]["engine_metrics"]["prefill_tokens"],
        },
    }
    output = ROOT / "m8_upstream_comparison_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print(f"wrote={output}")


if __name__ == "__main__":
    main()
