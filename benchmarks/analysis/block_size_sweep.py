import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.common import write_json


BLOCK_SIZES = (16, 32, 64, 128, 256)


def parse_args():
    parser = argparse.ArgumentParser(description="Isolated KV page-size sweep")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output", default="benchmarks/results/m4_block_size_sweep.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    detail_dir = output_path.parent / "m4_block_sizes"
    detail_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for block_size in BLOCK_SIZES:
        detail_path = detail_dir / f"block_{block_size}.json"
        command = [
            sys.executable,
            "-m",
            "benchmarks.end_to_end",
            "--model",
            args.model,
            "--backend",
            "flashinfer",
            "--block-size",
            str(block_size),
            "--output",
            str(detail_path),
            "--warmup-repeats",
            "1",
            "--repeats",
            "1",
        ]
        print(f"running block_size={block_size}", flush=True)
        subprocess.run(command, check=True)
        detail = json.loads(detail_path.read_text())
        throughputs = [
            workload["mean"]["output_tokens_per_second"]
            for workload in detail["workloads"]
        ]
        decode_throughputs = [
            workload["mean"]["decode_tokens_per_second"]
            for workload in detail["workloads"]
        ]
        config = detail["config"]
        results.append({
            "block_size": block_size,
            "geomean_output_tokens_per_second": statistics.geometric_mean(
                throughputs
            ),
            "geomean_decode_tokens_per_second": statistics.geometric_mean(
                decode_throughputs
            ),
            "num_kvcache_blocks": config["num_kvcache_blocks"],
            "kvcache_capacity_tokens": config["kvcache_capacity_tokens"],
            "mean_tail_waste_tokens_per_active_sequence": (block_size - 1) / 2,
            "worst_tail_waste_tokens_per_active_sequence": block_size - 1,
            "prefix_cache_granularity_tokens": block_size,
            "detail": str(detail_path),
        })
    fastest = max(
        results, key=lambda result: result["geomean_output_tokens_per_second"]
    )
    selected = min(
        result["block_size"]
        for result in results
        if result["geomean_output_tokens_per_second"]
        >= 0.95 * fastest["geomean_output_tokens_per_second"]
    )
    payload = {
        "schema_version": 1,
        "method": {
            "process_isolation": True,
            "warmup_repeats": 1,
            "measured_repeats": 1,
            "selection_rule": (
                "smallest page within 5% of fastest throughput, then capacity"
            ),
        },
        "fastest_block_size": fastest["block_size"],
        "selected_block_size": selected,
        "results": results,
    }
    write_json(output_path, payload)
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
