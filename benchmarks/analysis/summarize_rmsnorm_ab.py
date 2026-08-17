import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "output_tokens_per_second",
    "decode_tokens_per_second",
    "mean_tpot_seconds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RMSNorm backend A/B runs")
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/rmsnorm_ab_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payloads = {
        backend: [
            json.loads(
                (args.results_dir / f"rmsnorm_ab_{backend}_{run}.json").read_text(
                    encoding="utf-8"
                )
            )
            for run in range(1, args.runs + 1)
        ]
        for backend in ("torch", "sm120")
    }

    workloads = []
    workload_count = len(payloads["torch"][0]["workloads"])
    for index in range(workload_count):
        first = payloads["torch"][0]["workloads"][index]
        medians = {}
        independent_runs = {}
        for backend, runs in payloads.items():
            independent_runs[backend] = {
                metric: [run["workloads"][index]["mean"][metric] for run in runs]
                for metric in METRICS
            }
            medians[backend] = {
                metric: statistics.median(independent_runs[backend][metric])
                for metric in METRICS
            }

        torch_values = medians["torch"]
        sm120_values = medians["sm120"]
        workloads.append(
            {
                "requests": first["requests"],
                "input_len": first["input_len"],
                "output_len": first["output_len"],
                "median": medians,
                "independent_runs": independent_runs,
                "output_throughput_change_pct": (
                    sm120_values["output_tokens_per_second"]
                    / torch_values["output_tokens_per_second"]
                    - 1.0
                )
                * 100.0,
                "decode_throughput_change_pct": (
                    sm120_values["decode_tokens_per_second"]
                    / torch_values["decode_tokens_per_second"]
                    - 1.0
                )
                * 100.0,
                "tpot_change_pct": (
                    sm120_values["mean_tpot_seconds"]
                    / torch_values["mean_tpot_seconds"]
                    - 1.0
                )
                * 100.0,
            }
        )

    summary = {
        "schema_version": 1,
        "environment": payloads["torch"][0]["environment"],
        "protocol": {
            "independent_processes_per_backend": args.runs,
            "within_process_repeats": payloads["torch"][0]["config"]["repeats"],
            "warmup_repeats": payloads["torch"][0]["config"]["warmup_repeats"],
            "run_order": ["torch", "sm120", "sm120", "torch", "torch", "sm120"],
        },
        "workloads": workloads,
    }
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
