"""Run an end-to-end matrix in separate processes and summarize its variance."""
import argparse, json, statistics, subprocess, sys
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--backend", choices=["auto", "sdpa", "flashinfer", "flash"], default="flashinfer")
    parser.add_argument("--rmsnorm-backend", choices=["auto", "torch", "sm120"], default="sm120")
    parser.add_argument("--qk-norm-rope-backend", choices=["auto", "torch", "sm120"], default="torch")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--gpu-memory-reserve-gib", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261007)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2 for variance reporting")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for index in range(args.runs):
        output = output_dir / f"run_{index + 1:02d}.json"
        command = [
            sys.executable, "-m", "benchmarks.e2e.end_to_end",
            "--model", args.model, "--output", str(output),
            "--backend", args.backend, "--rmsnorm-backend", args.rmsnorm_backend,
            "--qk-norm-rope-backend", args.qk_norm_rope_backend,
            "--max-model-len", str(args.max_model_len),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--gpu-memory-reserve-gib", str(args.gpu_memory_reserve_gib),
            "--warmup-repeats", "1", "--repeats", "1",
            "--seed", str(args.seed + index),
        ]
        print(f"process {index + 1}/{args.runs}: starting", flush=True)
        subprocess.run(command, check=True)
        runs.append(json.loads(output.read_text(encoding="utf-8")))
    summary = {
        "schema_version": 1, "method": "independent_processes", "runs": args.runs,
        "environment": runs[-1]["environment"], "config": runs[-1]["config"],
        "workloads": [],
    }
    for wi, reference in enumerate(runs[0]["workloads"]):
        items = [item["workloads"][wi]["repeats"][0] for item in runs]
        metrics = {}
        non_scalar_metrics = {}
        for name in items[0]:
            values = [item[name] for item in items]
            if all(isinstance(value, (int, float)) and not isinstance(value, bool)
                   for value in values):
                numbers = [float(value) for value in values]
                metrics[name] = {"median": statistics.median(numbers),
                                 "mean": statistics.mean(numbers),
                                 "stdev": statistics.stdev(numbers),
                                 "min": min(numbers), "max": max(numbers)}
            else:
                non_scalar_metrics[name] = values
        summary["workloads"].append({
            "requests": reference["requests"], "input_len": reference["input_len"],
            "output_len": reference["output_len"], "metrics": metrics,
            "non_scalar_metrics": non_scalar_metrics,
        })
        e2e = metrics["output_tokens_per_second"]
        print(f"requests={reference['requests']} input={reference['input_len']} "
              f"e2e_median={e2e['median']:.1f} tok/s sd={e2e['stdev']:.2f} n={args.runs}",
              flush=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={path}")

if __name__ == "__main__":
    main()
