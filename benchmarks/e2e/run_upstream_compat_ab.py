"""Interleave same-workload SM120 compatibility-baseline and optimized runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compat-repo", required=True, help="Clean baseline-blackwell-v0 checkout")
    parser.add_argument("--optimized-repo", default=".", help="Optimized project checkout")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def run_one(args, output_dir, variant, round_index):
    compat = variant == "compat"
    repo = Path(args.compat_repo if compat else args.optimized_repo).resolve()
    output = output_dir / f"round_{round_index:02d}_{variant}.json"
    command = [
        sys.executable,
        str(Path(__file__).with_name("upstream_ab.py")),
        "--model", args.model,
        "--variant", "sm120-compat-baseline" if compat else "optimized-m8",
        "--suite", "github",
        "--output", str(output),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
        "--max-num-batched-tokens", str(args.max_num_batched_tokens),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--block-size", "256" if compat else "64",
        "--attention-backend", "sdpa" if compat else "flashinfer",
    ]
    if not compat:
        command += [
            "--rmsnorm-backend", "sm120",
            "--qk-norm-rope-backend", "torch",
            "--reserve-output-kv",
        ]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(repo) + (os.pathsep + old_pythonpath if old_pythonpath else "")
    print(f"round={round_index} variant={variant} repo={repo}", flush=True)
    with (output_dir / f"round_{round_index:02d}_{variant}.log").open("w") as log:
        subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def main():
    args = parse_args()
    if args.runs < 2:
        raise SystemExit("--runs must be at least 2")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {"compat": [], "optimized": []}
    for round_index in range(1, args.runs + 1):
        order = ("compat", "optimized") if round_index % 2 else ("optimized", "compat")
        for variant in order:
            results[variant].append(run_one(args, output_dir, variant, round_index))
    summary = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": "interleaved independent processes",
        "runs_per_variant": args.runs,
        "matched_workload": {
            "model": args.model,
            "suite": "upstream README request distribution",
            "requests": 256,
            "prompt_length_range": [100, 1024],
            "output_length_range": [100, 1024],
            "seed": 0,
            "temperature": 0.6,
            "ignore_eos": True,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "variants": {
            variant: [
                {
                    "git_commit": result["git_commit"],
                    "resolved_config": result["config"],
                    "workload": result["workload"],
                    "result": result["result"],
                }
                for result in runs
            ]
            for variant, runs in results.items()
        },
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary={path}", flush=True)


if __name__ == "__main__":
    main()
