import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run isolated, order-balanced QK Norm+RoPE A/B processes"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/results/qk_norm_rope_ab")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for run in range(1, args.runs + 1):
        order = ("torch", "sm120") if run % 2 else ("sm120", "torch")
        for backend in order:
            output = args.output_dir / f"{backend}_{run}.json"
            command = [
                sys.executable,
                "-m",
                "benchmarks.e2e.end_to_end",
                "--model",
                args.model,
                "--backend",
                "flashinfer",
                "--rmsnorm-backend",
                "sm120",
                "--qk-norm-rope-backend",
                backend,
                "--max-model-len",
                "512",
                "--gpu-memory-utilization",
                "0.75",
                "--block-size",
                "16",
                "--warmup-repeats",
                "1",
                "--repeats",
                "2",
                "--seed",
                "2026",
                "--output",
                str(output),
            ]
            print(f"run={run} backend={backend} output={output}", flush=True)
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
