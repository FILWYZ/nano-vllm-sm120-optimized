"""Long-running local inference soak with GPU power and clock telemetry."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from nanovllm import LLM, SamplingParams


NVSMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=timestamp,power.draw,temperature.gpu,clocks.gr,clocks.mem,utilization.gpu",
    "--format=csv,noheader,nounits",
    "--loop=1",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, help="Summary JSON path")
    parser.add_argument("--telemetry-csv", required=True)
    parser.add_argument("--calls-csv", required=True)
    parser.add_argument("--duration-minutes", type=float, default=30)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--idle-seconds", type=int, default=10)
    return parser.parse_args()


def as_float(value):
    try:
        return float(value.strip())
    except (AttributeError, TypeError, ValueError):
        return None


def start_telemetry():
    process = subprocess.Popen(
        NVSMI_QUERY,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        bufsize=1,
    )
    samples = []
    done = threading.Event()

    def read_samples():
        try:
            for row in csv.reader(process.stdout):
                if len(row) < 6:
                    continue
                samples.append(
                    {
                        "monotonic_s": time.monotonic(),
                        "timestamp": row[0].strip(),
                        "power_w": as_float(row[1]),
                        "temperature_c": as_float(row[2]),
                        "graphics_clock_mhz": as_float(row[3]),
                        "memory_clock_mhz": as_float(row[4]),
                        "gpu_utilization_percent": as_float(row[5]),
                    }
                )
        finally:
            done.set()

    thread = threading.Thread(target=read_samples, name="nvidia-smi-telemetry", daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not samples and time.monotonic() < deadline and process.poll() is None:
        time.sleep(0.1)
    if not samples:
        process.terminate()
        raise RuntimeError("nvidia-smi telemetry did not produce samples")
    return process, thread, samples, done


def make_prompts(batch, length, seed):
    rng = random.Random(seed)
    return [[rng.randint(10, 10_000) for _ in range(length)] for _ in range(batch)]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def integrate_power(samples, start, end):
    points = [(s["monotonic_s"], s["power_w"]) for s in samples if s["power_w"] is not None]
    energy = 0.0
    covered = 0.0
    for (t0, p0), (t1, p1) in zip(points, points[1:]):
        a, b = max(start, t0), min(end, t1)
        if b <= a or t1 <= t0:
            continue
        pa = p0 + (p1 - p0) * (a - t0) / (t1 - t0)
        pb = p0 + (p1 - p0) * (b - t0) / (t1 - t0)
        energy += (pa + pb) * 0.5 * (b - a)
        covered += b - a
    return energy, covered


def main():
    args = parse_args()
    if args.duration_minutes <= 0 or args.requests <= 0 or args.output_len <= 0:
        raise ValueError("duration, requests, and output length must be positive")

    llm = LLM(
        args.model,
        attention_backend="flashinfer",
        rmsnorm_backend="sm120",
        qk_norm_rope_backend="torch",
        enforce_eager=False,
        tensor_parallel_size=1,
        max_model_len=512,
        max_num_batched_tokens=4096,
        max_num_seqs=args.requests,
        gpu_memory_utilization=0.75,
        kvcache_block_size=16,
    )
    cases = [(64, args.output_len), (256, args.output_len)]
    params = {out_len: SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=out_len)
              for _, out_len in cases}
    proc, thread, telemetry, done = start_telemetry()
    calls = []
    engine_config = {
        "attention": llm.model_runner.config.attention_backend,
        "rmsnorm": llm.model_runner.config.rmsnorm_backend,
        "qk_norm_rope": llm.model_runner.config.qk_norm_rope_backend,
        "kv_page_tokens": llm.model_runner.config.kvcache_block_size,
        "cuda_graph": not llm.model_runner.enforce_eager,
    }
    try:
        for case_i, (prompt_len, output_len) in enumerate(cases):
            for warmup_i in range(3):
                llm.generate(
                    make_prompts(args.requests, prompt_len, args.seed + 1_000_000 + case_i * 100 + warmup_i),
                    params[output_len],
                    use_tqdm=False,
                )
        torch.cuda.synchronize()
        time.sleep(max(0, args.idle_seconds))
        measure_start = time.monotonic()
        deadline = measure_start + args.duration_minutes * 60
        call_index = 0
        while time.monotonic() < deadline:
            case_i = call_index % len(cases)
            prompt_len, output_len = cases[case_i]
            started = time.monotonic()
            torch.cuda.reset_peak_memory_stats()
            outputs = llm.generate(
                make_prompts(args.requests, prompt_len, args.seed + call_index),
                params[output_len],
                use_tqdm=False,
            )
            torch.cuda.synchronize()
            ended = time.monotonic()
            metrics = dict(llm.last_metrics)
            completion_tokens = sum(len(item["token_ids"]) for item in outputs)
            calls.append(
                {
                    "case": f"b{args.requests}_p{prompt_len}_o{output_len}",
                    "started_monotonic_s": started,
                    "ended_monotonic_s": ended,
                    "elapsed_s": ended - started,
                    "completion_tokens": completion_tokens,
                    "output_tokens_per_second": completion_tokens / (ended - started),
                    "preemptions": metrics.get("scheduler", {}).get("preemptions", 0),
                    "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
                }
            )
            call_index += 1
            if call_index % 100 == 0:
                elapsed = ended - measure_start
                print(
                    f"elapsed_min={elapsed / 60:.1f} calls={call_index} "
                    f"tokens={sum(c['completion_tokens'] for c in calls)} "
                    f"last_tps={calls[-1]['output_tokens_per_second']:.1f}",
                    flush=True,
                )
        measure_end = time.monotonic()
        torch.cuda.synchronize()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        thread.join(timeout=5)
        llm.exit()

    measured_samples = [s for s in telemetry if measure_start <= s["monotonic_s"] <= measure_end]
    idle_samples = [s for s in telemetry if measure_start - args.idle_seconds <= s["monotonic_s"] < measure_start]
    gross_energy_j, covered_s = integrate_power(telemetry, measure_start, measure_end)
    idle_w = statistics.median([s["power_w"] for s in idle_samples if s["power_w"] is not None]) if idle_samples else None
    total_tokens = sum(c["completion_tokens"] for c in calls)
    measured_s = measure_end - measure_start
    net_energy_j = max(0.0, gross_energy_j - (idle_w or 0.0) * covered_s)
    per_minute = []
    for minute in range(int(measured_s // 60) + 1):
        lo = measure_start + minute * 60
        hi = min(measure_end, lo + 60)
        tokens = sum(c["completion_tokens"] for c in calls if lo <= c["ended_monotonic_s"] < hi)
        if hi > lo:
            per_minute.append({"minute": minute, "seconds": hi - lo, "output_tokens": tokens,
                               "tokens_per_second": tokens / (hi - lo)})

    summary = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "model_revision": "c1899de289a04d12100db370d81485cdf75e47ca",
        "model": args.model,
        "dtype": "bfloat16",
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "software": {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                      "python": __import__("platform").python_version()},
        "engine": engine_config,
        "workload": {"requests": args.requests, "cases": [
            {"prompt_tokens": p, "output_tokens": o} for p, o in cases],
            "temperature": 0.6, "ignore_eos": True, "warmups_per_case": 3,
            "duration_target_s": args.duration_minutes * 60,
            "duration_measured_s": measured_s, "calls": len(calls),
            "successful_calls": len(calls), "failed_calls": 0,
            "preemptions": sum(c["preemptions"] for c in calls),
            "completion_tokens": total_tokens},
        "performance": {"whole_window_tokens_per_second": total_tokens / measured_s,
                        "call_tps": stats([c["output_tokens_per_second"] for c in calls]),
                        "per_minute": per_minute,
                        "peak_allocated_gib": max((c["peak_allocated_gib"] for c in calls), default=None),
                        "peak_reserved_gib": max((c["peak_reserved_gib"] for c in calls), default=None)},
        "energy": {"method": "trapezoidal integral of nvidia-smi GPU board power at 1 s; idle estimate from pre-run samples",
                   "sample_interval_s": 1, "samples_in_window": len(measured_samples),
                   "covered_s": covered_s, "idle_power_median_w": idle_w,
                   "gross_board_energy_j": gross_energy_j,
                   "idle_adjusted_board_energy_j_estimate": net_energy_j,
                   "gross_tokens_per_joule": total_tokens / gross_energy_j if gross_energy_j else None,
                   "idle_adjusted_tokens_per_joule_estimate": total_tokens / net_energy_j if net_energy_j else None},
        "gpu_telemetry": {key: stats([s[key] for s in measured_samples]) for key in (
            "power_w", "temperature_c", "graphics_clock_mhz", "memory_clock_mhz", "gpu_utilization_percent")},
        "limitations": ["GPU board energy excludes laptop CPU, display, memory, and AC/DC conversion losses.",
                        "Idle-adjusted energy subtracts a pre-run idle median and is an estimate.",
                        "This is a fixed synthetic-token prompt soak, not a diverse live-user traffic trace."],
        "calls": calls,
        "telemetry": telemetry,
    }
    # The JSON retains minute-level summaries; raw power and per-call rows are CSVs.
    summary.pop("calls")
    summary.pop("telemetry")

    out = Path(args.output)
    telemetry_path = Path(args.telemetry_csv)
    calls_path = Path(args.calls_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    with telemetry_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(telemetry[0]) if telemetry else [])
        writer.writeheader()
        writer.writerows(telemetry)
    with calls_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(calls[0]) if calls else [])
        writer.writeheader()
        writer.writerows(calls)
    print(json.dumps({"output": str(out), "telemetry_csv": str(telemetry_path),
                      "calls_csv": str(calls_path),
                      "performance": summary["performance"], "energy": summary["energy"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
