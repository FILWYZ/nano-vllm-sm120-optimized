import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "output_tokens_per_second",
    "batch_ttft_seconds",
    "mean_tpot_seconds",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.result_dir.glob("*.json"))
    grouped: dict[str, list[dict]] = {"torch": [], "sm120": []}
    for path in files:
        backend = path.stem.split("_", 1)[0]
        grouped[backend].append(json.loads(path.read_text(encoding="utf-8")))
    if not grouped["torch"] or len(grouped["torch"]) != len(grouped["sm120"]):
        raise RuntimeError("A/B inputs must contain the same non-zero number of runs")

    workloads = []
    for index, torch_case in enumerate(grouped["torch"][0]["workloads"]):
        result = {
            key: torch_case[key] for key in ("requests", "input_len", "output_len")
        }
        for metric in METRICS:
            torch_values = [run["workloads"][index]["mean"][metric] for run in grouped["torch"]]
            sm120_values = [run["workloads"][index]["mean"][metric] for run in grouped["sm120"]]
            torch_median = statistics.median(torch_values)
            sm120_median = statistics.median(sm120_values)
            higher_is_better = metric.endswith("tokens_per_second")
            delta = (
                (sm120_median / torch_median - 1.0) * 100.0
                if higher_is_better
                else (1.0 - sm120_median / torch_median) * 100.0
            )
            result[metric] = {
                "torch_median": torch_median,
                "sm120_median": sm120_median,
                "improvement_percent": delta,
            }
        workloads.append(result)
    payload = {
        "schema_version": 1,
        "runs_per_backend": len(grouped["torch"]),
        "method": "independent processes, alternating A/B order, median across runs",
        "environment": grouped["sm120"][0]["environment"],
        "config": grouped["sm120"][0]["config"],
        "workloads": workloads,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
