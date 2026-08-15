import argparse
import gc
import random

import torch

from benchmarks.common import environment, write_json
from nanovllm import LLM, SamplingParams

from nanovllm.layers.flashinfer_backend import reset_flashinfer_runtime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare eager and CUDA-graph completion tokens"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--output", default="benchmarks/results/m3_graph_parity.json"
    )
    return parser.parse_args()


def run(model, prompts, enforce_eager, seed):
    llm = LLM(
        model,
        attention_backend="flashinfer",
        enforce_eager=enforce_eager,
        max_model_len=512,
        max_num_batched_tokens=2048,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
    )
    torch.manual_seed(seed)
    params = SamplingParams(temperature=0.1, ignore_eos=True, max_tokens=16)
    outputs = llm.generate(prompts, params, use_tqdm=False)
    tokens = [output["token_ids"] for output in outputs]
    llm.exit()
    reset_flashinfer_runtime()
    del outputs
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return tokens


def main():
    args = parse_args()
    seed = 2026
    rng = random.Random(seed)
    prompts = [
        [rng.randint(10, 10000) for _ in range(255)]
        for _ in range(3)
    ]
    eager = run(args.model, prompts, True, seed)
    graphed = run(args.model, prompts, False, seed)
    exact_match = eager == graphed
    matched_tokens = sum(
        sum(left == right for left, right in zip(eager_row, graph_row))
        for eager_row, graph_row in zip(eager, graphed)
    )
    total_tokens = sum(map(len, eager))
    payload = {
        "schema_version": 1,
        "environment": environment(),
        "workload": {
            "requests": 3,
            "graph_bucket": 4,
            "input_len": 255,
            "output_len": 16,
            "crosses_page_boundary": True,
        },
        "exact_match": exact_match,
        "matched_tokens": matched_tokens,
        "total_tokens": total_tokens,
        "eager_tokens": eager,
        "graph_tokens": graphed,
    }
    write_json(args.output, payload)
    print(
        f"exact_match={exact_match} matched_tokens={matched_tokens}/{total_tokens}"
    )
    if not exact_match:
        raise SystemExit("CUDA graph output differs from eager output")


if __name__ == "__main__":
    main()
