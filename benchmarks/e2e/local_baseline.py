import argparse
import random
import time

import torch

from nanovllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce the local nano-vLLM baseline")
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--min-input-len", type=int, default=64)
    parser.add_argument("--max-input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    llm = LLM(
        args.model,
        attention_backend="auto",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=max(1024, args.max_model_len),
        max_num_seqs=args.requests,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    prompts = [
        [
            random.randint(0, 10000)
            for _ in range(random.randint(args.min_input_len, args.max_input_len))
        ]
        for _ in range(args.requests)
    ]
    params = SamplingParams(
        temperature=0.6,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    output_tokens = sum(len(output["token_ids"]) for output in outputs)

    print(f"device={torch.cuda.get_device_name()}")
    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"backend={llm.model_runner.config.attention_backend}")
    print(f"requests={len(prompts)} output_tokens={output_tokens}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"output_tokens_per_second={output_tokens / elapsed:.2f}")
    print(f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")
    print(f"peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}")
    llm.exit()


if __name__ == "__main__":
    main()
