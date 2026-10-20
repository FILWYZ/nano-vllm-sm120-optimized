\
"""Launch the local OpenAI-compatible nano-vLLM service."""
import argparse
import torch

from nanovllm import LLM
from nanovllm.server import serve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local Hugging Face model directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backend", choices=["auto", "flashinfer", "sdpa"], default="auto")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-chunk-size", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--gpu-memory-reserve-gib", type=float, default=0.50)
    parser.add_argument("--max-queue-size", type=int, default=32)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable; expose the Windows NVIDIA driver to WSL2 first."
        )
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not 0.1 <= args.gpu_memory_utilization < 1.0:
        raise SystemExit("--gpu-memory-utilization must be in [0.1, 1.0)")
    if args.gpu_memory_reserve_gib < 0:
        raise SystemExit("--gpu-memory-reserve-gib cannot be negative")
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("The server binds only to loopback; remote binding is disabled.")
    if min(args.max_num_seqs, args.max_batched_tokens,
           args.max_prefill_chunk_size, args.max_queue_size,
           args.max_output_tokens, args.max_model_len) < 1:
        raise SystemExit("token, sequence, queue and context limits must be positive")

    llm = LLM(
        args.model,
        attention_backend=args.backend,
        rmsnorm_backend="auto",
        qk_norm_rope_backend="auto",
        enforce_eager=False,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_batched_tokens,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        gpu_memory_reserve_gib=args.gpu_memory_reserve_gib,
        enable_mixed_batching=True,
        reserve_decode_kv=True,
    )
    serve(
        llm, args.host, args.port, args.max_queue_size,
        args.max_output_tokens, args.max_model_len,
    )


if __name__ == "__main__":
    main()
