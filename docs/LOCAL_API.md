# Local API

The service binds to loopback only and runs one engine-owning worker thread. It supports bounded admission, continuous batching, streaming text, queue backpressure, health checks, and cancellation when a streaming client disconnects.

## Start

    cd /home/asus/nano-vllm
    source .venv/bin/activate
    export FLASHINFER_WORKSPACE_BASE=/home/asus/.cache/nano-vllm-flashinfer
    python -m nanovllm.serve --model /home/asus/models/Qwen3-0.6B

Defaults are conservative for an 8 GiB card: max context 2048, at most 8 active sequences, GPU utilization 0.70, and a 0.50 GiB explicit reserve. Reduce context, active sequences, or utilization if other applications need GPU memory.

## Endpoints

- GET /health and /healthz
- GET /v1/models
- POST /v1/completions
- POST /v1/chat/completions

Both completion endpoints accept stream, temperature, max_tokens (or max_completion_tokens) and stop. Chat requests use the model tokenizer chat template. Text-only prompts are supported. Requests over context/output limits are rejected before admission. A full queue returns HTTP 429. API is local-only and has no authentication; do not proxy it to an untrusted network.

Example request:

    curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"你好"}],"stream":true,"max_tokens":64}'

The WSL service can normally be reached from Windows through http://localhost:8000 while it is running.
