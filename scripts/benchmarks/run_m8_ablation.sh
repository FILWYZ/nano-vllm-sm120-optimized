#!/usr/bin/env bash
set -euo pipefail

root=/home/asus/projects/nano-vllm-baseline
python_bin="$root/.venv/bin/python"
runner="$root/benchmarks/e2e/upstream_ab.py"
model="/mnt/c/Users/ASUS/Documents/ChatGPT/算子/models/Qwen3-0.6B"

for block_size in 16 64 256; do
    for policy in mixed prefill_first; do
        extra=()
        if [[ "$policy" == prefill_first ]]; then
            extra+=(--disable-mixed-batching)
        fi
        output="$root/benchmarks/results/m8_ablation/block${block_size}_${policy}.json"
        echo "START block=$block_size policy=$policy"
        cd "$root"
        PYTHONPATH="$root" "$python_bin" "$runner" \
            --model "$model" \
            --variant "m7_block${block_size}_${policy}" \
            --suite fixed \
            --requests 64 \
            --input-len 512 \
            --output-len 256 \
            --max-model-len 1024 \
            --gpu-memory-utilization 0.9 \
            --block-size "$block_size" \
            --seed 880064 \
            --output "$output" \
            "${extra[@]}"
        echo "DONE block=$block_size policy=$policy"
    done
done
