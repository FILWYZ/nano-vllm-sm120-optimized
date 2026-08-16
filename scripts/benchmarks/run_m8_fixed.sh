#!/usr/bin/env bash
set -euo pipefail

baseline_root=/home/asus/projects/nano-vllm-baseline
upstream_root=/home/asus/projects/nano-vllm-upstream-compat
python_bin="$baseline_root/.venv/bin/python"
runner="$baseline_root/benchmarks/e2e/upstream_ab.py"
model_path="/mnt/c/Users/ASUS/Documents/ChatGPT/算子/models/Qwen3-0.6B"

run_one() {
    local variant=$1
    local repeat=$2
    local requests=$3
    local input_len=$4
    local seed=$((810000 + repeat * 10000 + requests * 100 + input_len))
    local root block_size output
    if [[ "$variant" == upstream_compat ]]; then
        root=$upstream_root
        block_size=256
    else
        root=$baseline_root
        block_size=16
    fi
    output="$baseline_root/benchmarks/results/m8_ab/${variant}_r${repeat}_q${requests}_i${input_len}.json"
    echo "START variant=$variant repeat=$repeat requests=$requests input=$input_len"
    cd "$root"
    PYTHONPATH="$root" "$python_bin" "$runner" \
        --model "$model_path" \
        --variant "$variant" \
        --suite fixed \
        --requests "$requests" \
        --input-len "$input_len" \
        --output-len 32 \
        --seed "$seed" \
        --block-size "$block_size" \
        --gpu-memory-utilization 0.75 \
        --output "$output"
    echo "DONE variant=$variant repeat=$repeat requests=$requests input=$input_len"
}

for repeat in 1 2 3; do
    if [[ "$repeat" -eq 2 ]]; then
        variants=(optimized_m7 upstream_compat)
    else
        variants=(upstream_compat optimized_m7)
    fi
    for spec in "1 64" "4 64" "8 64" "1 256" "4 256" "8 256"; do
        read -r requests input_len <<< "$spec"
        for variant in "${variants[@]}"; do
            run_one "$variant" "$repeat" "$requests" "$input_len"
        done
    done
done

nvidia-smi --query-gpu=name,temperature.gpu,power.draw --format=csv,noheader
