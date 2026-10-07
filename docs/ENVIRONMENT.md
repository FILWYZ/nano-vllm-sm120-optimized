# WSL 环境安装与诊断

This project expects the NVIDIA Windows driver to expose CUDA into WSL2. Do not install a Linux NVIDIA display driver inside WSL. The helper installs the PyTorch CUDA 12.8 wheel and Python project; it does not install or replace the Windows driver or CUDA Toolkit.

From WSL Ubuntu:

    cd /home/asus/nano-vllm
    bash scripts/setup_wsl.sh

The helper checks WSL2, nvidia-smi, CUDA Toolkit nvcc and uv, creates a Python 3.12 virtual environment if needed, installs PyTorch 2.11 CUDA 12.8 and the optional FlashInfer extra, then runs dependency and import diagnostics.

For a concise device/runtime report:

    source .venv/bin/activate
    python scripts/check_environment.py

The checked-in requirements-lock.txt records the full package snapshot from the validated local environment. It is specific to this Linux/Python/CUDA environment; use the setup helper for a fresh install rather than syncing it on unrelated platforms. The SM120 fused operator companion remains optional; the engine has PyTorch fallbacks for unsupported shapes and GPUs.
