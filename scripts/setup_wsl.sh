#!/usr/bin/env bash
set -eo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.3}"
export CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/lib/wsl/lib"
export TORCH_CUDA_ARCH_LIST=12.0

if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "This setup helper is for WSL2 Ubuntu." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU is not exposed to WSL2. Install/update the NVIDIA driver on Windows; do not install a Linux NVIDIA driver in WSL." >&2
  exit 1
fi
if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "CUDA Toolkit nvcc not found at $CUDA_HOME/bin/nvcc." >&2
  echo "Install a CUDA Toolkit in WSL2; keep the Windows driver as the driver provider." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv for your Linux user, then rerun this script." >&2
  exit 1
fi

cd "$PROJECT_DIR"
uv venv --python 3.12
source .venv/bin/activate
uv pip install "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install -e ".[flashinfer]"
python scripts/check_environment.py
uv pip check
python -m compileall -q nanovllm
echo "Environment ready. Activate with: source $PROJECT_DIR/.venv/bin/activate"
