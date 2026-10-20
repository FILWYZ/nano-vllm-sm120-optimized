# WSL 环境与验证版本

Windows NVIDIA 驱动通过 WSL2 暴露 GPU；不要在 WSL 内安装 Linux 显示驱动。`CUDA_HOME` 指向本机 CUDA Toolkit。当前本地复测使用 `/usr/local/cuda-13.3` 和 `nvcc 13.3.73`。

## 最终验证环境

2026-10-20 本地复测环境：RTX 5060 Laptop，SM120，8 GiB；WSL2 Ubuntu；Python 3.12.3；PyTorch `2.14.1+cu130`；CUDA Runtime 13.0；Triton 3.8.0；FlashInfer 0.6.6；NVIDIA-SMI 610.57.01，WSL CUDA UMD 13.3。2026-10-10 干净提交复测也使用 PyTorch 2.14.1+cu130，结果为 45 passed 和 5 个 subtests；主线整合候选新加 loader 与服务覆盖后的结果见 [`VALIDATION_20261020.md`](VALIDATION_20261020.md)。

查看本机设备与包版本：

```bash
source .venv/bin/activate
python scripts/check_environment.py
python -m pip check
```

## 历史环境

`requirements-lock-20261007-cu128.txt` 和 [`ROADMAP_RESULTS.md`](optimizations/ROADMAP_RESULTS.md) 保留早期实验快照：PyTorch 2.11.0+cu128、CUDA 12.8、Triton 3.6、23 项测试。它们属于早期提交，不能当作最终主线环境的复测结果。

## 安装

安装项目依赖前，按 PyTorch 官方 cu130 wheel 源安装 PyTorch 2.14.1，再安装 FlashInfer extra。构建配套 SM120 扩展时使用 CUDA Toolkit 13.3 和 `TORCH_CUDA_ARCH_LIST=12.0`。`scripts/setup_wsl.sh` 提供检查与安装辅助；不同驱动、编译器或 wheel 组合需先用 `check_environment.py` 记录实际版本。
