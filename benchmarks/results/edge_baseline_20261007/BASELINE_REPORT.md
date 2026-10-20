# 本机端侧基线报告
日期：2026-10-07
设备：NVIDIA GeForce RTX 5060（SM120，8.0 GiB），Intel Core i7-14700KF，WSL2 Ubuntu 24.04
模型：Qwen3-0.6B（/home/asus/models/Qwen3-0.6B，约 606 MiB）

## 配置
- Python 3.12.3，PyTorch 2.11.0+cu128，CUDA Runtime 12.8，Triton 3.6.0，FlashInfer 0.6.6；CUDA Toolkit nvcc 12.8。
- Attention=FlashInfer，RMSNorm=SM120，Q/K Norm+RoPE=PyTorch；max_model_len=512，显存利用率=0.75，KV block=16，启用默认 CUDA Graph 路径。
- 6 种负载，每种预热 1 次、测 5 次，seed=20261007、temperature=0.6，每请求生成 32 tokens。
- 测试前 nvidia-smi 已用显存约 1.65 GiB，主要为桌面/WSL图形进程；PyTorch peak_allocated 不含其他进程。

## 基线结果
均值（E2E 标准差为 5 次样本标准差）；显存是 allocated / reserved。

| 并发 | 输入 | Prefill tok/s | Decode tok/s | 输出 E2E tok/s | E2E SD | batch TTFT | mean TPOT | 峰值显存 GiB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 3,945.9 | 213.9 | 198.0 | 2.1 | 16.3 ms | 4.68 ms | 4.71 / 4.80 |
| 4 | 64 | 16,063.7 | 847.9 | 787.2 | 0.8 | 15.9 ms | 4.72 ms | 4.72 / 4.81 |
| 8 | 64 | 23,777.8 | 1,634.4 | 1,473.7 | 4.6 | 21.5 ms | 4.89 ms | 4.73 / 4.81 |
| 1 | 256 | 15,615.9 | 211.0 | 195.5 | 1.4 | 16.4 ms | 4.74 ms | 4.72 / 4.81 |
| 4 | 256 | 28,930.4 | 799.4 | 670.3 | 2.7 | 35.4 ms | 5.00 ms | 4.74 / 4.83 |
| 8 | 256 | 31,482.5 | 1,474.2 | 1,095.3 | 3.6 | 65.1 ms | 5.43 ms | 4.78 / 4.87 |

E2E 是批次生成 token 总数除以批次总耗时；batch 提升 aggregate throughput，但不能当作单请求速度。

## 结论和边界
- 单请求约 195–198 tok/s；batch=8 aggregate 输出约 1,095–1,474 tok/s。
- 输入 256、batch=8 时 TTFT 均值 65.1 ms；端侧在线使用需控制并发和排队。
- PyTorch 峰值分配 4.78 GiB，但桌面图形进程另占显存；评估 OOM 需看整卡 nvidia-smi。
- 此基线是当前本机 ZIP 源码，非上游 nano-vLLM 或官方 vLLM 对比。项目没有 .git，commit 为 unknown；后续优化前应固定源码版本。

## 复现
```bash
cd /home/asus/nano-vllm
source .venv/bin/activate
export FLASHINFER_WORKSPACE_BASE=/home/asus/.cache/nano-vllm-flashinfer
python -m benchmarks.e2e.end_to_end \
  --model /home/asus/models/Qwen3-0.6B \
  --output benchmarks/results/edge_baseline_20261007/current_engine_baseline.json \
  --backend flashinfer --rmsnorm-backend sm120 --qk-norm-rope-backend torch \
  --max-model-len 512 --gpu-memory-utilization 0.75 \
  --warmup-repeats 1 --repeats 5 --seed 20261007
```
