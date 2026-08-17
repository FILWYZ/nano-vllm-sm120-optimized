# M9｜SM120 fused RMSNorm 集成

## 目标

验证独立 CUDA 微基准中的收益能否转化为 nano-vLLM 端到端收益。该阶段不改变
Attention、KV Cache 或 Scheduler，只替换存在 residual 的 RMSNorm 热路径。

## 实现

- `Config.rmsnorm_backend` 支持 `torch`、`sm120` 和 `auto`；默认保持 `torch`。
- `sm120` 路径使用与 FlashInfer/vLLM 风格一致的 in-place 语义：
  `residual += input`，随后用归一化结果覆盖 `input`。
- 无 residual 的首层 Norm、Q/K Norm 继续使用原 PyTorch 路径。
- 不满足 CUDA、contiguous 等条件时回退 PyTorch；非 1024 hidden size 在扩展内部
  回退通用 V2 kernel。
- 自定义 kernel 可以被 Decode CUDA Graph 捕获和重放。

## 正确性

新增 FP16/BF16、tokens 1/16/64/256 的集成对拍，覆盖 PyTorch 与 SM120 后端；
最终 `python -m pytest -q` 为 32 项测试和 5 个 subtests 全部通过。

## A/B 协议

- GPU：NVIDIA GeForce RTX 5060 Laptop，SM120；
- 模型：Qwen3-0.6B FP16；
- Attention：FlashInfer；KV Page：16；CUDA Graph 开启；
- 形状：请求数 1/4/8，输入长度 64/256，输出长度固定 32；
- 每个后端 3 个独立 Python 进程，每个形状 1 次 warmup、2 次测量；
- 运行顺序交错为 Torch→SM120、SM120→Torch、Torch→SM120；
- 每个后端对 3 个进程的形状均值取中位数。

## 端到端结果

| Requests | Input | Torch output tok/s | SM120 output tok/s | Output 变化 | Decode 变化 | TPOT 变化 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 195.46 | 201.33 | +3.00% | +1.22% | -1.20% |
| 4 | 64 | 761.69 | 790.29 | +3.75% | +2.37% | -2.31% |
| 8 | 64 | 1,475.46 | 1,504.73 | +1.98% | +1.77% | -1.74% |
| 1 | 256 | 194.47 | 198.28 | +1.96% | +0.60% | -0.60% |
| 4 | 256 | 680.96 | 685.28 | +0.63% | +0.74% | -0.73% |
| 8 | 256 | 1,127.90 | 1,128.33 | +0.04% | -0.14% | +0.14% |

机器可读汇总位于 `benchmarks/results/rmsnorm_ab_summary.json`。

## 结论边界

微基准中自定义 kernel 相对 FlashInfer 的 1.9–13.2% 延迟下降，没有等比例转化为
系统收益。原因是完整推理仍包含 GEMM、Attention、采样、调度及 CPU 开销；随着输入
长度和 batch 增大，RMSNorm 占比下降。安全结论是：在本机固定矩阵中，5/6 个形状的
Decode 吞吐提高 0.60–2.37%，一个高 batch 长输入形状落在约 0.14% 的噪声范围内。

这项结果用于证明“算子优化 → 推理引擎集成 → 端到端验证”的完整闭环，不应表述为
nano-vLLM 整体稳定提升 3.75%，也不能外推到其他模型和 GPU。

## 复现

分别运行 3 个独立进程，并交错后端顺序：

```bash
python -m benchmarks.e2e.end_to_end \
  --model /path/to/Qwen3-0.6B --backend flashinfer \
  --rmsnorm-backend torch --block-size 16 \
  --warmup-repeats 1 --repeats 2 \
  --output benchmarks/results/rmsnorm_ab_torch_1.json

python -m benchmarks.e2e.end_to_end \
  --model /path/to/Qwen3-0.6B --backend flashinfer \
  --rmsnorm-backend sm120 --block-size 16 \
  --warmup-repeats 1 --repeats 2 \
  --output benchmarks/results/rmsnorm_ab_sm120_1.json

python -m benchmarks.analysis.summarize_rmsnorm_ab
```
