# nano-vLLM 最终性能证据审计

更新：2026-10-10。设备：RTX 5060 Laptop、SM120、约 8 GiB。范围只包括 nano-vLLM 与其 SM120 CUDA 算子集成。

## 官方 vLLM 固定 workload 对照

Qwen3-0.6B 使用固定 revision `c1899de289a04d12100db370d81485cdf75e47ca`、BF16、同一 GPU 和六个相同 workload：batch 1/4/8，prompt 64/256，固定输出 32 token。两边脚本用相同 seed、case 顺序、Python 随机生成法构造 prompt IDs；temperature=0.6、忽略 EOS、1 次预热和 5 次测量。max model length=512、max sequences=8、max batched tokens=4096、GPU memory utilization=0.75，CUDA Graph 均启用，vLLM prefix caching 关闭。

| 项目 | nano-vLLM | 官方 vLLM |
|---|---|---|
| 版本 | 项目提交 `f1f3bb9` | 0.31.0 |
| PyTorch | 2.14.1+cu130 | 2.13.0+cu130 |
| Attention | FlashInfer | FlashAttention |
| RMSNorm | SM120 扩展 | vLLM 默认 |
| Paged KV | 16-token page | vLLM 默认 block 管理 |

六组吞吐几何平均 `vLLM / nano = 0.967×`，支持“这组固定负载下整体接近”的结论。由于 Torch 版本、Attention 后端及 KV 管理不同，这不是 kernel 等价测试，也不构成单个自定义 kernel 的因果归因。vLLM V1 的 CUDA 内存在 worker 进程，客户端侧显存读数不可用，因此记录为 null。

Qwen3-1.7B 固定 workload 进行了五轮交错独立进程 A/B：`nano / vLLM = 0.996×`、`TensorRT-LLM / nano = 1.060×`。这是该固定矩阵上的引擎级比值，不外推到其它模型、形状或软件栈。

## Q/K Norm + RoPE + KV 融合 dispatch

五轮交错独立进程矩阵的输出吞吐变化（SM120 对 Torch）：

| Prompt / Decode | Batch 1 | Batch 8 | Batch 16 | Batch 32 | Batch 64 |
|---|---:|---:|---:|---:|---:|
| 64 / 32 | 未测 | +1.36% | 未测 | +0.40% | −0.41% |
| 256 / 16 | +2.77% | −0.40% | 未测 | −0.28% | −0.40% |
| 64 / 64 | +2.78% | +0.07% | 未测 | −0.30% | +0.07% |

batch>1 的变化很小并会变号；当前证据只支持 **SM120 + Qwen3 `head_dim=128` + Decode batch=1**。Prefill、其它 batch 和未支持形状走框架实现。完整 RMSNorm 形状微基准及 NCU 资源数据在关联仓库 `docs/RTX5060_SHAPE_MAP_20261010.md`。

## Nsight Systems：一次 warm decode 时间线

配置：Qwen3-0.6B BF16、batch 8、prompt 64、output 32、FlashInfer、SM120 RMSNorm、Torch Q/K、16-token KV page。模型加载、编译、warmup 和 graph capture 在采集范围外。

- `cudaGraphLaunch` 31 次，合计 2.378 ms；中位 API 时长 73.5 μs。
- `cudaStreamSynchronize` 129 次，合计 2.164 ms；中位 API 时长 2.87 μs。
- 主要 BF16 GEMM kernel 约占 GPU kernel 总时长 83.6%；自定义 RMSNorm 占 0.3%，Triton RMSNorm 路径约 0.4%。
- 本次证据表明该 batch=8 负载的 GPU 热点在 GEMM；Graph launch 有可见 CPU API 开销，但不足以判定整个推理路径受 launch 限制。

API 调用时间可能与 GPU 执行重叠，累计 API 时间不能直接相加成端到端耗时。这是单次代表性 capture，不是在线延迟 SLA。profile helper 在 `benchmarks/e2e/profile_system_timeline.py`；完整 `.nsys-rep` 和导出文本保留在本机实验归档 `results/nsys/`，大体积 trace 不进 Git。

复跑命令（在 nano-vLLM WSL worktree 执行，按本机模型路径调整）：

```bash
nsys profile --trace=cuda,nvtx,osrt \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=results/nsys/qwen3_b8_p64_o32_system_timeline \
  .venv/bin/python benchmarks/e2e/profile_system_timeline.py \
  --model /home/asus/models/Qwen3-0.6B-verified \
  --requests 8 --input-len 64 --output-len 32 --max-model-len 512 \
  --backend flashinfer --rmsnorm-backend sm120 --qk-norm-rope-backend torch
```

## 两个仓库的对应关系

Kernel 仓库提供可独立构建的 CUDA 扩展、算子基准和正确性测试；nano-vLLM 在 RMSNorm 与 Q/K 热路径接入扩展。Kernel 微基准说明算子本身的延迟变化；只有 nano-vLLM 引擎 A/B 能验证收益是否传递到真实推理。此次未增加 kernel，也未扩大 dispatch。既有 180 项 GPU suite、nano 回归、memcheck 和 racecheck 记录保留；本次没有修改 CUDA 算法。
