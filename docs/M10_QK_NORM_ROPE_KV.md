# M10｜SM120 Q/K Norm + RoPE + Paged KV Store 融合

## 1. 为什么选择这个融合边界

Qwen3 的 Attention 在投影后依次执行逐 Head 的 Q/K RMSNorm、RoPE 和
Paged KV 写入。如果只把 RMSNorm 当作独立算子优化，Decode 热路径仍然
存在多个 Kernel Launch，并且写入 KV Cache 前需要再次读取旋转后的 K。

M10 将融合边界扩展到 KV Store：一个 CUDA Kernel 同时完成 Q/K
RMSNorm、半分式 RoPE、旋转后 K 写入和 V 写入；Attention 计算本身仍交给
FlashInfer，避免把模型数学、缓存管理和 Attention 后端过度耦合。

快速路径有意限制在已验证形状：SM120、Qwen3 `head_dim=128`、连续的
FP16/BF16/FP32 Tensor、int64 Positions、FP32 RoPE Cache 和 int32
`slot_mapping`。不满足条件时继续使用 PyTorch 路径。

## 2. CUDA 实现

- 每个 warp 负责一个 128 元素 Head，每个线程处理四个元素。
- RMSNorm 使用 FP32 累加和 warp shuffle reduction，不使用 Block Barrier。
- 为保持原始计算顺序，RMSNorm 结果先舍入到激活 dtype，再进行 FP32 RoPE。
- K Head 将旋转后的 K 和对应 V 直接散写到 Paged KV Cache。
- PyTorch Custom Operator Schema 显式声明 Q、K、K Cache 和 V Cache 的
  Mutation，支持 FakeTensor、`torch.compile(fullgraph=True)` 和 CUDA Graph。

独立 CUDA 仓库同时保留 one-warp/block 与 four-warps/block 两个版本，用于
复现实验和解释 Occupancy 不等于端到端性能。

## 3. 失败实验与最终调度

早期版本对 Prefill 和 Decode 全局启用自定义 Kernel。虽然独立 Q/K
Norm+RoPE 微基准相对等价 `torch.compile` 链路快 4.9–8.1×，部分 Prefill
负载仍回退最高 5.42%。原因是算子级收益会受到网格形状、Launch、GEMM、
Attention 和 Python 调度占比影响，不能直接外推到完整推理。

four-warps/block 版本在 NCU 中达到 100% 理论 Occupancy，独立 Kernel
时间也略低，但没有产生稳定的端到端收益。这说明小网格 Decode 中，单纯
提高 Occupancy 不是充分优化条件。

最终采用保守调度：

| 运行条件 | 执行后端 |
|---|---|
| SM120、Qwen3 `head_dim=128`、Decode batch=1 | one-warp/head 融合 CUDA Kernel |
| Prefill | PyTorch Inductor |
| Decode batch > 1 | PyTorch Inductor |
| 不支持的设备、形状、dtype 或内存布局 | PyTorch Fallback |

该策略优先保留可重复的真实收益，而不是扩大自定义 Kernel 的启用范围。

## 4. 最终 A/B

环境：RTX 5060 Laptop SM120、Qwen3-0.6B BF16、PyTorch 2.11.0+cu128、
FlashInfer、16-token KV Page、CUDA Graph Decode。

每个后端运行五个独立 Python 进程，A/B 顺序交替；每个负载包含一次预热和
两次正式测量，最终对各进程均值取中位数。

| 请求数 | 输入长度 | Torch Decode | SM120 Decode | Decode 变化 | 输出吞吐变化 |
|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 217.67 tok/s | 221.53 tok/s | **+1.77%** | **+1.24%** |
| 1 | 256 | 212.94 tok/s | 217.85 tok/s | **+2.31%** | **+1.72%** |

batch 4/8 时两组配置均执行同一 Inductor 路径；原始数据中的跨进程差异仅
视为测量噪声，不归因于自定义 Kernel。

机器可读汇总位于：
`benchmarks/results/qk_norm_rope_final_dispatch_ab_summary.json`。

## 5. 验证证据

- CUDA 算子仓库：180 项测试通过。
- nano-vLLM：41 项测试和 5 个 subtests 通过。
- 自定义 Mutation Schema 通过 FakeTensor 和 Fullgraph Capture。
- 真实引擎 CUDA Graph Capture/Replay 通过。
- Compute Sanitizer memcheck：0 errors。
- Compute Sanitizer racecheck：0 hazards。
- NCU 与 SASS 确认使用 warp shuffle reduction 和硬件 reciprocal square
  root，生产 Kernel 不包含 Block Barrier。

## 6. 复现命令

```bash
python -m benchmarks.run_qk_norm_rope_ab \
  --model /path/to/Qwen3-0.6B \
  --runs 5 \
  --output-dir benchmarks/results/qk_norm_rope_final_dispatch_ab

python -m benchmarks.summarize_qk_norm_rope_ab \
  benchmarks/results/qk_norm_rope_final_dispatch_ab \
  --output benchmarks/results/qk_norm_rope_final_dispatch_ab_summary.json
```

## 7. 结论边界

公开结论只适用于上述硬件、模型、软件环境和负载。该结果不表示自定义 CUDA
在所有 Shape 上都优于 Inductor，也不能作为相对官方 vLLM、其他 GPU 或其他
模型的通用性能结论。
