# nano-vLLM 优化项目：简历表述

## 推荐项目名称

**面向 NVIDIA Blackwell SM120 的 nano-vLLM 推理引擎优化**

## 推荐简历版本

基于开源教学项目 `GeeeekExplorer/nano-vLLM`，针对 RTX 5060 Laptop（SM120）完成 PyTorch/Triton/FlashInfer 适配，构建覆盖吞吐、TTFT、TPOT、显存、算子计数和正确性对拍的 M0–M8 可复现实验体系。

- 实现 FlashInfer Paged Attention、Triton 无同步 KV Append、分桶式 CUDA Graph、可变 KV Page、Mixed Continuous Batching、LRU Prefix Cache、Sampling Fast Path 及输出感知 KV Admission Control；
- 在 Qwen3-0.6B FP16、原仓库 README 的 256 请求/133,966 输出 Token 负载中，相对 GitHub 上游 `bb823b3` 的 SM120 最小兼容基线，将中位吞吐从 **247.66 提升到 1,898.00 tok/s（7.66×）**，中位运行时间从 **540.92 降至 70.58 秒（-86.95%）**；
- 在 6 组固定 Batch/Prompt 矩阵中全部胜出，中位加速比几何平均 **8.23×**；通过输出 KV 预留将代表性长负载的 Scheduler 抢占从 **15 降至 0**、Prefill Token 从 **42,949 降至 32,768**；
- 将 Decode 热路径 `nonzero` 调用从 **1,807 降至 15**、GPU 标量提取从 **449 降至 1**，CUDA Graph 将可见模型 Dispatch 减少 **92%+**；建立 **23 项 CPU/GPU 回归测试**，并保留阶段 Git Tag、原始 JSON、消融结果与复现文档。

## 一句话面试介绍

我不是简单给 nano-vLLM 换了一个 Kernel，而是先在 SM120 上建立正确性优先基线，再用 Profiler 逐步消除 KV 重建、Host 同步和 Decode Dispatch；最后复现原仓库自己的长负载时发现高压调度回退，并通过输出感知 KV Admission 将抢占和重复 Prefill 归零，获得 7.66× 的三轮中位加速。

## 必须保留的限定

- 写“nano-vLLM 教学项目”，不要写成“官方 vLLM”；
- 写“相对 SM120 最小兼容基线”，不要写成“普遍比原版快 7.66×”；
- 写明 RTX 5060 Laptop、Qwen3-0.6B FP16 和 README 负载；
- 峰值 Allocated 显存增加约 10.1%，被追问时应主动说明这是 Graph/Workspace/KV 预留的代价；
- 不要将 Greedy 与随机采样数字混在一起。

## 不推荐写法

> 将 vLLM 性能提升 8 倍。

这会被理解为优化了官方 `vllm-project/vllm`，与事实不符。

## 面试追问答法

**为什么基线不是未经修改的上游？**  
上游固定 Commit 依赖 FlashAttention，而当前 SM120/PyTorch 组合没有已验证可用路径。我只增加 SDPA Fallback 和正确性测试形成最小 Compat；未经修改版本不能运行属于兼容性优势，但性能加速只相对可运行 Compat 计算。

**7.66× 是否全是你的创新？**  
不是。它同时包含从 SDPA Eager 兼容路径恢复到 Paged Attention/CUDA Graph 的性能，以及我新增的 Mixed 调度观测、细粒度 KV Page、缓存指标、采样 Fast Path 和输出感知 Admission。简历中明确限定为相对 SM120 Compat，不把它描述成相对上游原生 FlashAttention 的纯算法加速。

**为什么第一次长负载反而更慢？**  
短矩阵没有暴露 KV 容量压力。256 个长输出请求使 16-token Page 频繁申请新页并触发 Preemption/Recompute。修复同轮抢占状态错误后，我增加输出长度感知的 KV 预留，让容量不足请求在 Admission 阶段等待，最终将抢占和重复 Prefill 降到零。

**代价是什么？**  
最终配置峰值 Allocated 显存约增加 10.1%，且预留策略降低瞬时入场并发；它适合已知 `max_tokens` 的离线批处理，不应直接替代在线 Mixed Batching。
