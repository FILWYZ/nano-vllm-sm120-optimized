# nano-vLLM 本地 SM120 优化结果

> 面试级逐阶段讲解、实现证据链与优缺点分析见 [`ROADMAP_INTERVIEW_GUIDE_CN.md`](ROADMAP_INTERVIEW_GUIDE_CN.md)。
> 相对 GitHub 上游的三轮直接 A/B 结果和简历口径见 [`M8_UPSTREAM_COMPARISON.md`](M8_UPSTREAM_COMPARISON.md) 与 [`RESUME_M8_CN.md`](RESUME_M8_CN.md)。

## 适用范围

该分支仍然定位为离线、教学型推理引擎。目标运行环境为本地 NVIDIA GeForce RTX 5060 Laptop GPU（计算能力 12.0）、WSL2、PyTorch 2.11.0+cu128、Triton 3.6、FlashInfer 0.6.6，以及 Qwen3-0.6B FP16。在没有重新运行对应版本的基准测试套件之前，不应将这些结果直接推广到其他模型、GPU、上下文长度或在线流量场景。

## 可复现版本索引

| 阶段 | Git 提交 | 标签 | 主要结果 |
|---|---|---|---|
| V0 | `7aa1f25` | `baseline-blackwell-v0` | SM120 SDPA 基线成功运行 |
| M0 | `acafa11` | `m0-measurement` | 固定指标与性能分析器基线 |
| M1 | `88b92d1` | `m1-flashinfer` | 批量分页式 FlashInfer Attention |
| M2 | `7d52dd8` | `m2-sync-free` | 无同步 KV 追加 |
| M3 | `a95c408` | `m3-cudagraph` | 分桶式 Decode CUDA Graph |
| M4 | `677d73c` | `m4-kv-pages` | 选定 16-token KV 页 |
| M5 | `6e5441e` | `m5-continuous-batching` | 混合连续批处理 |
| M6 | `3bcfe31` | `m6-prefix-cache` | 可观测的 LRU + 离线重排序 |
| M7 | 由标签 `m7-final` 记录 | `m7-final` | 贪心/缓冲式采样 |
| M8 | 由标签 `m8-upstream-proof` 记录 | `m8-upstream-proof` | 输出感知 KV Admission + 上游直接 A/B |

每个阶段都可作为回滚点。机器可读的 JSON 和精简版性能分析摘要位于 `benchmarks/results`；体积较大的 Chrome Trace 文件未纳入版本控制。

## M8 相对 GitHub 上游的直接证明

在原仓库 README 定义的 256 请求、133,966 输出 Token 负载中，三轮中位结果为：

| 版本 | 中位 tok/s | 中位耗时 | 相对加速 |
|---|---:|---:|---:|
| Upstream SM120 Compat `7aa1f25` | 247.66 | 540.92 s | 1.00× |
| Optimized M8 Offline | 1,898.00 | 70.58 s | **7.66×** |

该结论相对的是 GitHub 上游 `bb823b3` 派生的最小 SM120/SDPA 可运行基线，不是官方 `vllm-project/vllm`，也不是上游在 RTX 4070 原生 FlashAttention 环境中的性能。最终配置以约 10.1% 峰值 Allocated 显存增量换取 0 次抢占和 0 个重复 Prefill Token。

## 固定测试矩阵的核心结果

M0 SDPA 与最终 M7 随机采样（`temperature=0.6`）的对比：

| 请求数 | 输入长度 | M0 tok/s | M7 tok/s | 总体加速比 |
|---:|---:|---:|---:|---:|
| 1 | 64 | 27.9 | 198.3 | 7.1x |
| 4 | 64 | 88.2 | 745.4 | 8.5x |
| 8 | 64 | 143.5 | 1,440.2 | 10.0x |
| 1 | 256 | 27.3 | 188.6 | 6.9x |
| 4 | 256 | 86.5 | 683.9 | 7.9x |
| 8 | 256 | 137.2 | 1,107.4 | 8.1x |

这并非某一项优化单独产生的效果，而是从“正确性优先”的 Eager SDPA 路径逐步演进到分页式 Attention、消除同步、CUDA Graph 重放、更小的 KV 页，以及调度器、缓存和采样优化后的累积成果。

最终贪心模式在 8 个请求、输入长度 64 的场景下达到 1,543.3 tok/s。但由于贪心采样与随机采样的语义不同，因此该结果单独列出。

## 每个里程碑验证了什么

### M0：先测量，再改动

分别测量 Prefill/Decode 吞吐量、TTFT、TPOT、峰值显存、Attention 延迟，并保留性能分析证据。由此识别出 6,736 次分页 KV Gather，以及累计 3.28 GiB 的临时内存分配。

### M1：使用原生分页式 Attention 后端

FlashInfer 消除了每个请求都要执行的连续 KV 重建。Batch-8 吞吐量提升 64–65%，同时 Attention Decode 的 p50 延迟下降 53–57%。

### M2：消除 KV 写入周围的同步

Triton 追加 Kernel 将 `nonzero` 调用次数从 1,807 次降至 15 次，将标量提取次数从 449 次降至 1 次。性能分析器记录的 CPU Self Time 下降 49.7%。

### M3：重放稳定的 Decode 工作负载

分桶式 CUDA Graph 将可见的 TorchDynamo 查找和矩阵乘法调用减少了 92% 以上。Batch-8、输入长度 64 时，吞吐量达到 1,518.5 tok/s；跨填充分桶和页边界的 48/48 个 Token 均精确一致。

### M4：以低于 5% 的速度代价换取更高的可用 KV 容量

通过对五种页大小进行隔离扫描，最终选定 16-token 页。该方案将预期尾部浪费降低 94.1%，使前缀复用粒度细化 16 倍，同时仍满足预先声明的吞吐量预算。

### M5：在新 Prompt 到达时保护 Decode

混合 Decode+Prefill 批处理将特设在线实验中的最大 Token 间隔从 82.33 ms 降至 18.95 ms（下降 77%），且离线吞吐量没有回退。

### M6：让缓存行为显式且可观测

LRU/FIFO 指标可展示命中、冲突与逐出情况。稳定的离线前缀聚类将 Prefill Token 数减少 42.9%，把命中率从 60.0% 提高到 92.3%，同时严格保持原始输出顺序不变。

### M7：请求确定性输出时，避免执行随机采样工作

在 Qwen3 词表规模下，贪心采样器的 p50 延迟比随机采样低 65–74%，并使选定端到端工作负载的性能最高提升 7.2%。

## 正确性门禁

最终测试套件包含 23 项全部通过的 CPU/GPU 测试，覆盖：

- SDPA Packed/Prefix/Decode 参考路径；
- FlashInfer Ragged、Paged、Mixed 和 GQA Attention；
- Triton KV 写入与填充槽位；
- Graph/Eager 输出一致性与页边界；
- 可变页分配与前缀哈希复用；
- 混合调度器的公平性与消融行为；
- LRU/FIFO/冲突语义；
- 贪心、随机缓冲和混合采样行为。

`uv pip check` 显示已安装的 63 个软件包全部兼容。

## 如何解读这些指标

- 输出 tok/s 是用户可感知的离线吞吐量，但不能单独用于诊断瓶颈。
- Decode tok/s 和 TPOT 用于识别对显存访问或 Kernel 启动开销的敏感性。
- TTFT 和 Prefill Token 数用于揭示 Prompt 与缓存带来的影响。
- 最大 Token 间隔是交错到达场景下衡量调度公平性的信号。
- 缓存 Token 命中数能够证明模型计算确实被跳过；仅看命中率可能会掩盖请求大小的差异。
- 峰值已分配/已预留显存限定了容量上限，而预期尾部浪费用于估算其中有多少容量真正可用。
- 算子调用次数和 CPU Self Time 用于确认加速背后的因果机制。

## 标准复现实验

```bash
cd /home/asus/projects/nano-vllm-baseline
source .venv/bin/activate
MODEL=/mnt/c/Users/ASUS/Documents/ChatGPT/算子/models/Qwen3-0.6B

python -m unittest discover -s tests -v
uv pip check
python -m benchmarks.end_to_end \
  --model "$MODEL" --backend flashinfer --block-size 16 \
  --output benchmarks/results/reproduction.json
```

如需复现某一项具体结论，请使用对应里程碑文档中给出的复现命令。不要将冷启动运行结果与已完成 Shape Warmup 的 JSON 结果直接比较。

## 尚存局限（如实说明）

- 结果仅覆盖一个小型 Qwen 模型和一块 8 GiB 笔记本 GPU。
- 当前 WSL 驱动路径无法使用 CUPTI CUDA Activity；应以 CUDA Event 和同步后的端到端计时为准。
- FlashInfer 的规划阶段仍保留少量主机/设备元数据复制。
- Prompt KV 尚未按 Chunk 进行惰性分配。
- 本项目尚未宣称支持或完成 FP8 KV、投机解码、张量并行、top-k/top-p 或多模型评估。

以上内容是经测量后确定的下一步方向，而不是 M0–M7 基线中未公开的完成标准。
