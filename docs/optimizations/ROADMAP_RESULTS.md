# nano-vLLM｜SM120 离线推理优化路线与结果

## 1. 项目定位与结论边界

本项目是基于 GeeeekExplorer/nano-vLLM 的离线、教学型推理引擎优化分支，目标是在 RTX 5060 Laptop（Blackwell SM120）上建立可运行、可解释、可复现的 LLM 推理路径。

已验证环境：

- GPU：NVIDIA GeForce RTX 5060 Laptop，Compute Capability 12.0，8 GiB
- OS/Runtime：WSL2、Python 3.12、PyTorch 2.11.0+cu128、Triton 3.6、FlashInfer 0.6.6
- Model：Qwen3-0.6B FP16

性能结论只适用于上述硬件、软件、模型和负载。

本文的 7.66× 加速是相对 GitHub 上游 commit `bb823b3` 派生的 **SM120 最小可运行 SDPA 兼容基线**，不是相对官方 `vllm-project/vllm`，也不是相对上游在 RTX 4070 原生 FlashAttention 环境的结论。

## 2. 可复现版本索引

| 阶段 | Commit / Tag | 优化目标 |
|---|---|---|
| V0 | `7aa1f25` · `baseline-blackwell-v0` | 建立 SM120 SDPA 兼容基线 |
| M0 | `acafa11` · `m0-measurement` | 固定指标、Profiler 和验收门禁 |
| M1 | `88b92d1` · `m1-flashinfer` | 接入 FlashInfer Paged Attention |
| M2 | `7d52dd8` · `m2-sync-free` | 消除 KV 写入路径中的 Host 同步 |
| M3 | `a95c408` · `m3-cudagraph` | 对稳定 Decode Shape 使用 CUDA Graph |
| M4 | `677d73c` · `m4-kv-pages` | 通过页大小扫描选择 16-token KV Page |
| M5 | `6e5441e` · `m5-continuous-batching` | 引入 Decode/Prefill 混合连续批处理 |
| M6 | `3bcfe31` · `m6-prefix-cache` | 建立可观测的 LRU/FIFO Prefix Cache |
| M7 | `56fc371` · `m7-final` | 优化 Greedy、随机和缓冲式采样 |
| M8 | `e92cf71` · `m8-upstream-proof` | 输出感知 KV Admission 与上游直接 A/B |

每个阶段均可通过 Git Tag 回滚。

阶段原始 JSON 和大体积 Trace 不纳入版本控制。

## 3. M8 最终 A/B 结果

测试负载复现原仓库 README：

- 256 个请求
- 142,827 个输入 Token
- 133,966 个输出 Token
- 随机输入/输出长度 100–1024
- `seed=0`
- `temperature=0.6`
- `ignore_eos=True`
- 三轮独立运行取中位数

| 版本 | 中位吞吐 | 中位耗时 | 峰值 Allocated 显存 |
|---|---:|---:|---:|
| 上游 SM120 SDPA Compat | 247.66 tok/s | 540.92 s | 5.96 GiB |
| 优化版 M8 Offline | **1,898.00 tok/s** | **70.58 s** | 6.56 GiB |

结论：

- 吞吐提升 **7.66×**（+666.4%）。
- 端到端耗时降低 **86.95%**。
- M8 最终配置抢占 **0 次**，重复 Prefill Token **0 个**。
- 峰值 Allocated 显存增加约 **10.1%**，主要来自 CUDA Graph、FlashInfer Workspace/Metadata 和 Decode KV 预留。

M8 的离线配置为 Page 64、Prefill-first 和 Output KV Reservation；M7 的固定矩阵仍使用 Page 16 与随机采样，二者不混用。

## 4. M0–M7 优化证据链

每个里程碑均按“瓶颈定位 → 定向实现 → 正确性门禁 → 局部/端到端指标”验证。

### M0｜先测量，再改动

- **问题**：原始路径缺少可拆解指标，无法判断瓶颈来自 Attention、同步、调度还是显存分配。
- **实施**：增加 Prefill/Decode 吞吐、TTFT、TPOT、峰值显存、Attention 延迟和 Profiler 摘要。
- **证据**：识别出 6,736 次分页 KV Gather，以及累计 3.28 GiB 临时 CUDA 分配。
- **方向**：优先消除连续 KV 重建，再处理同步和 Kernel Dispatch。

### M1｜FlashInfer Paged Attention

- **问题**：SDPA 路径需要逐请求把分页 KV 重建为连续 Tensor。
- **实施**：接入 FlashInfer 批量 Paged Attention，保留 SDPA 作为参考路径。
- **证据**：Batch-8 吞吐提升 64–65%，Attention Decode p50 延迟下降 53–57%。
- **方向**：继续减少 KV 写入和元数据路径的 Host 参与。

### M2｜无同步 KV Append

- **问题**：KV 追加依赖 `nonzero`、标量提取和 Host/Device 同步。
- **实施**：使用 Triton Kernel 批量写入 KV，并复用预计算的槽位信息。
- **证据**：Profiler 中 `nonzero` 从 1,807 次降至 15 次，标量提取从 449 次降至 1 次，CPU Self Time 下降 49.7%。
- **方向**：将稳定 Decode Shape 固化，降低 Python/Dispatcher 开销。

### M3｜分桶式 CUDA Graph

- **问题**：Decode 热路径反复触发 TorchDynamo 查找和矩阵运算 Dispatch。
- **实施**：按 Batch/Token Shape 分桶并捕获可重放的 CUDA Graph。
- **证据**：可见 Dispatch 减少 92% 以上；Batch-8、输入长度 64 时吞吐达到 1,518.5 tok/s；48/48 个跨页边界 Token 对拍一致。
- **方向**：在 Graph Workspace 与显存容量之间做配置权衡。

### M4｜KV Page 大小选择

- **问题**：Page 过大增加尾部浪费，过小增加页表和管理开销。
- **实施**：隔离扫描 16/32/64/128/256-token Page。
- **证据**：最终选择 16-token Page；预期尾部浪费降低 94.1%，Prefix 复用粒度细化 16 倍，速度代价控制在 5% 以内。
- **方向**：在长输出高压场景引入输出长度感知的容量控制。

### M5｜Mixed Continuous Batching

- **问题**：新 Prompt 到达时，纯 Decode 调度会造成 Prefill 等待；优先 Prefill 又会阻塞 Decode。
- **实施**：将 Decode 和 Chunked Prefill 放入统一调度轮次，并设置 Chunk 上限和公平策略。
- **证据**：特设在线到达实验的最大 Token Gap 从 82.33 ms 降至 18.95 ms，下降 77%；离线吞吐未出现回退。
- **方向**：在未知输出长度和高显存压力下增加 Admission 级别的容量判断。

### M6｜Prefix Cache 可观测化

- **问题**：仅有缓存命中率无法解释命中、冲突、逐出与实际节省的 Prefill。
- **实施**：加入 LRU/FIFO 策略、Hash Collision 统计、Hit/Miss/Eviction 指标和离线前缀重排。
- **证据**：离线前缀聚类使 Prefill Token 数减少 42.9%，命中率从 60.0% 提升至 92.3%，原始输出顺序保持不变。
- **方向**：将缓存收益与调度 Admission、请求长度预测联合建模。

### M7｜Sampling Fast Path

- **问题**：确定性输出仍执行完整随机采样流程。
- **实施**：为 Greedy、随机采样和混合 Batch 提供独立 Fast Path，并进行输出对拍。
- **证据**：Qwen3 词表规模下，Greedy Sampler p50 延迟降低 65–74%；选定端到端负载最高提升 7.2%。
- **方向**：在统一采样接口下继续扩展更多采样策略。

## 5. 固定形状矩阵

M0 SDPA 与 M7 随机采样（`temperature=0.6`）的 6 组中位吞吐对比：

| 请求数 | 输入长度 | M0 tok/s | M7 tok/s | 加速比 |
|---:|---:|---:|---:|---:|
| 1 | 64 | 27.9 | 198.3 | 7.1× |
| 4 | 64 | 88.2 | 745.4 | 8.5× |
| 8 | 64 | 143.5 | 1,440.2 | 10.0× |
| 1 | 256 | 27.3 | 188.6 | 6.9× |
| 4 | 256 | 86.5 | 683.9 | 7.9× |
| 8 | 256 | 137.2 | 1,107.4 | 8.1× |

6 组矩阵的提升来自 M1–M7 的累积效果，不能归因于单一 Kernel。

Greedy 结果与随机采样语义不同，单独记录，不与上述表格混合。

## 6. 正确性与工程门禁

- 23 项 CPU/GPU 测试全部通过，覆盖 SDPA、FlashInfer、Triton KV 写入、CUDA Graph/Eager 对拍、页边界、Prefix Cache、混合调度和采样路径。
- `uv pip check`：63 个软件包全部兼容。
- 三轮长负载均生成预期的 133,966 个输出 Token。
- 端到端结果同时记录吞吐、耗时、显存、抢占、Prefill Token 和输出摘要，避免只报告单一 tok/s。

## 7. 标准复现

```bash
cd /home/asus/projects/nano-vllm-baseline
source .venv/bin/activate
MODEL=/path/to/Qwen3-0.6B

python -m unittest discover -s tests -v
uv pip check --python .venv/bin/python

python -m benchmarks.e2e.end_to_end \
  --model "$MODEL" --backend flashinfer --block-size 16 \
  --output benchmarks/results/reproduction.json
```


## 8. 局限与下一步

- 当前只验证 Qwen3-0.6B FP16、单卡 8 GiB RTX 5060 Laptop 和离线长负载。
- WSL 驱动路径未启用 CUPTI CUDA Activity；Profiler 结论以 CUDA Event 和同步后的端到端计时为准。
- FlashInfer 规划阶段仍有少量主机/设备元数据复制，Prompt KV 也尚未实现按 Chunk 的惰性分配。
- 尚未覆盖 FP8 KV、投机解码、张量并行、top-k/top-p 和多模型评估。
