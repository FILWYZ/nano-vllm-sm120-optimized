# nano-vLLM M0–M7 优化：面试级技术讲解与证据链

> 本文是 `ROADMAP_RESULTS.md` 的详细教学版。目标不是罗列“用了哪些技术”，而是训练你按照大厂系统/推理引擎面试的标准，完整说明：问题如何被发现、为什么选择该方案、代码如何实现、怎样证明结果正确、哪些数据构成性能因果证据，以及下一步优化为何由当前指标自然推出。

## 1. 先建立系统全局视角

一次请求在本项目中的主链路是：

```text
LLM.generate
  -> LLMEngine：接收请求、循环执行、汇总指标
  -> Scheduler：选择本轮 Prefill / Decode 请求
  -> BlockManager：分配、复用或回收 KV Cache 页
  -> ModelRunner：准备 Token、位置、页表及采样元数据
  -> Qwen3 Model：Embedding -> Transformer Layers -> LM Head
  -> Attention Backend：SDPA / FlashInfer
  -> Sampler：贪心或随机采样
  -> Scheduler.postprocess：更新序列状态并决定结束/继续
```

关键实现入口：

| 子系统 | 主要文件 | 责任 |
|---|---|---|
| 配置 | `nanovllm/config.py` | Page Size、批大小、调度与后端开关 |
| 引擎 | `nanovllm/engine/llm_engine.py` | 请求生命周期和指标汇总 |
| 调度器 | `nanovllm/engine/scheduler.py` | Continuous Batching、Chunked Prefill、公平性 |
| KV 管理 | `nanovllm/engine/block_manager.py` | 页分配、Prefix Cache、LRU/FIFO |
| 模型执行 | `nanovllm/engine/model_runner.py` | 输入准备、CUDA Graph、KV Cache 分配 |
| Attention | `nanovllm/layers/attention.py` | SDPA 参考路径、Triton KV 写入 |
| FlashInfer 适配 | `nanovllm/layers/flashinfer_backend.py` | Paged Attention、Plan 与 Workspace |
| 采样 | `nanovllm/layers/sampler.py` | Greedy/Stochastic Fast Path |
| 基准测试 | `benchmarks/` | 微基准、端到端、在线到达、正确性对拍 |

### 1.1 Prefill 和 Decode 为什么必须分开理解

- **Prefill** 一次处理 Prompt 中的多个 Token，矩阵较大，通常更偏计算密集；核心指标是 Prefill tok/s 和 TTFT。
- **Decode** 每个请求每轮只产生一个 Token，矩阵较小但会反复读取不断增长的 KV Cache，通常更偏显存带宽、Kernel Launch 和 CPU Dispatch；核心指标是 Decode tok/s、TPOT 和 Token 间隔。
- 小模型尤其容易被 Python、同步、Kernel Launch 等固定开销支配，因此“换一个更快的 Attention Kernel”不一定等于端到端等比例加速。

### 1.2 面试中如何证明一项优化成立

不能只给一个 tok/s。完整证据链至少包含四层：

1. **正确性证据**：与参考实现数值对拍、输出 Token 完全一致、边界条件测试通过。
2. **机制证据**：被优化的算子调用、同步、分配或调度事件确实减少。
3. **局部性能证据**：CUDA Event 微基准或 Profiler 指标验证局部瓶颈下降。
4. **端到端证据**：固定环境、固定 Shape、Warmup 后的业务指标改善，且未损害其他关键场景。

如果只满足第 4 层，结果可能来自测量噪声；只满足第 3 层，则可能是“微基准赢了、系统反而退化”；没有第 1 层，速度没有工程意义。

## 2. 指标字典：面试时要能解释，不要只会背数字

| 指标 | 含义 | 能回答的问题 | 不能单独回答的问题 |
|---|---|---|---|
| Output tok/s | 单位时间产生的输出 Token | 离线总体吞吐是否提升 | 瓶颈具体在哪里 |
| Prefill tok/s | Prompt 处理吞吐 | Prefill 计算效率 | 在线 Decode 是否被阻塞 |
| Decode tok/s | Decode 阶段吞吐 | 逐 Token 执行效率 | 单请求尾延迟与公平性 |
| TTFT | Time To First Token | 用户多久看到第一个 Token | 后续 Token 是否流畅 |
| TPOT | Time Per Output Token | 平均生成间隔 | 少数极端卡顿是否存在 |
| 最大 Token 间隔 | 单请求最严重的生成停顿 | 到达流量下的 Head-of-Line Blocking | 整体吞吐能力 |
| p50/p95 延迟 | 中位与尾部时延 | 稳态和抖动 | 极少数单点阻塞，需结合 max |
| Peak allocated/reserved | 已分配/预留峰值显存 | 容量上限和 Allocator 压力 | 显存是否被有效利用 |
| Tail Waste | 每个活跃序列末页浪费 | Page Size 的有效容量 | Kernel 是否更快 |
| Cache Token Hits | 实际跳过的 Prefill Token | 缓存节省了多少模型计算 | 仅靠 Hit Rate 会受请求长度影响 |
| Operator Count | 某算子调用次数 | 优化机制是否命中目标 | 单次算子的成本 |
| CPU Self Time | CPU 自身调度/算子开销 | Host 是否成为瓶颈 | GPU Kernel 的完整耗时 |

## 3. M0：先建立可信测量体系

### 3.1 为什么先做 M0

原项目只有聚合 Output tok/s。这个数字无法区分 Attention 慢、Prefill 慢、Decode 慢、Python 慢、同步多，还是显存分配抖动。没有可观测性就直接优化，很容易针对错误瓶颈。

因此 M0 不改推理算法，只建立实验科学所需的基线。这是面试中的第一个加分点：**优化前先定义指标、控制变量和验收标准**。

### 3.2 如何实施

- 在 `LLMEngine.last_metrics` 中记录 Prefill/Decode Token 数、迭代数、耗时、TTFT、平均 TPOT、输出吞吐和显存。
- `benchmarks/end_to_end.py` 固定请求数与 Prompt 长度矩阵，并输出机器可读 JSON。
- `benchmarks/attention_micro.py` 使用 CUDA Event 隔离测量 Attention Prefill/Decode 的 p50/p95。
- `benchmarks/profile_decode.py` 生成 PyTorch Profiler 摘要与 Chrome Trace。
- 每份 JSON 记录 GPU、计算能力、软件版本、显存与 Git Commit，防止跨环境错误比较。
- 每个 Shape 先进行不计时 Warmup，再测稳态结果，排除 `torch.compile`、CUDA Context 和 Allocator 冷启动。

### 3.3 如何证明测量是可信的

- Packed Prefill、Prefix Prefill、Paged Decode 都与 PyTorch SDPA 参考路径对拍。
- 使用 CUDA Event 而不是仅用 CPU 墙钟测 GPU 微基准。
- CUPTI 在当前 WSL 驱动路径不可用，因此报告明确将 CUDA Event 与同步后的端到端时间作为权威数据，而不伪造缺失的 CUDA Activity。
- 旧的 55.56 tok/s Smoke Test 因 Prompt Shape 可变且无 Warmup，被新的固定矩阵正式替代。

### 3.4 M0 得出了什么

- 8 请求、输入 64 时仅 143.5 tok/s。
- Batch-1 Paged Decode Attention 约 1.14 ms，Batch-8 仍约 1.22 ms，说明固定开销明显。
- 8 请求 × 16 输出 Token 的 Profile 出现 6,736 次 `index_select` 和累计 3.28 GiB 临时 CUDA 分配。
- 同时出现 1,792 次 `nonzero`、448 次标量提取和 7,757 次 Copy。

### 3.5 为什么这些指标指向 M1

`index_select` 和 3.28 GiB 临时分配来自每层、每请求把分页 KV 重建为连续 Tensor。它既增加显存流量，也让 SDPA 逐请求调用。所以下一步优先消除 KV Gather，而不是先改调度器或量化。

### 3.6 面试追问

**问：为什么不先上 CUDA Graph？**  
答：当时图内仍包含大量无效 KV Gather 和临时分配。先优化数据布局与 Attention 路径，可以减少需要捕获的工作，并使后续 Profile 更清晰。

**问：为什么不能只测平均吞吐？**  
答：平均值无法反映 TTFT、TPOT、尾延迟、缓存复用和调度公平性，也无法建立加速的因果机制。

## 4. M1：用 FlashInfer 实现原生 Paged Attention

### 4.1 为什么选择该优化

M0 已证明主要浪费来自“Paged KV -> 连续 KV -> SDPA”的重建路径。理想方案应直接消费页表和 KV Cache，而不是继续优化 Gather。

选择 FlashInfer 0.6.6 的原因：

- 原生支持 Batch Paged Prefill/Decode 和 GQA；
- 支持本机 SM120；
- 可保持 PyTorch 2.11.0+cu128 与 Triton 3.6 环境不变；
- 0.6.17 会把依赖整体升级到 PyTorch 2.13/CUDA 13，不利于单变量实验，因此未采用。

### 4.2 如何实施

- `nanovllm/layers/flashinfer_backend.py` 封装 `FlashInferRuntime`，隔离第三方后端细节。
- 普通 Packed Prefill 使用 `BatchPrefillWithRaggedKVCacheWrapper`。
- Prefix/Chunked Prefill 使用 `BatchPrefillWithPagedKVCacheWrapper`。
- Decode 使用 `BatchDecodeWithPagedKVCacheWrapper`，直接读取分页 KV。
- 每进程、每 GPU 共享一个 128 MiB Workspace。
- 同一次模型迭代中所有 Transformer Layer 复用同一份 Batch Plan，避免每层重复规划。
- `auto` 在 SM120 且依赖可用时选择 FlashInfer；SDPA 保留为可读参考实现和正确性 Oracle。

### 4.3 正确性如何验证

- 独立 SM120 Probe 对比 PyTorch SDPA，最大绝对误差为 0.000244。
- FP16 容差固定为 `rtol=atol=2e-3`。
- 覆盖 Ragged 变长 Prefill、Bottom-right Causal Mask 的 Prefix Prefill，以及多请求 GQA Paged Decode。

这里要强调：浮点 Kernel 改变后不一定逐 Bit 相同，所以 Attention 层用数值容差；用户可见的完整生成路径则在后续阶段使用 Token 完全一致门禁。

### 4.4 性能和因果证据

- Batch-8 吞吐提升 64.2%–65.4%。
- Prefill Attention p50 从 0.154 ms 降至 0.032 ms（-79.2%）。
- Decode p50 从 1.139/1.224 ms 降至 0.533/0.521 ms（-53.2%/-57.4%）。
- `index_select` 从 6,736 次、3.28 GiB 累积分配降到 16 次、2.23 MiB。
- Copy 从 7,757 次降到 638 次，CPU Self Time 从 987 ms 降到 557 ms。

这形成完整因果链：KV 重建操作消失 → Attention 微基准下降 → 端到端所有 Shape 提升。

### 4.5 哪些指标指向 M2

Attention 已明显加速，但 Profile 仍有 1,807 次 `nonzero`、449 次标量提取以及每层 Boolean Index KV 写入。这些 Host 可见操作在小模型短 Kernel 场景中可能比 KV Copy 本身更贵，因此下一步消除同步和小分配。

### 4.6 面试追问

**问：Attention 快了 2 倍，为什么端到端没有全部快 2 倍？**  
答：符合 Amdahl 定律。Attention 只是总耗时的一部分；KV 写入、模型层 Dispatch、LM Head、采样和 Host Metadata 仍存在。

**问：引入第三方库最大的工程风险是什么？**  
答：依赖兼容、后端行为差异、Workspace 显存、Plan 开销和版本漂移。因此保留 SDPA Oracle，并锁定版本与环境元数据。

## 5. M2：无同步 KV Append 与 Buffer 复用

### 5.1 为什么选择该优化

M1 后每层 KV 写入仍执行 `int64` 转换、Validity Mask、`torch.any` 与两次 Boolean Index。`torch.any`/标量判断会迫使 GPU 结果回到 Host，破坏异步流水；Boolean Index 还会生成临时 Tensor。

### 5.2 如何实施

- `nanovllm/layers/attention.py` 中的 Triton KV Append Kernel 直接消费 `int32 slot_mapping`。
- Kernel 内判断 `slot == -1` 并跳过 CUDA Graph Padding 行，不在 Python 侧构造 Mask。
- FlashInfer 输出 Buffer 按 Device、Dtype、Shape 缓存，并通过 `out=` 复用。
- Page Column Tensor 按 Device 与页表宽度缓存，避免每轮创建 `arange`。
- 只复用 Shape 稳定的 Buffer，不缓存请求相关 Page Index 和 Length，避免跨页后读到陈旧 Metadata。
- SDPA 路径保留显式 PyTorch 写法，继续承担教学参考与 Oracle 责任。

### 5.3 正确性如何验证

- 真实 Slot 必须写入精确位置。
- `slot=-1` 的 Padding 行不得改动 KV Cache。
- 七项 SM120 GPU 测试覆盖 SDPA、FlashInfer、GQA、Prefix、Paged Decode 和 Triton 写入。

### 5.4 性能和因果证据

- 各固定 Shape 端到端提升 80.1%–97.7%。
- `nonzero` 从 1,807 次降至 15 次。
- 标量提取从 449 次降至 1 次。
- Copy 从 638 次降至 190 次。
- CPU Self Time 从 557 ms 降至 280 ms（-49.7%）。
- KV 相关 `index_select` 分配归零。

这里最强的证据不是单独的吞吐提升，而是“目标操作近乎消失、CPU Dispatch 减半、所有端到端 Shape 同向提升、Padding 正确性测试通过”。

### 5.5 哪些指标指向 M3

M2 后 Profile 主体变成 2,720 次 TorchDynamo Cache Lookup/Compiled Region Prologue 和 1,808 次可见矩阵乘法 Launch。说明瓶颈已从数据搬运转为模型 Dispatch，应针对 Decode 稳定 Shape 做 CUDA Graph。

### 5.6 面试追问

**问：为什么 Buffer 可以跨层复用，不会被覆盖？**  
答：依赖它的后续 Kernel 与下一层写入位于同一 CUDA Stream，Stream 顺序保证前一个消费者完成后才覆盖。但不能跨并发 Stream 无条件复用。

**问：所谓“无同步”是否绝对没有同步？**  
答：不是。它指 KV 写入热路径移除了 Host-visible Scalar Sync；端到端采样和计时仍可能同步，FlashInfer Plan 也仍有少量 Metadata 交互。

## 6. M3：分桶式 Decode CUDA Graph

### 6.1 为什么选择该优化

Decode 每轮每请求只有一个 Token，Shape 在给定 Batch Bucket 内稳定；Qwen3-0.6B 的单个 Kernel 很短，Python 逐层发射的固定开销占比很高。这正是 CUDA Graph 的适用场景。

Prefill 长度动态且组合复杂，因此保持 Eager；只捕获 Decode，降低复杂度和显存池开销。

### 6.2 如何实施

- `ModelRunner.capture_cudagraph()` 为有界 Batch Bucket 捕获 Decode Graph。
- Bucket 不超过 `max_num_seqs`，非 2 次幂上限也显式加入，避免不必要 Padding。
- 每个 Bucket 拥有独立稳定 GPU Buffer、FlashInfer Wrapper、Page Indptr、Page Indices 和 Last Page Length。
- FlashInfer `plan()` 放在 Graph 外，每轮先刷新稳定 Metadata，再 Replay。
- 空余 Bucket 行设置 `context_len=1`、Block 0、`slot=-1`；计算结果丢弃，KV Kernel 不写缓存。
- 每次复制较小 Batch 前清空共享输入 Buffer，防止上一次 Replay 的请求和页表残留。
- `--enforce-eager` 提供同版本受控消融。

### 6.3 正确性如何验证

- 3 个真实请求进入带 Padding 的 Bucket。
- Prompt 长度 255、输出 16，强制跨越 KV 页边界。
- Eager 与 Graph 使用相同 Prompt、Seed、Temperature。
- 48/48 个 Completion Token 完全一致。

这个测试同时覆盖三类高风险错误：陈旧页表、Padding 错写 KV、Graph/Eager 数值偏差导致采样分叉。

### 6.4 性能和因果证据

- 各 Shape 提升 176.4%–251.7%。
- 8 请求、输入 64 达到 1,518.5 tok/s。
- TorchDynamo Lookup 从 2,720 降至 185（-93.2%）。
- 可见 `aten::mm` 从 1,808 降至 128（-92.9%）。
- CPU Self Time 从 280 ms 降至 103 ms（-63.1%）。

注意：矩阵乘法没有真的少算 92.9%，而是被封装在 Graph Replay 内，不再以逐次 Host Launch 的形式出现。面试中必须解释这一点。

### 6.5 代价与下一步信号

- 峰值显存小幅上升到 4.758 GiB Allocated、4.861 GiB Reserved，因为 Graph 使用私有 Memory Pool，Bucket 也持有稳定 Buffer。
- Profile 中 `aten::copy_` 仍约 75 ms，主要是 FlashInfer Plan 和 Metadata 更新。
- 原始 256-token KV 页只是教学项目遗留值，浪费尾页且 Prefix 复用粒度太粗。因此 M4 转向容量与 Page Size 的实证选择。

### 6.6 面试追问

**问：为什么需要 Batch Bucket？**  
答：CUDA Graph 要求地址和执行拓扑稳定，而在线活跃请求数会变。Bucket 用有限数量的稳定 Shape 覆盖动态 Batch，以少量 Padding 换取 Replay。

**问：CUDA Graph 的常见正确性陷阱？**  
答：输入地址变化、残留 Buffer、动态页表未刷新、Padding 写 KV、随机状态和 Capture 内执行不支持的 Host 操作。

## 7. M4：实证选择 16-token KV Page

### 7.1 为什么选择该优化

Page 越大，页表越短、Plan 开销可能越低；但每个活跃序列的最后一页平均浪费约 `(page_size-1)/2` 个 Token，Prefix 也只能按整页复用。Page 越小则相反。因此它是吞吐、Metadata 和有效容量之间的权衡，不能想当然选择最小值。

### 7.2 如何实施

- 配置和调度器支持 16/32/64/128/256 五种 2 次幂页大小。
- FlashInfer/SDPA 本地默认改为 16；原 FlashAttention 兼容路径仍限制 256。
- `Sequence` 页运算从 `Config` 同步，避免模块间 Page Size 不一致。
- JSON 新增 Page Size、物理块数和原始 KV Token 容量。
- `benchmarks/block_size_sweep.py` 为每个候选启动独立进程，避免 CUDA Allocator 与 Graph Pool 污染后续候选。
- 在实验前固定选择规则：**选择几何平均吞吐距最快值不超过 5% 的最小页**。

### 7.3 为什么选择规则必须预先声明

如果看到结果后才决定标准，就容易挑选对结论有利的指标。预注册规则使“性能预算”和“容量目标”可审计，也避免把 0.x% 的运行噪声误当真实胜出。

### 7.4 性能、容量和正确性证据

- Page 64 的几何平均吞吐最快：572.86 tok/s。
- Page 16 为 568.30 tok/s，仅慢 0.8%，满足 5% 预算，因此按规则选中。
- 相比 Page 256，预期尾页浪费从 127.5 降至 7.5 Token（-94.1%）。
- Prefix 复用粒度细化 16 倍。
- 8 个活跃序列估算可用容量从 30,468 提升到 31,588 Token（+3.68%）。
- 最终两轮矩阵所有 Shape 相比 M3 均在 +1.9% 到 -4.0%，未突破性能预算。
- 255 Token Prompt 会跨 16 个页边界，Graph/Eager 仍有 48/48 Token 一致。

M4 不是“Kernel 更快”的优化，而是**以可控吞吐代价换取更低碎片、更细缓存粒度和更高有效容量**。

### 7.5 哪些指标指向 M5

小页允许 Prompt 更细粒度处理，但调度器仍可能让一个长 Prompt 占满 Prefill Budget；只要有 Prefill 可运行，Decode 就整轮等待。这会产生明显 Token Stall，因此下一步必须优化调度公平性，而不是继续追求离线 tok/s。

### 7.6 面试追问

**问：Raw Block Count 为什么不能证明容量更高？**  
答：不同 Page Size 的每块 Token 数不同。应比较 Raw Token Capacity、活跃序列 Tail Waste 和实际可用容量。

**问：为什么使用几何平均？**  
答：不同 Shape 的吞吐量量级不同，几何平均更适合汇总相对性能，避免大数值 Shape 过度支配算术平均。

## 8. M5：Chunked Prefill 与混合 Continuous Batching

### 8.1 为什么选择该优化

原调度策略存在 Head-of-Line Blocking：长 Prompt 到达后，其多个 Prefill Chunk 会优先执行，正在 Decode 的短请求在此期间无法生成 Token。离线吞吐可能仍好看，但在线体验会出现明显卡顿。

FlashInfer Paged Prefill 可以在一个 Packed Batch 中同时表示：Query Length=1 的 Decode 请求，以及 Query Length>1 的 Prompt Chunk，因此不必维护两套模型执行路径。

### 8.2 如何实施

- 新增 `max_prefill_chunk_size=512` 作为每个 Prompt Chunk 的计算上限。
- 新增 `enable_mixed_batching`，默认开启且可用于消融。
- 每轮先调度活跃 Decode，再用剩余序列数与 Token Budget 填入 Prompt Chunk。
- 部分 Prefill 的请求在 Waiting Queue 中轮转，避免队首长请求垄断。
- Mixed Batch 统一走 Paged Prefill Preparation；LM Head 对每个 Query 只选最后一个位置，因此每个 Decode/Chunk 正好生成一个采样项。
- 无等待 Prompt 时仍使用 Decode-only CUDA Graph，保留 M3 收益。
- 指标新增 Mixed/Prefill-only/Decode-only Batch、Prefill Chunk、Preemption 等计数，并正确计入 Mixed Batch 中的 Decode Time。

### 8.3 正确性如何验证

- Decode-first 调度顺序和 Prefill-first 消融行为。
- 多 Waiting Request 的 Round-robin Chunk Cap。
- 一个 Decode Token 与一个未完成 Prompt Chunk 在同批次的 Postprocess。
- Mixed Attention 与独立 SDPA Reference 在 FP16 `2e-3` 容差内一致。
- 原有 Graph、页边界、Prefix 和 KV 写入测试继续通过。

### 8.4 为什么使用最大 Token 间隔而不是只看 p95

实验只有约 20 个 Gap，并刻意注入一次 Arrival Stall。Nearest-rank p95 可能恰好排除那个唯一最大值，从而掩盖真正的 Head-of-Line Blocking。最大 Gap 直接回答“用户最严重会卡多久”。

### 8.5 性能和因果证据

- 最大 Token 间隔从 82.330 ms 降到 18.948 ms（-77.0%）。
- 平均间隔下降 12.8%。
- Arrival 后 Scheduler Step 从 24 降至 20（-16.7%）。
- 实际出现 4 个 Mixed Batch，Prefill-only Batch 从 5 降至 1。
- 固定离线矩阵未回退；小幅正变化被诚实归为运行噪声，而不是算法吞吐提升。

Scheduler Counter 证明四个 Prompt Chunk 确实与四个 Decode Step 重叠；Step 减少和最大 Gap 下降共同建立调度优化的因果链。

### 8.6 哪些指标指向 M6

M4 已提供 16-token Prefix 粒度，但缓存策略仍是隐式 Deque，既没有命中/冲突/逐出指标，也不能解释缓存为何失效。下一步要将 Prefix Cache 变成显式、可观测、可消融的子系统。

### 8.7 面试追问

**问：Continuous Batching 与普通 Dynamic Batching 的差别？**  
答：Dynamic Batching 通常以整个请求/批次为边界聚合；Continuous Batching 在每个迭代边界移入新请求、移出完成请求，并混合不同生命周期阶段，提高资源利用率和在线公平性。

**问：M5 尚未解决什么？**  
答：Prompt KV 仍在 Admission 时为完整 Prompt 预留。Chunking 细化了计算，却没有做到 KV Page 的 Lazy Allocation；后者需与 Admission Control 和部分 Prefix Eviction 一起设计。

## 9. M6：可观测 LRU Prefix Cache 与离线重排序

### 9.1 为什么选择该优化

原实现会在释放的 KV Block 上保留 Hash，但逐出顺序只是 Deque 的隐含结果，没有命名策略与统计指标。即使性能变化，也无法回答是命中、冲突还是逐出造成的。

### 9.2 如何实施

- 新增 `prefix_cache_policy={lru,fifo}`，默认 LRU。
- LRU 命中一个空闲缓存块时，将其移动到 MRU 端；FIFO 命中不改变顺序，用作消融。
- 记录 Query、Block/Token Hit、Miss、Collision Rejection 和 Cached Eviction。
- Hash 命中后仍逐 Token 比较存储块，防止 Hash Collision 导致错误复用。
- `LLM.last_metrics` 暴露每次 Generate 的缓存指标增量。
- `num_kvcache_blocks` 可人为限制缓存大小，建立可控压力实验。
- 离线模式可按 Token Prefix Key 稳定重排序；完成后使用 Original Index -> Sequence ID 映射恢复调用者输出顺序。

### 9.3 为什么重排序只适合离线场景

在线交互请求不能无限等待未来请求以获得更高 Prefix Locality，否则 TTFT 和公平性会受损。离线批处理已知完整请求集合，可以合法地聚类相同前缀。

### 9.4 正确性如何验证

- LRU 命中会改变下一次 Eviction Victim，FIFO 不会。
- 人工注入 Hash Collision，必须计数并拒绝复用。
- FIFO、LRU、LRU+Reorder 三种 GPU 运行的 12 个输出 Token 与原请求顺序完全一致。

### 9.5 性能和因果证据

在 12 个请求、3 个交错 Prefix Family、缓存仅 12 页的压力实验中：

- FIFO 与 LRU 都是 672 个 Prefill Token、18 个 Block Hit、60% Hit Rate。原因是三个等频工作集轮转，而容量约只能容纳两个；LRU 不可能凭策略突破工作集容量。
- LRU+Reorder 将 Prefill Token 从 672 降到 384（-42.9%）。
- Block Hit 从 18 增至 36，Cached Token Hit 从 288 增至 576。
- Eviction 从 30 降至 12（-60%），Hit Rate 从 60.0% 升到 92.3%。
- Generate Time 仅下降 6.7%，因为 Decode、Python、Plan 和 Sampling 固定成本没有减少。
- 随机唯一 Prompt 的固定矩阵在 -3.2% 到 +0.3%，说明未命中场景没有显著回退。

这里最值得面试讲的是“LRU 没赢 FIFO”也被保留。它说明报告没有选择性展示数据，并揭示了 Policy 无法战胜 Capacity/Working-set 限制。

### 9.6 哪些指标指向 M7

Attention、KV 写入和模型 Dispatch 已大幅优化，剩余固定成本包括 Sampling、LM Head、输入 Metadata Copy 和小型 Compiled Region。Qwen3 词表达到 151,936，原采样器无论是否需要确定性输出都会执行 FP32、Softmax、随机噪声和除法，因此确定性场景存在明确冗余。

### 9.7 面试追问

**问：Hit Rate 为什么可能误导？**  
答：它忽略 Block/请求大小。命中一个大 Prefix 和命中一个小块在 Hit Rate 中可能权重相同，应同时报告 Cached Token Hits 和实际 Prefill Token 减少量。

**问：为什么 Hash 后还要比较 Token？**  
答：Hash 不是无碰撞证明。错误复用 KV 会悄悄污染后续所有 Token，因此需要 Collision-safe Verification。

## 10. M7：采样 Fast Path 与最终消融

### 10.1 为什么选择该优化

原采样器始终把完整词表 Logits 转 FP32、计算 Softmax、分配并填充指数噪声、执行除法和 Argmax。即便用户想要确定性 Greedy，也支付完整随机采样成本；在 151,936 词表下该开销可测。

### 10.2 如何实施

- `temperature=0` 明确定义为 Greedy；负数拒绝。
- `Sampler._greedy()` 直接对 Logits 执行编译后的 `argmax`，跳过 FP32 Logits、Softmax、RNG、除法和 Noise Storage。
- 正温度继续使用 Stochastic Gumbel/Exponential 路径，保持语义兼容。
- FP32 Noise Buffer 按 Device 与 Logits Shape 缓存，Warmup 后不再每步重新分配完整词表 Tensor。
- Mixed Batch 中随机行走 Stochastic Path，Greedy 行再用精确 Argmax 覆盖。
- `ModelRunner` 从 CPU Request Metadata 得到 `all_greedy/has_greedy`，避免用 GPU Scalar Sync 判断分支。

项目没有进一步写自定义融合 Softmax/RNG Kernel，因为当前目标仍是教学可读、容易验证，而不是以维护复杂度换取尚未量化的收益。

### 10.3 正确性如何验证

- Greedy 输出与 `logits.argmax` 精确相等。
- 零温度可用、负温度报错。
- Mixed Batch 中 Greedy Row 仍精确。
- Stochastic Noise Buffer 多次调用 Data Pointer 不变，证明复用而非仅代码声明。
- 255 Token Prompt、16-token 页、Padded Graph Bucket 下，Eager/Graph 的 48/48 Token 完全一致。

### 10.4 性能和因果证据

- Batch-1 Sampler p50：0.1684 ms -> 0.0595 ms（-64.7%）。
- Batch-8 Sampler p50：0.1821 ms -> 0.0480 ms（-73.6%）。
- 8 请求、输入 64：1,440.2 -> 1,543.3 tok/s（+7.2%）。
- Batch-1、输入 64 端到端为 -0.1%，说明约 0.11 ms 的局部节省被模型与 Host 固定开销掩盖。
- Stochastic 路径相对 M6 在 -4.5% 到 +1.8%，处于 5% 稳态容忍范围；Buffer Reuse 被定义为分配稳定性优化，不夸大为 Kernel 加速。

### 10.5 如何解释“微基准快 70%，端到端只快 7%”

Sampler 只占总耗时的一小部分。按 Amdahl 定律，即使局部无限加速，系统收益也受该部分原始占比限制。微基准回答“Fast Path 是否真的更快”，端到端回答“用户是否能感知”；两者都需要。

### 10.6 M7 后的下一步指标方向

- 若 Profile 显示 FlashInfer Plan/Metadata Copy 占比继续上升，可从 CPU Scheduler Array 直接构建稳定 Metadata。
- 若长 Prompt Admission 因 KV 预留失败，应做 Lazy KV Allocation 与 Admission Control。
- 若 KV 容量/带宽成为主瓶颈，可实验 FP8 KV，但必须增加 Perplexity、Long-context Accuracy 和输出质量验证。
- 若 Decode 仍由单 Token 串行限制，可做 Speculative Decoding，并报告 Acceptance Rate、Draft Cost 和 Wall Time，而非只报“每步接受 Token 数”。
- Top-k/top-p 需要自定义 CUDA/Triton Kernel 与每请求 RNG State，正确性要覆盖分布、Seed 可复现和 Mixed Sampling。
- 更大模型、长上下文、多并发矩阵是对结论外推前的必要工作。

## 11. M0–M7 的整体结果应如何陈述

在本机 RTX 5060 Laptop GPU、Qwen3-0.6B FP16、固定软件版本与 Shape Warmup 条件下，M0 SDPA 到 M7 随机采样的固定矩阵获得 6.9×–10.0× 累积加速；8 请求、输入 64 从 143.5 提升到 1,440.2 tok/s。Greedy 最终达到 1,543.3 tok/s。

不要说“我把 vLLM 提升了 10 倍”。更准确的面试表述是：

> 我在一个教学型 nano-vLLM 分支上，针对单块 RTX 5060 Laptop GPU 和 Qwen3-0.6B FP16，建立可复现基线后，依次消除了 Paged KV 重建、Host 同步和 Decode Dispatch，并优化 KV 分页、在线调度、Prefix Cache 与采样路径。在固定矩阵中，最终随机采样吞吐相对 M0 提升 6.9×–10.0×；每一步都通过参考实现对拍、边界测试、Profiler 机制证据和端到端指标共同验收。

这段话同时限定了对象、硬件、模型、方法、结果和证据，不会过度外推。

## 12. 当前版本的优点

### 12.1 技术优点

- **证据驱动**：每个里程碑由上一步 Profile 暴露的新瓶颈触发，不是随意堆功能。
- **端到端证据闭环**：局部微基准、算子计数、CPU Self Time、端到端吞吐和正确性门禁互相印证。
- **关键工业机制齐全**：Paged Attention、同步消除、CUDA Graph、Continuous Batching、Prefix Cache 和 Sampling Fast Path 均有简化但真实的实现。
- **适配本机 SM120**：FlashInfer、Triton 与 PyTorch 版本经过实际 GPU 验证。
- **性能突出**：固定矩阵相对 M0 获得 6.9×–10.0× 累积提升。
- **更合理的 KV 利用率**：16-token 页显著降低尾部碎片并提高 Prefix 复用粒度。
- **在线公平性改善**：Mixed Batching 将特设到达实验最大 Token Gap 降低 77%。
- **缓存可解释**：LRU/FIFO、Hit/Miss/Collision/Eviction 指标齐全，能做受控消融。

### 12.2 工程与学习优点

- SDPA 参考路径仍保留，便于学习和对拍。
- 后端适配集中在独立文件，模型与调度器没有被 FlashInfer API 污染。
- 每阶段都有 Git Tag，可回滚并复现实验。
- 23 项 CPU/GPU 测试覆盖 Attention、KV、Graph、Scheduler、Cache 与 Sampling。
- 结果 JSON 带环境与 Commit 信息，减少“在我机器上更快”的不可复现问题。
- 对负结果和边界如“LRU 未胜 FIFO”“M4 不是吞吐优化”进行了诚实记录，这在面试中非常加分。

## 13. 当前版本的缺点与风险

### 13.1 结论外推能力有限

- 只验证 Qwen3-0.6B FP16 和一块 8 GiB Laptop GPU。
- 缺少更大模型、不同 Head Dim/GQA 比例、长上下文和高并发矩阵。
- 主要是离线固定矩阵与一个人工在线到达实验，不等价于真实泊松流量、Burst 和 SLO 压测。

### 13.2 推理能力仍不完整

- 没有 Top-k、Top-p、Min-p、Penalty 等常用采样能力。
- 没有 Speculative Decoding、FP8 KV、量化权重、Tensor Parallel 或 Multi-GPU。
- 没有 LoRA、多模型 Serving、Streaming API、请求取消、超时与优先级。

### 13.3 KV 与调度仍有容量缺陷

- Prompt KV 在 Admission 时仍按完整 Prompt 预留，Chunked Prefill 没有做到 Lazy Page Allocation。
- 缺少完善 Admission Control、Swap/Offload 和部分 Prefix Eviction。
- Prefix Reorder 只适合离线，并且只是稳定词典序 Key，不是 Radix Tree/Trie 或更成熟的 Prefix-aware Scheduler。
- Cache Metadata 仅进程内存在，重启后不持久化，也不能跨 Worker 共享。

### 13.4 CUDA Graph 与后端存在复杂度成本

- 每个 Bucket 的稳定 Buffer 和 Graph Private Pool 增加显存占用。
- FlashInfer `plan()` 仍在 Graph 外，有小量 Host/Device Metadata Copy。
- Bucket Padding 会做少量无效计算；活跃 Batch 变化剧烈时，Bucket 选择与 Capture 成本需要继续评估。
- 强依赖 FlashInfer 0.6.6 的行为与兼容性，升级 PyTorch/CUDA 后需重新验证。

### 13.5 测量体系仍可增强

- 当前 WSL 路径无法采集完整 CUPTI CUDA Activity，因此缺少最完整的 Kernel Timeline 证据。
- 每个固定 Shape 主要只有两次测量，适合本地迭代，但不足以给出严格置信区间。
- 尚未记录功耗、能效、显存带宽利用率、P99/P999、多租户干扰和长时间稳定性。
- Greedy 与 Stochastic 语义不同，二者不能被当作同一质量条件下的纯性能对比。

### 13.6 教学版与工业版的差距

- 代码强调可读性，缺少生产级 RPC/HTTP Server、分布式协调、监控告警和故障恢复。
- Scheduler、Block Manager 和 Backend 的并发/多 Worker 生命周期远比工业 vLLM 简化。
- 缺少完整模型兼容层、算子自动调优、Kernel Fallback 矩阵和跨平台 CI。

## 14. 建议的下一轮优化优先级

按当前证据和本机硬件，建议顺序如下：

1. **M8：真实在线负载与 SLO 基准**。先补 Poisson/Burst Arrival、TTFT/TPOT P50/P95/P99、Goodput，防止继续只优化离线吞吐。
2. **M9：Lazy Prompt KV Allocation + Admission Control**。解决 M5 明确留下的容量与长 Prompt 入场问题。
3. **M10：FlashInfer Metadata/Plan 优化**。重新 Profile，只有当 Copy/Plan 仍是主导时才实施。
4. **M11：Top-k/Top-p 融合采样**。补齐常用语义，同时用分布正确性和 Seed 可复现验收。
5. **M12：FP8 KV Cache**。以吞吐、容量、Perplexity 和长上下文准确率共同决策。
6. **M13：Speculative Decoding**。报告 Acceptance Rate、Draft/Verify Cost、吞吐和尾延迟。
7. **扩展验证矩阵**。至少加入更大 Qwen 模型、不同上下文长度和不同并发，判断优化是否具有一般性。

## 15. 推荐学习与面试训练方式

每个阶段用下面五句话复述，直到不看文档也能讲清：

1. 上一阶段哪个 Profile 数据暴露了瓶颈？
2. 为什么该方案比两个备选方案更适合当前硬件与项目定位？
3. 数据结构、执行路径和关键 API 具体改了什么？
4. 哪个测试证明没有算错，哪个指标证明优化机制确实生效？
5. 哪个剩余热点自然导出下一阶段，而不是凭主观猜测？

然后依次阅读：

1. `docs/optimizations/M0_BASELINE.md` 到 `M7_SAMPLING_FINAL.md`；
2. 对应的 `benchmarks/results/*.json`；
3. `attention.py` 与 `flashinfer_backend.py`；
4. `model_runner.py` 的 Graph 与 Metadata 准备；
5. `scheduler.py` 和 `block_manager.py`；
6. `sampler.py`；
7. `tests/` 中每个边界条件为何存在。

真正达到大厂面试标准的标志，不是能背出“10 倍加速”，而是面试官改变 Batch、Context、模型大小或流量分布后，你仍能预测瓶颈会如何迁移，并设计下一组能证伪自己假设的实验。

## 16. 复现入口

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

具体里程碑的微基准、消融和正确性命令以 `docs/optimizations/M0_BASELINE.md` 至 `M7_SAMPLING_FINAL.md` 中的 Reproduction 小节为准。不要比较 Cold Start 与已完成 Shape Warmup 的结果，也不要跨 GPU、模型或依赖版本直接宣称相同加速比。
