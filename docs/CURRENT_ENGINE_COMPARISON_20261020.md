# 当前 nano-vLLM 引擎对比报告 — 2026-10-20

## 结论

在这台 RTX 5060 Laptop 上测得的六组固定 Qwen3-0.6B 负载中，当前候选版本（`bd28d2e9`）的**几何平均输出吞吐量达到 vLLM 0.32.0 的 89.4%**。vLLM 0.32.0 在六组负载中均更快，逐组差距为 **5.4%–18.0%**。在本次设置下，当前候选版本运行前后通过 `nvidia-smi` 采样到的 GPU 总显存占用约为 **6.35–6.38 GiB**，vLLM 0.32.0 约为 **7.47–7.54 GiB**。

与本机能够运行的 SM120 兼容基线相比，当前候选版本的几何平均吞吐量为 **8.79 倍**。该基线基于上游 nano-vLLM，并做了使其能在本机 SM120 环境中运行所必需的最少修改；它**不是未经修改的上游版本**。

此前与 vLLM 0.31.0 的对比中，当前候选版本的几何平均吞吐量是其 **1.03 倍**。根据测量日 vLLM 官方发布页，vLLM 0.32.0 是当时最新的稳定版本，并于 2026-10-18 发布。vLLM 0.31.0 的结果保留作历史参考，不应再作为最新版本对比结论。

因此，当前证据支持一个有明确范围的结论：在这组六种负载上，本项目明显快于本机可运行的 SM120 兼容基线；与当时最新的官方 vLLM 相比，吞吐量仍有幅度不大但每组都存在的差距。这些结果不能证明项目在其他模型、长上下文、更大批次、在线服务或其他硬件上也有相同性能。

## 对比版本与运行环境

| 系统 | 代码或发行版本 | Python / PyTorch / CUDA / Triton / FlashInfer | Attention 与 KV Cache 路径 |
|---|---|---|---|
| 当前 nano-vLLM | `bd28d2e9c23172e147fa47a01e96fca9cc9bd4db` | Python 3.12.3 / PyTorch 2.14.1+cu130 / Triton 3.8.0 / FlashInfer 0.6.6 | FlashInfer Paged Attention、64-token KV 块、优化 RMSNorm 和按形状启用的 Q/K 融合 |
| SM120 兼容基线 | `7aa1f251cf1e041ded3371bab41f90e6736111ac`（`baseline-blackwell-v0`） | Python 3.12.3 / PyTorch 2.14.1+cu130 / Triton 3.8.0 / FlashInfer 0.6.6 | PyTorch SDPA、256-token KV 块 |
| vLLM 0.31.0 | vLLM 0.31.0 发行版 | Python 3.12.3 / PyTorch 2.13.0+cu130 / Triton 3.7.1 / FlashInfer 0.7.0.post1 | 官方发行 wheel，启用 CUDA Graphs |
| vLLM 0.32.0 | vLLM 0.32.0 发行版 | Python 3.12.3 / PyTorch 2.13.0+cu132 / Triton 3.7.1 / FlashInfer 0.7.0.post1 | 官方发行 wheel，启用 CUDA Graphs |

硬件为 NVIDIA GeForce RTX 5060 Laptop GPU，计算能力 12.0（SM120），显存 8 GiB，运行于 WSL2。模型为 Qwen3-0.6B，revision 为 `c1899de289a04d12100db370d81485cdf75e47ca`；`model.safetensors` 的 SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。

## 负载与测量方法

每个引擎独立启动进程运行三轮。轮次之间轮换引擎执行顺序，六种负载的执行顺序也做轮换。每个引擎进程在计时前，都会对每种形状先做预热。

| 请求数 | 每个请求的输入 token 数 | 每个请求的输出 token 数 |
|---:|---:|---:|
| 1、4、8 | 64 | 32 |
| 1、4、8 | 256 | 32 |

测试使用 BF16、随机种子 `20261020`、温度 `0.6`、`ignore_eos=True`、最大模型长度 512、最多 8 个序列、最多 2048 个批处理 token，以及 GPU 显存使用率设置 `0.70`。测试关闭 Prefix Cache。基准测试 worker 根据随机种子和负载形状确定性地生成输入 token ID，各引擎使用相同的输入。每组测试均返回了预定数量的请求和输出 token。每个对比矩阵包含九次进程运行（每引擎三轮），所有运行都完成了六种负载，没有记录到负载失败。

这里比较的是**完整推理引擎**。不同引擎使用的 Attention 内核、KV Cache 策略、块大小、编译路径和框架版本并不相同。因此，结果反映的是本次测试中整套软件栈的表现，不能单独归因于某一个 CUDA 内核或功能。vLLM 0.32.0 使用 PyTorch 2.13.0+cu132，而 nano-vLLM 使用 PyTorch 2.14.1+cu130。

## 性能结果

下表各项吞吐量和批次延迟是三轮测量的中位数。批次延迟表示整个批次从开始到完成的时间，不是单个请求的延迟。

| 负载（请求数 × 每请求输入 token 数） | nano-vLLM 吞吐（tok/s） | 兼容基线吞吐（tok/s） | vLLM 0.31 吞吐（tok/s） | vLLM 0.32 吞吐（tok/s） | nano / vLLM 0.32 | nano 批次延迟 | vLLM 0.32 批次延迟 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 × 64 | 205.20 | 26.31 | 210.26 | 247.03 | 83.1% | 0.1559 s | 0.1295 s |
| 4 × 64 | 763.37 | 82.09 | 733.78 | 845.68 | 90.3% | 0.1677 s | 0.1514 s |
| 8 × 64 | 1,481.69 | 131.50 | 1,401.06 | 1,600.05 | 92.6% | 0.1728 s | 0.1600 s |
| 1 × 256 | 198.66 | 26.08 | 207.63 | 242.33 | 82.0% | 0.1611 s | 0.1321 s |
| 4 × 256 | 678.60 | 80.38 | 631.82 | 717.29 | 94.6% | 0.1886 s | 0.1784 s |
| 8 × 256 | 1,123.18 | 127.90 | 1,061.45 | 1,187.12 | 94.6% | 0.2279 s | 0.2156 s |
| 几何平均吞吐比 | **相对兼容基线 8.79×** | — | **相对 vLLM 0.31 为 1.03×** | **相对 vLLM 0.32 为 0.894×** | — | — | — |

### 如何理解这些指标

- **tok/s（每秒 token 数）**：测量期间生成的输出 token 总数除以耗时，数值越高表示这组负载完成得越快。
- **89.4%**：把六组负载逐组计算 nano-vLLM 与 vLLM 0.32.0 的吞吐比，再取几何平均，结果为 0.894。可以理解为这组六组形状的总体相对吞吐表现；它不是说每一组都恰好慢 10.6%。实际逐组比例为 82.0%–94.6%。
- **8.79×**：同样以六组负载的几何平均吞吐比计算，当前候选版本相对兼容基线约快 8.79 倍。它不能改写成相对未经修改上游快 8.79 倍。
- **几何平均**：对多组相对比例求乘积后开相应次方，常用于汇总不同规模下的加速比；它不等于把各组 tok/s 简单相加或平均。
- **批次延迟**：一次处理整个批次所用的时间。吞吐更高通常意味着相同批次耗时更短，但吞吐和延迟回答的问题不同，所以表中分别报告。

## 显存与延迟指标的限制

GPU 显存数据取每个测量批次运行前、运行后两次 `nvidia-smi` 采样值中的较大者。短暂出现的峰值可能被漏掉，而且总量包含同一块 GPU 上其他程序的占用。因此，这些数据是采样快照，不能当成精确的单引擎峰值显存分配。vLLM 的离线运行由独立的 `EngineCore` 进程执行，基准 worker 无法读取该进程的 PyTorch allocator 数据，所以 vLLM 的 allocator 字段为零；不能把这些零与 nano-vLLM 进程内的 allocator 数值直接比较。

vLLM 的离线输出没有提供本 worker 可用的逐请求到达和完成时间戳，因此本报告对所有系统都报告批次延迟。测试验证了固定 token 数和成功完成情况，但没有比较生成文本质量或逐 token 完全一致性。

## 未修改上游 nano-vLLM 的运行情况

未经修改的上游代码固定在 `GeeeekExplorer/nano-vllm` 的 commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。该版本未能在本机成功运行，因此没有直接上游吞吐数据。上游代码依赖 `flash_attn`；尝试在本机 PyTorch 2.14.1+cu130 环境中安装 FlashAttention 2.8.3 时，编译因 ATen 头文件要求 C++20、构建却使用 C++17 而失败。尝试的预编译 wheel 地址返回 404。在 PyTorch 2.13.0+cu130 的参考环境中再次从源码构建时，NVCC 编译期间多次同时出现 WSL 服务故障。因此，按测量计划，本报告改用明确标注的 SM120 兼容基线作为替代对照。

## 原始证据与复现方式

- 汇总 JSON：[`summary.json`](../benchmarks/results/current_engine_comparison_20261020/summary.json)
- nano / 兼容基线 / vLLM 0.31 三方运行结果：[`current_engine_ab_20261020_formal`](../benchmarks/results/current_engine_ab_20261020_formal/)
- nano / 最新 vLLM 0.32 运行结果：[`current_vs_vllm032_20261020`](../benchmarks/results/current_vs_vllm032_20261020/)
- 测试 worker：[`compare_engine_matrix_worker.py`](../benchmarks/e2e/compare_engine_matrix_worker.py)
- 汇总脚本：[`summarize_current_comparison.py`](../benchmarks/e2e/summarize_current_comparison.py)

每次运行的命令行参数、完整环境信息、耗时、吞吐、显存快照和输出哈希均保存在各自的 JSON 结果中。可以用以下命令从归档 JSON 重新生成汇总文件：

```bash
python3 benchmarks/e2e/summarize_current_comparison.py \
  --output benchmarks/results/current_engine_comparison_20261020/summary.json
```
