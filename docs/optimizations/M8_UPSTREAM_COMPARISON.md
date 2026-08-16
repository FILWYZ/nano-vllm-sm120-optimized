# M8-lite：相对 GitHub nano-vLLM 的直接 A/B 结果

## 结论

在本机 NVIDIA GeForce RTX 5060 Laptop/SM120、Qwen3-0.6B FP16、同一 Python/PyTorch/CUDA 环境和完全相同请求 Token 下，最终离线配置相对 GitHub nano-vLLM 的 SM120 兼容基线，在原仓库 README 定义的 256 请求长负载中获得 **7.66× 中位吞吐加速**：

| 版本 | 三轮 tok/s | 中位 tok/s | 中位耗时 |
|---|---|---:|---:|
| Upstream SM120 Compat | 245.78 / 248.45 / 247.66 | 247.66 | 540.92 s |
| Optimized M8 Offline | 1,901.77 / 1,882.35 / 1,898.00 | 1,898.00 | 70.58 s |

相对提升：

- 吞吐提升至 **7.66×**，即增加 **666.4%**；
- 完成时间降低 **86.95%**；
- 三轮均完整生成 **133,966** 个输出 Token；
- 两个变体各自的三轮输出 SHA-256 均稳定一致；
- 最终配置的 Scheduler 抢占次数为 **0**，实际 Prefill Token 等于输入 Token **142,827**，没有重复 Prefill。

该结论仅适用于报告中明确列出的硬件、模型、软件版本和负载，不表示全面优于官方 `vllm-project/vllm`。

## 对比对象为何是 SM120 Compat

GitHub 上游冻结在：

- 仓库：`GeeeekExplorer/nano-vllm`；
- 上游 Commit：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`；
- 上游 SM120 兼容 Commit：`7aa1f251cf1e041ded3371bab41f90e6736111ac`。

未经修改的上游依赖 FlashAttention。当前 PyTorch 2.11.0+cu128、SM120 环境没有经验证可用的原版 FlashAttention 路径，因此 `7aa1f25` 只增加 SDPA Fallback、SM120 后端选择和正确性测试，不包含 M1–M8 性能优化。

这意味着本报告能够证明：

> 当前项目在 RTX 5060 Laptop SM120 上，显著优于从 GitHub 原版派生的最小可运行兼容基线。

它不能证明当前项目在 RTX 4070 或上游 FlashAttention 原生可运行环境中仍有相同加速比。

## 固定矩阵直接结果

每个 Shape 使用独立 Python 进程，按交替顺序执行三个正式重复。表中为三轮中位数：

| 请求 | 输入 | Compat tok/s | M7 tok/s | 加速比 |
|---:|---:|---:|---:|---:|
| 1 | 64 | 27.35 | 197.77 | 7.23× |
| 4 | 64 | 87.43 | 781.85 | 8.94× |
| 8 | 64 | 141.61 | 1,463.52 | 10.34× |
| 1 | 256 | 27.42 | 193.62 | 7.06× |
| 4 | 256 | 84.52 | 670.62 | 7.93× |
| 8 | 256 | 135.33 | 1,125.40 | 8.32× |

六个 Shape 的中位加速比几何平均为 **8.23×**。所有 36 个正式进程都生成了预期 Token 数，没有 Shape 回退，满足预注册的固定矩阵胜出标准。

## GitHub README 原始负载

请求生成逻辑逐项复现上游 `bench.py`：

- 256 请求；
- Seed 0；
- 输入长度随机 100–1024，共 142,827 个输入 Token；
- 输出长度随机 100–1024，共 133,966 个输出 Token；
- `temperature=0.6`；
- `ignore_eos=True`；
- `max_model_len=4096`；
- 正式计时前执行 `"Benchmark: "` Warmup。

### 显存代价

| 版本 | 峰值 Allocated | 峰值 Reserved |
|---|---:|---:|
| Compat | 5.96 GiB | 6.36 GiB |
| M8 Offline | 6.56 GiB | 6.61 GiB |

M8 的峰值已分配显存增加约 **10.1%**。这是 CUDA Graph、FlashInfer Workspace/Metadata 和 Decode KV 预留换取吞吐稳定性的代价，不应隐藏。

## 首次失败及其价值

预注册后的第一次 M7 长负载没有直接成功：

1. 初始运行在高 KV 压力 Mixed Batch 中触发 `slot_mapping` 与 K/V Token 数不一致；
2. 诊断发现 `_schedule_decode` 会在同一调度轮次抢占已经加入 `scheduled_seqs` 的早期序列；
3. Commit `0a181f0` 将已选 Decode 序列暂存到独立列表，完成候选决策后再放回 `running`；
4. 新增真实页边界回归测试，当前测试总数为 23；
5. 修复后默认 M7 成功完成，但耗时 723.72 秒、185.11 tok/s，比 Compat 回退 24.7%。

这次负结果暴露了第二个瓶颈：16-token Page、256 个长输出请求和完整 Prompt KV 预分配组合会触发反复抢占与重复 Prefill。

## M8 输出感知 Admission Control

Commit `2133478` 新增可选 `reserve_decode_kv`：

- Admission 时按 `prompt_tokens + max_output_tokens` 预留 KV Page；
- 已预留页在 Decode 跨边界时直接复用，不重复分配；
- 容量不足的请求继续等待，而不是先运行再被抢占并从头 Recompute；
- 离线配置使用 Page 64、Prefill-first 和 Output KV Reservation；
- 在线 Mixed Batching 默认行为仍保留，避免用离线策略冒充在线最优。

代表性 64 请求长负载中：

- 抢占 15→0；
- Prefill Token 42,949→32,768；
- 峰值 Allocated 6.87→6.41 GiB；
- 吞吐 2,408.4→2,446.0 tok/s。

完整 README 负载中，最终配置实现 0 次抢占和 0 个重复 Prefill Token，从而把默认 M7 的 185.11 tok/s 提升到 1,898.00 tok/s 中位数。

## 正确性与可复现性

- 最终完整测试：23 项 CPU/GPU 测试全部通过；
- `uv pip check`：63 个依赖包全部兼容；
- Compat SDPA Packed/Prefix/Decode 与 PyTorch 参考实现对拍；
- M7 FlashInfer、Triton KV、CUDA Graph、页边界、调度、缓存和采样均有独立测试；
- 三轮完整负载实际输出 Token 数均等于 133,966；
- 随机采样后端数值路径不同，因此不要求跨后端 SHA-256 相同，但每个变体内部三轮摘要稳定。

## 当前优点

- 在本机 SM120 上具有原版不具备的已验证可运行路径；
- README 原始负载中位吞吐达到 Compat 的 7.66×；
- 固定矩阵六个 Shape 全部胜出，几何平均 8.23×；
- FlashInfer Paged Attention、无同步 KV、CUDA Graph 和输出感知 Admission 形成完整因果链；
- 23 项测试、原始 JSON、Commit、预注册计划和复现脚本齐全；
- 同时保留在线 Mixed 模式与离线吞吐模式，不把一个策略强行用于所有场景。

## 当前缺点

- 主要对比是 SDPA Eager 的 SM120 兼容基线，不是上游在 RTX 4070 上的原生 FlashAttention 性能；
- 最终离线配置增加约 10.1% 峰值 Allocated 显存；
- KV 全量输出预留会降低瞬时 Admission 并发，不适合未知/超长输出或严格在线公平性场景；
- 只验证 Qwen3-0.6B、单块 8 GiB GPU 和一个长负载，尚不能外推到其他模型；
- 没有生产级 HTTP Server、LoRA、量化、Speculative Decoding 或分布式 Serving。

## 标准复现

```bash
cd /home/asus/projects/nano-vllm-baseline
source .venv/bin/activate

python benchmarks/upstream_ab.py \
  --model /path/to/Qwen3-0.6B \
  --variant optimized_m8_offline \
  --suite github \
  --block-size 64 \
  --reserve-output-kv \
  --disable-mixed-batching \
  --output benchmarks/results/m8_ab/optimized_m8_offline_github.json

python benchmarks/summarize_m8.py
```

对照版本需从 `/home/asus/projects/nano-vllm-upstream-compat` 作为 `cwd/PYTHONPATH` 运行同一个适配器。完整命令和验收规则见 `M8_COMPARISON_PLAN.md`。
