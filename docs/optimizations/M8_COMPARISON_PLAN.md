# M8-lite：GitHub nano-vLLM 与 M7 公平 A/B 预注册计划

## 目标与限定

目标是在同一台 RTX 5060 Laptop SM120、相同 Qwen3-0.6B FP16 权重、相同请求 Token 和相同采样语义下，判断 M7 是否优于 `GeeeekExplorer/nano-vllm` 的冻结上游版本。

本实验不与官方 `vllm-project/vllm` 比较，也不宣称跨 GPU、模型或负载普遍领先。

## 冻结版本

| 变体 | Commit | 定义 |
|---|---|---|
| GitHub upstream | `bb823b3` | 2026-04-26 的 `origin/main` |
| Upstream SM120 compat | `7aa1f25` | 上游加最小 SDPA/SM120 兼容层，不含 M1–M7 |
| Optimized M7 | `56fc371` | 当前 FlashInfer/Graph/Scheduler/Cache/Sampling 优化版本 |

未经修改的上游依赖 FlashAttention；当前已验证环境没有可用的 SM120 FlashAttention 路径。因此性能 A/B 使用 `7aa1f25` 作为可运行对照，并必须在报告中写明这是 SDPA Eager 兼容基线。上游不能直接运行本身只构成兼容性优势，不能换算为性能加速。

## 控制变量

- WSL2、GPU、驱动、模型文件、Python、PyTorch、CUDA 和随机请求完全相同。
- 两个变体由同一个 Python 环境运行，`cwd`/`PYTHONPATH` 决定加载的代码版本。
- 每次正式测量使用独立 Python 进程；模型初始化不计入生成时间。
- 正式计时前执行 Warmup 并同步 CUDA；每次记录实际输出 Token、峰值显存和原始 JSON。
- M7 使用其选定的 16-token Page；Compat 使用原版 256-token Page。这属于被评估的 M4 优化，而不是未声明的环境差异。

## 测试套件

### 固定矩阵

- 请求数：1、4、8。
- 输入长度：64、256。
- 输出长度：32。
- `temperature=0.6`、`ignore_eos=True`。
- 每个 Shape 至少三个独立正式重复；可承受时扩展为五次。

### GitHub README 原始负载

- 256 请求。
- 输入长度按 Seed 0 随机 100–1024。
- 输出长度按同一 RNG 流随机 100–1024。
- `temperature=0.6`、`ignore_eos=True`、`max_model_len=4096`。
- 先执行原脚本的 `"Benchmark: "` Warmup。
- 至少一个完整正式运行；若单次时间和热稳定性允许，再执行三个独立重复。

### 正确性与稳定性

- 最终 23 项 CPU/GPU 测试必须全部通过。
- Compat 的 SDPA Packed/Prefix/Decode 三项参考测试必须通过。
- 每次 Benchmark 的实际输出 Token 数必须等于请求的 `max_tokens` 总和。
- JSON 必须记录 Git Commit、设备、软件环境和输出摘要。

## 预注册胜出标准

M7 可在简历中表述为“相对 GitHub nano-vLLM 的 SM120 兼容基线更快”，需满足：

1. 固定矩阵六个 Shape 的吞吐几何平均至少提升 10%。
2. 至少五个 Shape 不回退；任何回退不得超过 5%。
3. GitHub README 负载成功完成且吞吐至少提升 10%，或 Compat 因已记录的资源/时间限制无法完成，此时不得给出该负载的加速比。
4. 两个变体的实际输出 Token 数与预期一致。
5. 正确性测试全部通过。

在线最大 Token Gap、KV 尾部浪费、Prefix Token Hit 和 Sampler 延迟作为 M1–M7 的补充机制证据，不冒充与上游直接 A/B 的吞吐结果。

## 简历表述规则

- 必须写“nano-vLLM 教学项目”或明确仓库名，不写成“官方 vLLM”。
- 必须写硬件、模型、上游 Commit/兼容基线和测试负载。
- 使用“提升至 X 倍”或“提升 Y%”，不混淆倍数与百分比。
- 不把随机采样与 Greedy 结果作为同语义比较。
- 不将单一 GPU/模型结论外推为普遍领先。
