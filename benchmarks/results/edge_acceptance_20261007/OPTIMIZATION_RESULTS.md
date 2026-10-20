# 端侧优化实施与验收结果

日期：2026-10-07
源码提交：6befefaed848e88173e66acd04c2eefa6a1ddcd8
设备：RTX 5060（SM120，8 GiB），WSL2 Ubuntu，PyTorch 2.11.0+cu128、CUDA 12.8、FlashInfer 0.6.6
模型：Qwen3-0.6B

## 已交付

### 版本与复现

- 将原来无 .git 的 ZIP 源码做成本地 Git 仓库，保留 ZIP 状态首个提交和 GitHub origin。
- 增加已验证环境依赖快照、WSL 安装脚本、环境诊断和独立进程重复基准工具。
- GitHub 提交可查询，但 WSL HTTPS 拉取超时，未能把 ZIP 源码与远端提交做逐文件比对；本地版本可由上方 commit 精确追溯。

### 显存与错误处理

- 加入显式 gpu_memory_reserve_gib，服务默认预留 0.50 GiB，显存不足时启动阶段报告 free/total、KV blocks 和缩小上下文/并发的建议。
- 启动失败会恢复 PyTorch 默认设备/dtype 并销毁 NCCL process group。
- 故意配置 100 GiB 预留时验证到提前失败，无 NCCL 退出警告。
- 默认服务参数（2048 context、8 个 active sequences、0.70 显存利用率、0.50 GiB reserve）可在本机启动。模型就绪后健康接口观测 PyTorch allocated/reserved 为 3.82/3.92 GiB，整卡 free 约 2.90 GiB。

### 本机服务

- 新增只监听 localhost 的 OpenAI 兼容 /v1/completions 和 /v1/chat/completions，支持 SSE 流式输出、最大上下文/输出限制、bounded queue、429 背压、/health 显存与队列状态。
- 流式客户端断开后会取消对应 Sequence 并释放 KV blocks。
- 实机验证健康检查、模型列表、聊天、补全、流式结束、双请求并发；客户端断开后 active_requests 回到 0，健康状态正常，后续请求成功。
- 服务是单用户、本机范围的推理接口；没有认证，不应暴露到局域网或公网。

## 性能验收

每种负载运行 5 个独立进程；每进程预热 1 次、测 1 次。表内吞吐、TTFT、TPOT、显存均为跨进程中位数，E2E 的 SD 是样本标准差。完整逐次结果位于同目录 final_matrix/。

| 并发 | 输入 tokens | Decode tok/s | 输出 E2E tok/s | E2E SD | batch TTFT | mean TPOT | 峰值 allocated |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 64 | 212.9 | 197.9 | 0.9 | 16.0 ms | 4.70 ms | 4.71 GiB |
| 4 | 64 | 845.9 | 782.6 | 4.2 | 16.5 ms | 4.73 ms | 4.72 GiB |
| 8 | 64 | 1,630.5 | 1,472.3 | 8.1 | 21.4 ms | 4.91 ms | 4.73 GiB |
| 1 | 256 | 210.0 | 194.3 | 1.4 | 16.7 ms | 4.76 ms | 4.72 GiB |
| 4 | 256 | 795.2 | 667.1 | 9.4 | 35.6 ms | 5.03 ms | 4.74 GiB |
| 8 | 256 | 1,474.6 | 1,094.6 | 3.1 | 65.0 ms | 5.43 ms | 4.78 GiB |

与建立目录时的 5 次同进程基线相比，六组 E2E 变化约 -0.1% 至 -0.6%。测试方法不同（本次为独立进程中位数），所以这表示推理核心吞吐基本保持，没有宣称新增服务层带来 kernel 加速。

### 在线长请求到达实验

交替运行 mixed 与 prefill-first 各 5 个独立进程。场景是在短请求已解码时加入一个 512-token 长 Prefill：

| 调度策略 | 短请求平均 token 间隔中位数 | token 间隔 p95 中位数 | 最大间隔中位数 |
|---|---:|---:|---:|
| mixed | 7.62 ms | 19.28 ms | 19.94 ms |
| prefill-first | 8.51 ms | 5.26 ms | 80.90 ms |

mixed 的 p95 高于 prefill-first，因为每轮 Prefill chunk 会形成较稳定的间隔；prefill-first 多数 token 间隔较低，但偶发长停顿约 81 ms。交互服务保留 mixed，可避免这类长停顿。原始结果在同目录 online_latency/。

### 功耗与连续请求稳定性

- RTX 5060 功耗采样间隔 200 ms，共 230 个样本、约 46 s；平均 38.99 W，中位数 31.52 W，范围 11.75–83.71 W。
- 同一窗口执行 5 次独立进程、六组负载，计入预热共生成 4,160 个 token；按整卡窗口功耗估算约 1,793.3 J，约 2.32 token/J（0.431 J/token）。该估计包含模型/进程启动、预热和空闲功耗，没有扣除空闲基线，适合做整次运行的保守参考，不代表纯 decode 能效。原始样本和计算结果在 power_efficiency/。
- 本机 API 连续负载 40 个 Chat Completions、4 路并发、每个请求 10 tokens：40/40 成功，400 个输出 token，1.62 s，24.6 请求/s；请求延迟 P50 75 ms、P95 939 ms。结束后 health 仍为 ok，queue_depth 和 active_requests 均为 0；PyTorch allocated 从 3.836 到 3.852 GiB。
- 这次是短时并发 soak，不等同于数小时耐久测试；长时间运行仍需在目标部署时补测。原始数据见 service_soak.json。

## 验证

- 完整回归：45 passed，5 subtests passed；FlashInfer 使用隔离 JIT 缓存。
- uv pip check：75 个包兼容。
- Python compileall、WSL setup 脚本 bash -n 通过。
- scripts/check_environment.py 能报告 nvcc、CUDA、驱动映射、GPU 能力、free memory 和 Git commit。

## 有意保留的后续项

- **INT4 权重量化未接入。** 当前 Qwen3 使用自定义 LinearBase/F.linear 和分片权重加载；torchao 文档列出的稳定 INT4 weight-only 流程主要面向 torch.nn.Linear。当前本机也没有适配后的量化模型或内核 A/B 结果。直接包装或反复反量化可能牺牲速度/数值质量，因此不把未经验证的 INT4 列为已交付。等目标扩大到更大模型时，应实现专用 LinearBase 量化加载路径，再测显存、速度和输出质量。[torchao 推理量化文档](https://docs.pytorch.org/ao/stable/workflows/inference.html)
- INT4 权重量化作为有意 no-go 保留；数小时级 soak test 仍未做。已完成 40 请求/4 并发的短时服务稳定性检查，以及整卡窗口功耗/能效估算。
- LoRA、推测解码、多卡和多租户仍不在此单卡端侧版本范围。
