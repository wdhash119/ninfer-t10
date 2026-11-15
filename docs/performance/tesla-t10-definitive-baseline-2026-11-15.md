# vLLM-2080Ti-Definitive 基线修正

日期：2026-11-15

按用户指定，以 `/public/llm/sm75-vllm-qwen3.8.sh` 和 `/public/src/vLLM-2080Ti-Definitive/.venv` 重测，替代此前主线 `/public/src/vllm` 的基线。

## 实测

| 配置 | 工作负载 | 结果 |
|---|---|---:|
| Definitive，原始 `max_num_batched_tokens=2048` | 64K prefill | **1241.6 tok/s** |
| Definitive，调整为 8192 | 64K prefill | **1269.2 tok/s** |
| Definitive，MTP3 | 4K，512 output tokens | **79.9 tok/s** |

MTP3 的 TTFT 为 0.393 s，decode 阶段为 6.394 s，按 511 个后续 token 计算为约 79.9 tok/s。8192 只使 64K prefill 提升约 2.2%，因此当前脚本实测未达到约 1600 tok/s。

## 与 q4lynn NInfer 的关系

| 指标 | Definitive | q4lynn NInfer |
|---|---:|---:|
| 64K prefill | **1269.2 tok/s** | 865.2 tok/s |
| 64K MTP3 decode | 未单独复测 | **97.0 tok/s** |
| 4K MTP3 decode | **79.9 tok/s** | **94.8 tok/s** |

结论：Definitive 在 prefill 上更快，q4lynn NInfer 的 MTP3 decode 更快。此前主线 vLLM 的 56.4 tok/s 不应作为 Definitive 的基线。

原始结果：`/public/src/vLLM-2080Ti-Definitive/profiles/bench/t10-definitive/`。所有测试服务已停止，GPU 显存已释放。
