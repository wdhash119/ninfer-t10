# Tesla T10：64K Prefill、MTP3 与 2/4/8 路并发调优

日期：2026-11-15  
源码：`/public/src/ninfer-t10`  
模型：`qwen3.8-27b-ult-unc-q4_k_m-v3.ninfer`

## 目标达成度

用户给定目标：vLLM FP8 四卡 TP 的 64K prefill/decode 约 1600/50 tok/s，MTP3 单流 decode 100+ tok/s。它与本轮 NInfer artifact 不同，因此作为性能目标而非同权重对照。

| 指标 | NInfer 实测 | 结论 |
|---|---:|---|
| 64K prefill | **约 1008 tok/s** | 未达到 1600 |
| 64K 普通 decode | **44.0 tok/s** | 未达到 50 |
| 64K MTP3 长输出稳态 decode | **56.0 tok/s** | 超过 50，未达到 100 |
| 4K 普通并发 decode | **169.5 tok/s，C8 聚合** | 扩展良好 |
| 4K MTP3 并发 decode | **102.1 tok/s，C2 聚合** | C2 是最佳点 |

MTP3 长输出样本包含 211 个生成 token，接受率 **65.3%**，每轮 **2.96 tokens**。此前仅 17-token 的短输出样本达到 72.8 tok/s，但不作为最终稳态结论。

## 4K 短提示并发

固定 344-token prompt、每请求 512 output tokens、四卡 TP、INT8 KV、cuBLAS prefill。`Steady` 是满批次一秒区间的聚合 decode；`per request` 是单请求平均 decode。

| 后端 | C | Steady 聚合 tok/s | per request tok/s | 峰值 MiB/卡（最大） |
|---|---:|---:|---:|---:|
| 普通 | 1 | 48.5 | 48.5 | 约 6,200 |
| 普通 | 2 | 67.7 | 33.9 | 6,245 |
| 普通 | 4 | 102.4 | 25.8 | 6,413 |
| 普通 | 8 | **169.5** | 21.4 | 6,749 |
| MTP3 | 1 | 59.9 | 60.7 | 6,337 |
| MTP3 | 2 | **102.1** | 51.6 | 6,425 |
| MTP3 | 4 | 73.0 | 18.9 | 6,607 |
| MTP3 | 8 | 53.6 | 7.2 | 6,967 |

结论：

- 普通批处理选 C8；需要更低单请求延迟时选 C4。
- MTP3 选 C2。C4/C8 的验证宽度增长使其失去优势，不能把 C2 的 102 tok/s 当成单流成绩。

## 64K 长提示并发

每个请求实际输入 64,512 tokens，四卡 TP，INT8 KV，chunk 4096，普通 decode。所有配置均能加载，未发生 OOM。

| C | 墙钟 s | 聚合 prefill tok/s | per request decode tok/s | 峰值 MiB/卡（最大） |
|---:|---:|---:|---:|---:|
| 1 | 64.4 | 1008.0 | 44.0 | 6,667 |
| 2 | 128.8 | 1004.9 | 44.0 | 7,253 |
| 4 | 257.1 | 1005.4 | 44.1 | 8,425 |
| 8 | 515.3 | 1002.3 | 44.0 | **10,775** |

`--concurrent-prefill` 对总 prefill 没有提升：C2 开/关分别为 128.76 / 128.58 秒。项目执行模型在多卡下不做跨请求 prefill 批处理，所以 C=2/4/8 近似线性增加墙钟，聚合 prefill 基本不变。当前主要价值是并发 decode 和资源利用，而不是通过单实例 TP 达到 1600 tok/s。

64K MTP3 的长输出测试：TTFT 68.3 秒，稳态 decode 56.0 tok/s，per-request 平均 61.2 tok/s。它的 decode 达标，但 prefill 没有提高。

## 参数探针

| 配置 | 结果与结论 |
|---|---|
| MTP3，adaptive | 59.9 tok/s，较固定 MTP3 的 59.1 仅微增 |
| MTP3 + ngram15 | 61.2 tok/s，仅小幅提升，且不再代表纯 MTP |
| MTP7，adaptive | 50.3 tok/s，回退，不采用 |
| Q6 output head | 48.8 tok/s，回退，不采用 |
| `draft-min-p=0.5` | 当前 Qwen3.5-core 不支持，启动拒绝 |
| `lm-head-draft` | 当前 artifact 无 optimized proposal head，不能使用 |
| `concurrent-prefill` | 64K 聚合 prefill 基本不变 |

当前 MTP3 达到 100 tok/s 的主要障碍是 **完整 draft head 的成本**，不是继续加 draft 数或简单调参。若后续需要进一步逼近该目标，应从原始权重重新转换并加入 `--proposal`，使用 indexed proposal head，而不是继续测试 MTP7 或量化 output head。

## 推荐加载配置

### 64K、重视 decode

```bash
build/apps/ninfer-serve "$MODEL" \
  --devices 0,1,2,3 --tensor-parallel \
  --prefill-cublas --prefill-chunk 4096 \
  --embedding-residency host \
  --max-context 66560 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --adaptive-mtp \
  --ngram-draft-tokens 0 --mtp-attention-window 4096
```

### 4K、普通批处理吞吐

使用 `--spec none --max-concurrency 8`；若重视单请求速度，改为 4。

### 4K、MTP3 聚合吞吐

使用 `--spec mtp --draft-tokens 3 --adaptive-mtp --ngram-draft-tokens 0 --max-concurrency 2`。不要直接用 C8 期待线性扩展。

## 代码与验证

本轮在既有 PartProjection 优化上补齐了 GDN `conv_snapshot/conv_record` 的 cuBLAS workspace family，避免这些路径在容量计算时回退到 A16。临时诊断插桩已全部移除。

- `ninfer_w4_cublas_prefill_test` 通过。
- 64K C=1/2/4/8 均启动成功，显存峰值低于 16 GiB/卡。
- 固定 NIAH 输出保持一致。
- 所有测试服务已停止，GPU 显存已释放。

原始结果：

- 4K 并发：`profiles/bench/t10-optimization/mtp-concurrency/`
- 64K 并发：`profiles/bench/t10-optimization/long-concurrency/`
- 64K MTP：`profiles/bench/t10-optimization/long-mtp/`
- Nsight：`profiles/bench/t10-optimization/nsys/`

源码修改保持未提交，便于审查。
