# T10 调研落地：quasar 对照与 prompt partials 优化

日期：2026-11-15

## 结论

调研文档可直接指导配置优化。本轮未改引擎代码，已得到比此前更好的 q4lynn 配置：

- **均衡档**：q4lynn + INT8 prompt partials + `--tensor-split 0.8,1,1,1`
  - 8K prefill：约 **1200 tok/s**
  - 64K prefill：约 **1060 tok/s**
  - 4K MTP3：**96.6 tok/s**
- **纯 64K prefill 档**：q4lynn + INT8 prompt partials、不使用 tensor-split
  - 64K prefill：约 **1080 tok/s**
- **保守档**：将 INT8 partials 改成 BF16
  - 8K/64K prefill：约 **1060 / 992.3 tok/s**

INT8/BF16 partials 会改变计算关联。固定 NIAH 输出均保持 `ORCHID=493817; COLOR=COBALT`，但完整 perplexity 尚未重跑，不能据此宣称无损。质量敏感场景先用 BF16，INT8 作为待评估的性能档。

## 1. quasar proposal 权重对照

权重：

```text
/public/llm/models/qwen/ninfer-qwen3.8-27b/qwen3_8_27b_nvfp4-quasar-proposal.ninfer
```

TP4、INT8 KV、MTP3、相同 fixture：

| quasar 路径 | 8K prefill | 4K MTP3 decode |
|---|---:|---:|
| 默认 A16 | 226.5 | 31.4 |
| `--nvfp4-w4a8-int` | **446.2** | **52.0** |

W4A8 使 prefill 接近翻倍、MTP3 提升约 66%，说明文档中的 INT8 Tensor Core 路线确实有效。但它仍明显低于 q4lynn 的当前配置，因此：

- quasar 可作为 NVFP4 INT8 kernel 的对照与测试权重；
- 当前不建议用它替换 q4lynn 作为 T10 生产权重。

## 2. prompt partials A/B

q4lynn artifact：`qwen3.8-27b-coder390-q4lynnstyle-q8mtp.ninfer`。

| 配置 | 8K prefill | 64K prefill | 4K MTP3 |
|---|---:|---:|---:|
| 默认 FP32 partials | 912.2 | 865.2（既有同路线结果） | 94.0 |
| BF16 partials | 约 1060 | 992.3 | 未测 |
| INT8 partials | 约 1170 | 约 1080 | 91.2 |
| INT8 + split 0.8 | **约 1200** | 约 1060 | **96.6** |

单位均为 tok/s。INT8 + split 的 8K prefill 相对默认提升约 31.5%；64K 提升约 22.5%。split 使各卡权重视图从约 5.35/4.38 GiB 调整为 4.97/4.65 GiB，MTP3 从 91.2 提升到 96.6；但 64K prefill 从不分片约 1080 小幅降至约 1060。

Definitive 的 64K prefill 1269.2 tok/s 仍领先，但 q4lynn 已能在同一 artifact 上兼顾约 1060–1080 prefill 与 96.6 MTP3，不再需要为了两项指标分别转换权重。

## 3. 推荐启动方式

### 均衡档

```bash
cd /public/src/ninfer-t10
MODEL=/public/llm/models/qwen/ninfer-qwen3.8-27b/qwen3.8-27b-coder390-q4lynnstyle-q8mtp.ninfer

build/apps/ninfer-serve "$MODEL" \
  --devices 0,1,2,3 --tensor-parallel \
  --tensor-split 0.8,1,1,1 \
  --tensor-parallel-prompt-partials int8 \
  --prefill-cublas --prefill-chunk 4096 \
  --embedding-residency host \
  --max-context 66560 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --adaptive-mtp \
  --ngram-draft-tokens 0 --mtp-attention-window 4096
```

质量优先时将 `int8` 改成 `bf16`，并去掉 `--tensor-split` 后再做长上下文 A/B。

### quasar INT8 对照

在上述命令中替换 MODEL，并加入：

```bash
--nvfp4-w4a8-int
```

## 4. 对调研文档的落地判断

1. **立即有效**：prompt partials、tensor-split、NVFP4 W4A8 开关。本轮均已实测。
2. **Marlin 结论一致**：无 repack 的 Marlin-like 原型不值得继续调 tile；若重做，必须先做加载期 repack、fragment 共享与流水线。
3. **下一代码优化点**：用 quasar 作为 W4A8 测试权重，分析其 446 tok/s 与理论 INT8 线的差距；同时优先扩大 q4lynn 的 INT8 MMQ coverage 和优化 GDN。
4. **decode 后续**：继续围绕 rank 0 做 profile 与 split 细调；当前 0.8 是首个有效点，不代表全局最优。
5. **质量门槛**：对 INT8/BF16 partials 做完整 perplexity 和长输出对比后，才能把性能档设为默认。

原始结果：`/public/src/ninfer-t10/profiles/bench/t10-quasar/`。所有服务已停止，GPU 均已释放。
