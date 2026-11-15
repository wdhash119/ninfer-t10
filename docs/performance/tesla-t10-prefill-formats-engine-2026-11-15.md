# Tesla T10：权重格式与 prefill 引擎继续调优

日期：2026-11-15

## 结论

1. **换权重格式可以提升 NInfer prefill，但幅度有限。** 同源 FP8 HF 解码后重打包为 row-split Q4/Q5，再覆盖 BF16 cuBLAS store 路由，8K/64K 达到 **1036.0 / 1000.5 tok/s**；相对 q4lynn GGUF 提升 **10.3% / 15.6%**。
2. **row-FP8 在 SM75 上明显回退。** 8K/64K 只有约 **210 tok/s**，强制 legacy 也没有改善。该实验权重已删除。
3. **decode 仍选 q4lynn。** 新 Q4/Q5 权重虽然带 proposal head，MTP3 仅 **66.7 tok/s**，低于 q4lynn 的 4K 94.8、64K 97.0 tok/s。
4. **代码层面仍有空间。** 本轮新增 BF16 cuBLAS store 已带来约 5–6% 提升；Nsight 还显示残留 Q4/Q5 A16、GDN chunked 和 FP32 store 三类优化点。

## 权重格式 A/B

同一 Coder390 FP8 HF 源：`/public/llm/models/qwen/qwen3.8-27b-merkyor-coder390/FP8`。

| NInfer 权重/代码 | 8K prefill | 64K prefill | MTP3 decode |
|---|---:|---:|---:|
| q4lynn GGUF | 939.3 | 865.2 | 4K 94.8；64K 97.0 |
| FP8 HF → row-split Q4/Q5 | 977.8 | 950.2 | 4K 66.7 |
| 上述 Q4/Q5 + BF16 cuBLAS store | **1036.0** | **1000.5** | 未重测 |
| FP8 HF → row-FP8，unified | 210.0 | 210.2 | 未测 |
| row-FP8，强制 legacy | 208.8 | 未测 | 未测 |

单位均为 tok/s。NIAH 响应始终为 `ORCHID=493817; COLOR=COBALT`，且 `cached_tokens=0`。

作为参照，vLLM-2080Ti-Definitive 的 64K prefill 为 **1269.2 tok/s**，仍比当前最快 NInfer 高约 27%。

### 转换路径

原始 FP8 是 128×128 block FP8，不能直接走 `import_encoded` 导入 NInfer row-FP8/NVFP4；`qwen3_8_27b_nvfp4` 因而不适用。可行路径是解码 block FP8，再用 `qwen3_8_27b` recipe 重打包为 row-split Q4/Q5，并加 `--proposal`。

保留的实验权重：

```text
/public/llm/models/qwen/ninfer-qwen3.8-27b/coder390-fp8-to-q4q5-proposal.ninfer
```

转换和原始结果：

```text
/public/src/ninfer-t10/profiles/bench/t10-fp8-convert/
```

## 引擎代码优化

### 本轮已保留：BF16 cuBLAS store

此前普通 `linear()` 只有 FP32 TP partial 能进入 cuBLAS prefill，BF16 output 会落回 Q4/Q5 A16。新增：

- `store_bf16_epilogue`：int32 GEMM 结果按 row/token scale 恢复并写 BF16；
- `w4_cublas_store_bf16_launch`；
- `linear.cpp` 对相同 shape gate 的 BF16 输出选择该路由。

收益：

- 8K：977.8 → **1036.0 tok/s**，约 +6.0%；
- 64K：950.2 → **1000.5 tok/s**，约 +5.3%。

既有 `ninfer_w4_cublas_prefill_test` 全部通过，NIAH 端到端输出一致。完整 perplexity 未重跑，源码保持未提交。

### 新 Nsight 轮廓

文件：

```text
/public/src/ninfer-t10/profiles/bench/t10-fp8-convert/q4q5-bf16store-tp4-8k.nsys-rep
/public/src/ninfer-t10/profiles/bench/t10-fp8-convert/q4q5-bf16store-kernels_cuda_gpu_kern_sum.csv
```

主要耗时：

- Q4/Q5 A16 sliced-K FP32/BF16 kernel 合计约 **28%**：仍有 shape 未进入 cuBLAS gate；
- GDN `output`、`prepare_wy_wu`、`state_passing` 合计约 **15.6%**；
- `store_fp32_epilogue` 约 **5.3%**；
- token int8 量化与 channel absmax 合计约 **6.2%**；
- 权重反量化到 int8 合计约 **3.4%**。

### 后续优先级

1. **扩展剩余 Q4/Q5 shape 的 cuBLAS coverage。** 先记录每个残留 A16 调用的 `(N,K,T,dtype)`，只给实测能胜出的 shape 加 gate，避免全局切换造成回退。
2. **跨 prefill chunk 缓存物化的 int8 权重。** 当前每个 chunk 都重新物化同一权重；若显存预算允许，跨 chunk 复用可去掉约 3.4% 的直接反量化成本及部分同步空隙。必须先验证 TP4/64K 显存峰值。
3. **融合 FP32 store 与 TP partial exchange。** `store_fp32_epilogue` 本身约 5.3%，减少一次 FP32 写读比继续微调 GEMM tile 更直接。
4. **优化 GDN chunked 三阶段。** 约 15.6% 的总时间值得做 chunk shape、向量读写和 state-passing 的独立 A/B。
5. **q4lynn GGUF 若还要继续提速，才考虑 SM75 Marlin-like kernel。** 这是较大工程；当前 row-split 路线的小规模 coverage 优化应优先。

## 当前推荐

- **Prefill-heavy**：仍优先 vLLM-2080Ti-Definitive。
- **NInfer prefill 备选**：本轮 Q4/Q5 artifact + BF16 cuBLAS store，64K 约 1000 tok/s。
- **MTP decode**：继续使用 q4lynn NInfer，不能换成新 Q4/Q5 artifact。

测试服务均已停止，四张 GPU 均为 0 MiB。慢速 row-FP8 权重已删除，转换 override 与日志保留。
