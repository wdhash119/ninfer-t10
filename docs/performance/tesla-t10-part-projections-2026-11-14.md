# Tesla T10：从 vLLM 四卡 TP 到 NInfer 1000+ tok/s Prefill

日期：2026-11-14  
优化源码：`/public/src/ninfer-t10`  
对比源码：`/public/src/vLLM-2080Ti-Definitive`

## 结论

已把 NInfer 四卡 TP prefill 从 **436.1 tok/s 提升到 1,113.0 tok/s**，双卡从 **286.8 提升到 1,287.8 tok/s**。8K 与 64K 均达到 1,000+ tok/s；128K 最优为 897.8 tok/s。

同输入 vLLM 四卡 TP 的实测为 **1,547 tok/s**。NInfer 双卡已达到其约 83%，四卡约 72%；但 vLLM 使用 `merkyor-coder390 INT4/W4A16`，NInfer 使用另一份 Q4_K_M 转换 artifact，不能把剩余差异视为严格的同 kernel 比较。

## vLLM 基线

配置：4 × Tesla T10，TP4，FP16 activation，Marlin，FP8 KV，`max_num_batched_tokens=8192`。

同一 `long_niah_8k.json`：

- prompt：7,680 tokens
- TTFT：4.965 s
- 总请求时间：5.241 s（17 output tokens）
- 冷 prefill 吞吐：约 1,547 tok/s
- 响应：`ORCHID=493817; COLOR=COBALT`

这说明 vLLM 的高吞吐主要来自 **W4A16 Marlin、FP16 activation 和大 chunk**，不是单纯 TP 通信。

## NInfer 的根因

Nsight 对上一版 NInfer 四卡 8K 的统计显示：

- Q5 A16 kernel 占约 48.9% GPU 时间；
- Q4 A16 kernel 占约 21.0%；
- 合计约 69% 仍在 A16；
- cuBLAS 路线占比很小。

实际 TP 调用使用 `PartProjectionWeights`，此前存在三层遗漏：

1. `PartProjectionWeights::policy` 没有继承 `--prefill-cublas`；
2. `project_parts()` 不经过 split-input cuBLAS 路径，而是逐 part 调普通 linear；
3. 前一阶段只覆盖了 FFN/output 的 FP32 partial，没有覆盖 attention/GDN 输入 projections。

因此，仅调整 chunk、BF16 partial 或 peer transport 都无法接近 vLLM。

## 本轮修改

### 1. PartProjection policy 传播

在 `src/models/qwen3_5/execution/parameters.cpp` 中，Turing 且显式启用 `--prefill-cublas` 时，让 TP 的 `PartProjectionWeights` 保留该 policy。

### 2. 多 parent、多 destination 的共享 cuBLAS projection

在 `src/ops/linear/part_projection.cpp/.h` 中新增 SM75 路径：

- 每个 TP part 作为一个 parent；
- 所有 parent 共享一次 activation quantisation；
- 每个 part 的 epilogue 写入目标张量的准确 row range；
- 继续复用既有 `w4_cublas_projection_supported/launch`；
- 不支持时保留原逐 part linear 回退，非 Turing 行为不变。

同时更新 attention/GDN wrappers 的 workspace 计算和 family gate。

### 3. 与前一阶段合并

保留此前的 FP32-partial store、TP2/TP4 output/down 与 gate/up 覆盖，以及 Turing 默认关闭 prompt pieces。

## 最终性能

统一条件：INT8 KV、MTP0、无 prefix reuse、host embedding、greedy、`--prefill-cublas`。除 128K 的最后一行使用 8192 外，其余使用 4096 chunk。

| GPU / 阶段 | 输入 | Prefill tok/s | TTFT s | Decode tok/s | 完整请求 s |
|---|---:|---:|---:|---:|---:|
| vLLM 四卡 TP（不同 artifact） | 7,680 | **1,547** | **4.97** | — | 5.24 |
| NInfer 双卡，上一阶段 | 7,680 | 286.8 | 26.79 | 34.4 | 27.25 |
| NInfer 双卡，本轮 | 7,680 | **1,287.8** | **5.98** | 34.8 | **6.44** |
| NInfer 四卡，上一阶段 | 7,680 | 436.1 | 17.62 | 47.6 | 17.96 |
| NInfer 四卡，本轮 | 7,680 | **1,113.0** | **6.91** | 47.5 | **7.25** |
| NInfer 双卡，本轮 | 64,512 | **1,001.9** | 64.46 | 29.9 | 65.00 |
| NInfer 四卡，上一阶段 | 64,512 | 430.3 | 149.97 | 44.0 | 150.35 |
| NInfer 四卡，本轮 | 64,512 | **1,008.0** | **64.06** | 44.0 | **64.44** |
| NInfer 四卡，本轮，chunk 4096 | 130,048 | 889.4 | 146.36 | 40.7 | 146.77 |
| NInfer 四卡，本轮，chunk 8192 | 130,048 | **897.8** | **144.98** | 40.8 | **145.40** |

相对最初基线，双卡 8K 从 66.0 到 1,287.8 tok/s，约 **19.5×**；相对上一阶段约 **4.49×**。四卡 8K 相对上一阶段约 **2.55×**。

## 正确性

- 既有及新增 `ninfer_w4_cublas_prefill_test` 全部通过。
- 本轮 8K/64K/128K 响应均为 `ORCHID=493817; COLOR=COBALT`，与基线和 vLLM 一致。
- 非 Turing 回退与默认未开启 cuBLAS 的路径未改变。
- 未重新跑完整 perplexity；仍遵循项目既有 cuBLAS 精度权衡说明。

## 使用命令

```bash
cd /public/src/ninfer-t10
MODEL=/public/llm/models/qwen/ninfer-qwen3.8-27b/qwen3.8-27b-ult-unc-q4_k_m-v3.ninfer

build/apps/ninfer-serve "$MODEL" \
  --devices 0,1 --tensor-parallel \
  --prefill-cublas --prefill-chunk 4096 \
  --embedding-residency host \
  --max-context 66560 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 --spec none \
  --host 127.0.0.1 --port 18080
```

- 8K：双卡当前最快；四卡仍有更高 decode。
- 64K：双卡/四卡 prefill 基本相同，按 decode 和显存需求选择。
- 128K：可用四卡和 `--prefill-chunk 8192`，但当前只有约 898 tok/s。

## 原始证据

- vLLM：`/public/src/vLLM-2080Ti-Definitive/profiles/bench/t10-tp4-prefill/`
- NInfer 最终：`/public/src/ninfer-t10/profiles/bench/t10-optimization/part-projections/`
- NInfer 修改前 Nsight：`/public/src/ninfer-t10/profiles/bench/t10-optimization/nsys/`

修改保持未提交，便于继续审查。`ninfer-all` 未修改。
