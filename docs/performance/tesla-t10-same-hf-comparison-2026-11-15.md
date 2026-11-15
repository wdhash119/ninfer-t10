# 同源 HF 权重：vLLM vs NInfer（Tesla T10 / TP4）

日期：2026-11-15  
vLLM：`/public/src/vllm` + `/public/llm/vllm-qwen3.8-27b.sh`  
NInfer：`/public/src/ninfer-t10`

## 同源与测试口径

vLLM 加载 `merkyor-coder390/INT4/W4A16`。NInfer 对比：

- `coder390-q4lynnstyle-q8mtp.ninfer`：实际走 GGUF `mul_mat_q`，不是此前 row-split Q4/Q5 路线。
- `coder390-nvfp4-w4a4-mtp-dflash2.ninfer`：测试 NVFP4 与 DFlash2。

因此，本次既比较同源模型，也明确区分 **Marlin、GGUF MMQ、NVFP4 与 draft 实现**。相同 `long_niah_8k/64k` 和 512-token decode fixture，四卡 TP、FP16 activation；NInfer KV 主要使用 INT8，vLLM 使用 FP8。

## 核心结果

| 工作负载 | vLLM | NInfer q4lynn | 结论 |
|---|---:|---:|---|
| 8K prefill | **1,547 tok/s** | 939.3 tok/s | vLLM 约 1.65× |
| 64K prefill | **1,330.6 tok/s** | 865.2 tok/s | vLLM 约 1.54× |
| 4K 普通 decode | 56.4 tok/s | 47.9 tok/s | vLLM 更快，约 1.18× |
| 4K MTP3 decode | 56.4 tok/s | **94.8 tok/s** | NInfer 约 1.68× |
| 64K MTP3 decode | 未复测 | **97.0 tok/s** | 接近 100，接受率 76.7% |

NInfer q4lynn 的 64K MTP3 样本含 165 个生成 token，平均 decode 94.2 tok/s，稳态 97.0 tok/s。其 Q8 MTP/full-head 路线明显优于旧 artifact 的 59.1 tok/s，尽管它没有 optimized proposal head。

## 为什么 vLLM prefill 更快

vLLM 使用 W4A16 Marlin 和 FP16 activation，大 chunk 下不需要执行 NInfer GGUF 的逐 block 低精度矩阵路径。q4lynn 的 Nsight 显示主要时间在不同 GGUF `mul_mat_q` 类型，而不是此前已优化的 row-split projection。

这意味着此前针对 row-split Q4/Q5 的 cuBLAS 优化不能直接套用到 q4lynn；差距是 **量化矩阵实现差异**，不是简单的 TP 通信或加载参数问题。

## 优化尝试与取舍

### 整层反量化 + cuBLAS BF16 GEMM

复用 `gguf::dequantized_product`，8K 总时间由约 8.5 秒变为 **20.6 秒**。反量化整个权重矩阵的成本过高，已回退，不采用。

### GGUF MMQ tile 上限

- J=64：约 8.53 秒，与原 J=128 基本一致。
- J=32：约 9.74 秒，回退。
- 当前 J=128 仍是最佳选择，已移除实验开关。

### NVFP4 / DFlash2

- 默认 prefill：223.3 tok/s。
- `--nvfp4-w4a8-int`：408.7 tok/s。
- DFlash2 稳态 decode：37.3 tok/s，接受率 41.2%，不推荐用于当前 T10 工作负载。

### q4lynn MTP 调参

- MTP3 固定或 adaptive 均约 94.8 tok/s。
- MTP4 降至 86.8 tok/s，不采用。
- `gdn-state-fp16 + mlp-a8-decode` 降至 88.6 tok/s，不采用。
- 推荐保持 MTP3 + `--mtp-attention-window 4096`，ngram 关闭。

## 推荐配置

### 主要追求 prefill / 长文档冷启动

使用当前 vLLM 脚本。其 8K/64K prefill 分别为 1547 / 1330.6 tok/s，优于 NInfer GGUF。

### 主要追求 MTP3 decode

使用 q4lynn NInfer：

```bash
cd /public/src/ninfer-t10
MODEL=/public/llm/models/qwen/ninfer-qwen3.8-27b/qwen3.8-27b-coder390-q4lynnstyle-q8mtp.ninfer

build/apps/ninfer-serve "$MODEL" \
  --devices 0,1,2,3 --tensor-parallel \
  --prefill-cublas --prefill-chunk 4096 \
  --embedding-residency host \
  --max-context 66560 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --adaptive-mtp \
  --ngram-draft-tokens 0 --mtp-attention-window 4096
```

其 64K decode 接近 100 tok/s；但 prefill 仍低于 vLLM，不应只按 decode 指标选择全部工作负载。

## 验证与原始数据

- 同 fixture 输出均为 `ORCHID=493817; COLOR=COBALT`。
- 既有 `ninfer_w4_cublas_prefill_test` 通过。
- 实验性反量化和 MMQ tile 修改已回退，只保留此前的有效优化及 GDN workspace 修正。
- 所有测试服务已停止，GPU 显存已释放。

路径：

- vLLM：`/public/src/vllm/profiles/bench/t10-same-hf/`
- NInfer 普通：`/public/src/ninfer-t10/profiles/bench/t10-optimization/same-hf/`
- NInfer MTP：`/public/src/ninfer-t10/profiles/bench/t10-optimization/same-hf-mtp/`
- GGUF tile A/B：`/public/src/ninfer-t10/profiles/bench/t10-optimization/mmq-tiles/`
- Nsight：`/public/src/ninfer-t10/profiles/bench/t10-optimization/nsys/q4lynn-kernels.txt`

源码修改保持未提交，便于继续审查。
