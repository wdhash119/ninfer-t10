# SM75 Marlin-like GGUF kernel 原型

日期：2026-11-15

## 结论

已完成可运行、数值正确的 SM75 Marlin-like 原型，但当前版本**不优于既有 MMQ**，因此默认关闭，不影响现有推理路径。

| 路径 | 8K prefill | 相对默认 MMQ |
|---|---:|---:|
| Marlin-like，J=64 | 594.6 tok/s | -34.8% |
| Marlin-like，J=128 | 759.3 tok/s | -16.8% |
| 默认 MMQ，同二进制复测 | **912.2 tok/s** | — |

J=128 比 J=64 快约 27.7%，但仍比 MMQ 慢约 16.8%。NIAH 输出均为 `ORCHID=493817; COLOR=COBALT`，`cached_tokens=0`。

## 实现

新增：`src/ops/linear/gguf/ggml_marlin_sm75.cu`。

- 支持 GGUF Q4_K/Q5_K dense prefill，T≥128、K 为 256 的倍数、CC=7.5。
- 每 CTA 64 rows，4 warps；每 warp 16 rows，J=128 columns。
- 使用 SM75 支持的 `mma.m16n8k8.row.col.f32.f16.f16.f32`。`m16n8k16` 是 SM80+，不能用于 T10。
- 直接解码原始 GGUF block 到 FP16 fragments，不做整层反量化或离线 repack。
- activation 转 FP16，按列使用 power-of-two 范围因子，FP32 累加。
- 支持 input gather、FP32/BF16 输出、Accumulate、SwiGLU/GateProduct epilogue。
- 最终采用确定性单 CTA K 循环和普通 store。

启用方式：

```bash
NINFER_GGUF_MARLIN=1
```

默认不设置或设为 0 时继续使用 MMQ。

## 调试过程中排除的方案

1. **尾列 lane 发散**：部分 lane 跳过 warp-wide `mma.sync` 会造成错误结果及后续停滞。现改为所有 lane 执行 MMA，越界列由共享内存零填充，store 裁剪。
2. **split-K FP32 原子归约**：可以运行且误差在容限内，但求和顺序不固定，导致独立 FP32/BF16 调用不能逐位一致，已放弃。
3. **J=64**：权重解码随 column tile 重复，594.6 tok/s；J=128 明显改善但仍不足。

Q4_K J=128 编译后约 128 registers、34 KiB shared；Q5_K 约 142 registers，未显示异常 local-memory spill。

## 验证

通过过滤后的既有 GGUF FP64/oracle 测试：

- Q4_K，K=2560，scalar/cooperative vector schedules；
- Q5_K，K=2560；
- T=1…257，包括 Marlin 的 T=130/257 与尾列；
- plain/gathered linear、linear_add、SwiGLU、FP32/BF16 输出一致性。

测试用的过滤和打印修改已全部移除，未留下诊断 instrumentation。完整 perplexity 未重跑。

## 为什么仍慢

当前瓶颈不是 MMA 指令，而是：

- 直接从 Q4_K/Q5_K block 生成 fragment 的解码成本；
- 同一 quad 多 lane 重复 weight decode/fragment 生成；
- 没有 Marlin 友好的离线权重重排和 `ldmatrix` 数据路径；
- 共享内存流水线仍较简单。

若要继续，需要改为**离线 Marlin 权重 repack + quad/warp fragment 共享 + 更明确的双缓冲流水线**，而不是继续调 J 或 MMA shape。当前 MMQ 的 INT8 Tensor Core 路线更适合直接读取现有 GGUF。

原始结果：`/public/src/ninfer-t10/profiles/bench/t10-marlin/`。源码保持未提交，默认 MMQ 已复测，GPU 已全部释放。
