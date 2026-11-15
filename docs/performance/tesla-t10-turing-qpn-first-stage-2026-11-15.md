# SM75 QPN 第一阶段实验：T1–64 / TP4 gate/up/down

日期：2026-11-15

## 结论

已完成可运行、数值正确的 SM75 QPN 原型，但**当前实现没有性能收益，因此默认关闭，未继续扩大 attention/GDN 覆盖**。

同一 2K 输入、512 输出、TP4、MTP4：

| 路径 | Decode |
|---|---:|
| 默认 W4A8，最终二进制复测 | **67.6 tok/s** |
| 实验 SM75 QPN | **56.2 tok/s** |

QPN 比当前 W4A8 慢约 17%。默认路径已恢复并复测，所有诊断输出均已移除。

## 实现内容

新增：`src/ops/linear/nvfp4/nvfp4_turing_qpn.cu`。

- 原始 NVFP4 E2M1 codes + BlockScaleK16M128x4 scales，无第二份权重；
- BF16 activation 一次转换为 FP16；
- SM75 `mma.m16n8k8`，FP32 累加；
- 每 CTA 32 output rows、8 warps、4 个 K split；
- T1–64 分块：Tiles 1/2/4/5/8；
- 支持 plain Linear FP32/BF16、LinearAdd residual、SwiGLU 两个 FP32 projection + 既有 `silu_mul`；
- 支持完整形状及 TP4 分片，包括 gate/up `8704×5120`、down `5120×4352`；
- 仅 `NINFER_NVFP4_QPN=1` 时启用，默认 W4A8/A16 不变。

## 调试中修复的问题

1. 归约 shared plane 行维错误，Compute Sanitizer 检出后改为 4 K split × 32 行。
2. 工作区按 `1…1025` 规划时，必须按区间内最大 QPN 宽度 64 预留，而非只看右端点。
3. 生产 T5 verify 与 T9–64 不符，补充 T1–8。
4. TP4 分片不是完整形状，补充 `8704×5120` 和 `5120×4352`。
5. TP rank 的 gate/up/down 实际还会走 plain Linear FP32 partial，已接入 plain dispatch。

## 验证

- `ninfer_linear_nvfp4_a16_test`：通过。
- `ninfer_linear_add_nvfp4_test`：通过。
- SwiGLU：没有新增失败；`this build selects the legacy Linear table only` 在默认基线也失败，与 QPN 无关。
- Int8Grouped：同样的既有 route-table 失败，基线一致。
- 2K/MTP4 请求与 CUDA Graph 可运行，最终默认路径 67.6 tok/s。
- 完整 perplexity 未重跑。

## 负结果判断

当前版本采用直接 nibble/table 解码和独立 FP32 projection，尚未具备 V100 性能来源的完整组合：

- E2M1/E4M3 的向量化 shift+rebias 解码与 activation PRMT 配对；
- 跨 gate/up 的寄存器级 paired epilogue，而非两份 FP32 plane；
- 更高效的 fragment 共享和 packed projection；
- TP push + sum/norm handoff。

因此不能把第一版慢解释为“QPN 不适合 T10”，只能说明**朴素 m16n8k8 移植不足以超过当前 W4A8 narrow/grouped kernel**。若继续，下一步应先做单算子 profile 和向量化 codec，而不是直接扩大调用覆盖。

原始结果：`/public/src/ninfer-t10/profiles/bench/t10-quasar/`。源码未提交，实验路径默认关闭，GPU 已全部释放。
