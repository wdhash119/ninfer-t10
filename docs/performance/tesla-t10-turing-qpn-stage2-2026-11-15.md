# SM75 QPN 第二阶段：向量化 codec 与 paired SwiGLU

日期：2026-11-15

## 结论

向量化 codec 有小幅收益，但 paired epilogue 没有继续提升端到端速度；QPN 仍低于默认 W4A8，因此默认关闭。

| 版本 | 2K/MTP4 decode |
|---|---:|
| 默认 W4A8 | **67.6 tok/s** |
| 第一阶段 QPN | 56.2 tok/s |
| 向量化 codec + activation PRMT | **60.5 tok/s** |
| joined/two-parent paired SwiGLU | 60.5 tok/s |

向量化 codec 相对第一阶段提升约 7.7%，但仍比默认慢约 10.5%。

## 完成内容

### 向量化 codec

- 每次读取 32-bit、8 个 E2M1 codes；
- 使用 V100 的 shift+rebias 解码，而非逐 nibble/table 解码；
- 用 activation PRMT 生成与 decoder `(j,j+4)` 对应的配对；
- 保留原始 NVFP4 权重布局和 m16n8k8 主循环。

### Paired SwiGLU

- 同一 CTA 内完成 gate/up 两次 projection；
- 直接计算并存储 BF16 SwiGLU；
- 同时支持 joined 权重的 `up_row_offset=n/2` 和 two-parent 权重；
- 去掉两份 FP32 plane 和独立 SiLU kernel。

修复 joined grid 应取 `n/2` output rows 的问题；既有 SwiGLU 测试没有新增失败，A16 Linear 和 LinearAdd 通过。A4 路由已与实验 QPN 隔离。

## 判断

paired 没再提速，说明当前主要剩余成本已在 QPN 主循环/fragment 生成和现有其他 projection，而不是 FP32 plane 或 SiLU。当前实现不足以进入 TP push、attention/GDN 大范围改造。

下一阶段若继续，应先做 QPN 单算子级的 fragment 共享、激活 staging 和 K split/occupancy 调参，并与现有 narrow kernel 直接比较，而非继续增加包装层。

实验仍只在 `NINFER_NVFP4_QPN=1` 时启用。诊断输出已移除，默认 W4A8 未改变，GPU 已释放。原始结果：`/public/src/ninfer-t10/profiles/bench/t10-quasar/`。
