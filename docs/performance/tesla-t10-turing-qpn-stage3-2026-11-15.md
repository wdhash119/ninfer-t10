# SM75 QPN 第三阶段：K split 与 fragment 共享

日期：2026-11-15

## 结论

两项候选均回退，已恢复第二阶段最佳的 **60.5 tok/s**。默认 W4A8 仍为 67.6，实验 QPN 继续默认关闭。

| 变体 | 2K/MTP4 decode |
|---|---:|
| 默认 W4A8 | **67.6 tok/s** |
| 当前最佳 QPN：K split=4、向量化 codec、paired | **60.5 tok/s** |
| K split=2 | 58.7 tok/s |
| K split=4 + quad leader/shuffle fragment 共享 | 58.4 tok/s |

## 结果解释

- K split=2 减少归约和 shared 使用量，但每 warp 的 K 工作翻倍，端到端下降。
- Quad leader 解码后用 shuffle 分发 fragment，shuffle 成本高于四个 lane 的重复解码，端到端同样下降。
- 未继续加入每 K-group 的 shared activation staging：需要额外线程同步，且当前 activation 已能利用 L1/L2；V100 的头注也明确采用无 shared 主循环。

## 验证与恢复

- A16 Linear、LinearAdd 重新通过。
- 当前最佳版本再次实测为 60.5 tok/s。
- 实验仍只在 `NINFER_NVFP4_QPN=1` 时启用；诊断输出已移除，GPU 已释放。

后续若继续，应转向单算子级寄存器/指令调度或更系统的 TP/调度移植，而不是继续微调 K split 或增加 shared staging。报告与原始结果位于 `/public/src/ninfer-t10/profiles/bench/t10-quasar/`。
