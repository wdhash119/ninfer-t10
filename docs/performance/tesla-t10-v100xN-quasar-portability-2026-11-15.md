# ninfer-v100xN / QUASAR 对 T10 的可移植性

日期：2026-11-15

## 结论

**很值得参考，尤其是 NVFP4 QPN、批量 projection 和通信融合；但必须重写 SM75 kernel，不能搬 Volta MMA 或直接套用性能数字。**

V100 表格的口径是：2K 输入、512 输出、MTP4、优化 proposal、CUDA Graph、greedy，并开启 `volta-long-attention` 与 `volta-fast-decode` 两个实验数值路径；数值为不含 prefill 的 committed decode，多并发为聚合吞吐。它不是 prefill 数字，也不是同输出、同质量比较。

## 同工作负载基线

使用同一份 `code-chat-2048.ids`、512 输出、TP4、MTP4、INT8 KV、优化 proposal、greedy：

| 实现 | 条件 | Decode |
|---|---|---:|
| T10 quasar W4A8 | 4×T10，PCIe，无 fast profile | **67.7 tok/s** |
| V100 TP4 | 4×V100，NVLink，long+fast | 312.1 tok/s |
| V100 TP2 | 2×V100，PCIe P2P，long+fast | 220.8 tok/s |

T10 接受率为 72.9%，与 V100 常见约 74–76% 接近。因此差距主要来自 kernel、projection 组织和通信调度，而不是简单由 MTP 接受率造成。V100 fast/long 会改变数值关联，且没有新 GPQA，所以仍不能将上述差异解释为同质量、纯架构胜负。

## 最值得移植的机制

### 1. NVFP4 QPN 与 codec

来源：`src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh`、`nvfp4_volta_batch.cuh`。

可直接借鉴：

- E2M1/E4M3 的 shift+rebias 位解码；
- scale 折叠与 FP16 溢出边界；
- 不重排权重时，用 activation 的 PRMT 配对适配 decoder 的 `(j,j+4)` 顺序；
- 64K chunk 中连续读取四个 K16 group scale；
- K 分区和独立 accumulator 链；
- T9–64 的 packed activation supply，避免每个 projection 重复转换；
- gate/up、attention、GDN input/output 的统一覆盖，不保留第二份权重。

SM75 重写点：将 `volta_mma_qp_n` / `m8n8k4` 改为 `ldmatrix` + `m16n8k8` 的新 fragment 映射。调研实测 Volta shape 在 T10 只有约 11.3 TFLOP/s，而 m16n8k8 为 88.5，直接搬会浪费约 7/8 的 Tensor Core 能力。

### 2. 批量 projection 与 verify

V100 的 `nvfp4_volta_batch_supported()` 专门覆盖 T9–64 的 TP2/TP4 dense shapes，并针对 T9–16、17–32、33–40、41–64 选择不同物理 warp/K phase。它正是 MTP verify 的关键区间，而不是只优化单 token decode。

T10 profile 显示，quasar 请求仍有大量 NVFP4 A16 projection，W4A8 grouped 只覆盖一部分。优先做 SM75 QPN 的 T9–64 verify，比先微调通信更直接。

### 3. TP push 与跨层融合

来源：`src/ops/kernel/allreduce.cuh` 与 TP4 packed batch 实现。

可借鉴：

- 本地贡献留寄存器，向其他 rank 推送 BF16 packet；
- 按全局 rank 顺序求和；
- projection→packet→sum/RMSNorm handoff；
- 大 channel 避免小消息多步传输。

协议本身不依赖 SM70，但 T10 是纯 PCIe、PIX/PHB 两档，必须重新标定 packet 大小和门限，不能沿用 NVLink 的 640 KiB 结论。

### 4. 调度与验收

- MTP4 packed verify、prefill/decode 交错、实际 batch 统计可直接参考；
- committed wall、完整轮次、FP64 oracle、Graph/eager、guard/read-only 检查应照搬；
- `volta-fast-decode` 与 `volta-long-attention` 只能作为独立实验档位，必须标注输出/接受率可能改变。

## 不建议直接移植

- `m8n8k4`、Volta warp/fragment 常量、寄存器和 occupancy 假设；
- NVLink 的 TP 带宽比例和 packet 门限；
- “V100 老架构反而更快”的直觉结论。V100 的优势来自高度定制的 QPN、fast numerics、调度和 NVLink，不是架构本身；
- 把多并发聚合值当作单请求速度，或把 decode 当作 prefill。

## 推荐落地顺序

1. **SM75 QPN decode/verify 原型**：先覆盖 T9–64 gate/up、down，复用 V100 codec 和 packed activation，使用 `sm75_mma.cuh` 的 m16n8k8。
2. **扩大到 attention/GDN input/output**：消除 profile 中残留 NVFP4 A16 的主耗时，与当前 W4A8 做同 fixture A/B。
3. **projection 复用与直接 BF16 发布**：再引入 packed supply 和跨 projection 共享。
4. **TP push + sum/norm 融合**：在 kernel 收益成立后，按 T10 PCIe 拓扑重新标定。
5. **并发和 fast/long 实验档**：最后用相同 2K/512、C1/C2/C4/C8 矩阵验证，并单独记录输出与质量变化。

当前 q4lynn 的约 96.6 MTP3 仍是生产推荐；quasar/SM75 QPN 是值得开展的实验路线。

原始基线与 Nsight：`/public/src/ninfer-t10/profiles/bench/t10-quasar/`。
