# Tesla T10 ×4（SM75）多卡 Qwen3.8-27B 推理：调研与借鉴清单

调研日期：2026-11-15。对象主机 `wsdev`（WS-DEVGUET，192.168.71.50），4 × Tesla T10 16 GB（TU102GL，SM75，PCIe Gen3 x16，无 NVLink）。
基线：`vLLM-2080Ti-Definitive`（`/public/llm/sm75-vllm-qwen3.8.sh`），65K 单流 prefill/decode 约 1250 / 80 tok/s。
改造树：`/public/src/ninfer-t10`（用户针对 T10 的 ninfer-all 派生树，含实验性 SM75 Marlin 内核）；对照树 `/public/src/ninfer-all`（共同起点 `c0321db0`）。
姊妹调研：V100 侧的 `v100-decode-survey.md`（在 `wsdev:/public/src/ninfer-v100xN/docs/`，SM70、双卡 PCIe + 远端四卡 NVLink）。本文只覆盖 SM75 + 四卡纯 PCIe，两篇的对应关系见附录 B。

本报告回答三件事：**本机环境的硬边界在哪里**（实测）、**现有实现离边界还有多远**（roofline 与实测差）、**读哪些项目、借什么机制、落到 ninfer 的哪个文件、如何判断值得保留**。
本轮只读代码与测探针，**没有改 ninfer 的任何内核**；除注明外所有 tok/s 数字引自既有 profile 文档，不是本轮重测。

标注约定：**【实测】**= 本轮在 wsdev 上直接测得；**【文档】**= 引用仓库内既有测量记录；**【待核实】**= 外部来源，本轮未在本机验证。

---

## 1. 直接结论

1. **prefill 的上限不在带宽，而在量化 GEMM 与 GDN 的算力利用率。** 四卡 FP16 张量核算力线约 **6,550 tok/s**（INT8 线约 13,100），vLLM Marlin 1,547 tok/s 只用到 FP16 线的 **24%**，NInfer 最快 1,113 tok/s 用到 17%【实测+计算，见 §4】。也就是说 prefill 还有 4–6× 的结构性空间，但 W4A16 这条路线的天花板就是 FP16 张量核；再往上要靠 **INT8/INT4 张量核**（sm75 有，实测 178 TOPS）和 **GDN 递推摆脱 FP32 乘积**。
2. **decode 已接近带宽线，剩下的主要是 rank 0 串行与通信开销。** q4lynn artifact 每卡 5.03 GB，实测 375 GB/s → 普通 decode 上限约 **74.5 tok/s**，实测 47.9（64%）；MTP3 94.8 tok/s 已越过单 token 权重线（一轮验证摊 2.96 个 token）。剩余空间集中在：lm head / MTP head 落在 rank 0、TP4 每交换延迟比 TP2 高约 80%、以及 CUDA Graph 覆盖不足【文档 + §5 实测交换表】。
3. **sm75 不是"少一个 cp.async 的 sm80"。** 本机 ptxas 实测：`cp.async` 全系列、`mma.m16n8k16`、`mma.m16n8k32`（s8）、`.tf32` mma、`mbarrier`、`redux.sync`、`stmatrix` 全部 **sm_75 不可用**（§3 表）。任何"把 sm80/1Cat-vLLM 的内核搬到 T10"的判断都必须先过这张表；可搬的是**算法组织**（分块、K 划分、GQA packed、push 归约、dequant 技巧），不是指令。
4. **本机 P2P 全部可用**（4 卡两两 `canAccessPeer` = yes，大 BAR 已开），同桥对（GPU0/1、GPU2/3）P2P 双向 **12.7 GB/s/方向**，跨桥 **9.9 GB/s/方向**；ninfer 启动时自测的 TP4 交换：`peer-write words` ≤15 KiB 为 **17.7 µs**，`scatter-flags` 54 KiB 为 25.6 µs【实测】。按 64 层 × 2 次交换/层估算，decode 每 token 通信约 **1.3–3.3 ms**（对 47.9 tok/s 的 6–14%）：值得优化但不是第一瓶颈。
5. **实验性 SM75 Marlin 目前比自家 MMQ 慢，根因大概率是"不重排权重"。** 今早的 profile：marlin 8K TTFT 12.92 s（prefill 595 tok/s）、j128 变体 10.12 s、MMQ 默认 8.42 s（约 912 tok/s）【实测日志】。文件头注释写明 "No GGUF repacking"，而 vLLM 的 Marlin（含上游 Turing 移植）是在 **repack 后的权重**上做 lop3/prmt + `ldmatrix` + 多 stage smem 流水。要么按 Marlin 的前提先重排（加载期一次性），要么放弃 Marlin、把精力放在 **llama.cpp MMQ 的 Turing INT8-TC 路线调优**（sm75 有 INT8 TC，理论线是 FP16 的 2×）。
6. **T10 的 device profile 已在 ninfer-t10 里标定（`nvidia-tesla-t10-sm75`，87 keys），但上游 `ninfer-all` 没有 T10 条目**（内置表只有 TITAN RTX / 2060S / 2070 / 2070S / 2080 Ti / Quadro 6000 / 8000 / Tesla T4）。T10 只有 56 SM、403 GB/s、150 W 封顶，与 68–72 SM 的 2080 Ti / TITAN 差 20–30%，**不能借用邻近 profile**。
7. **通信侧最省力的两条路已经存在开关，只是没在 T10 上标定**：`--tensor-parallel-prompt-partials bf16|int8`（把 prompt 分片和压到 1/2 与 9/32 字节）、`--tensor-split`（按卡调 TP 份额，可补偿 rank 0 承担 head/MTP 的额外字节）。另外 vLLM 在同一拓扑上实测"四卡 SM75 PCIe，32 KiB 起两阶段（reduce-scatter+allgather）就更快"，正好可以和 ninfer 的 `words/whole/scatter` 门限对照（该注释本轮已读，属源码事实）。
8. **优先级建议（详见 §8）**：A. 先把 rank 0 串行与 `--tensor-split` 这类零成本实验做掉；B. 决定 Marlin 的前提（repack 与否）并二选一深挖；C. INT8 张量核覆盖 q4lynn 的 prefill；D. GDN prefill 的 FP16-TC 化；E. 通信融合与本机门限标定。不要把 1Cat/geoffwatts 的 SM70 QPN 机制直接搬下来（§9）。
9. **`ninfer-v100xN`（SM70）值得读，但要按层取**：它的 TP push 协议、FP8/FP4 位解码与 scale 折叠、prefill/decode 交错调度、验收方法在 T10 上几乎无折扣；它的内核 tile 常量不能搬——实测同一份 Volta `mma.m8n8k4` 在 T10 上编译通过但只有 **11.3 TFLOP/s，是 `m16n8k8` 的 13%**。另外它的 CMake 只支持 `70|86|89`（没有 75 tier），移植等于把机制重写到 Turing tier。逐机制判定见附录 C。

---

## 2. 本机环境与实测

### 2.1 主机与软件栈【实测】

| 项 | 值 |
|---|---|
| OS / 内核 | Debian 13 (trixie)，6.12.113+deb13-amd64 |
| CPU / 内存 | Xeon E5-2698 v3 @2.30 GHz，32 线程，单 NUMA；46 GB RAM |
| 驱动 / CUDA | 590.48.01（CUDA 13.1 API），`/usr/local/cuda` → 13.4，另有 12.8 |
| 编译器 | GCC 14.2.0，CMake 3.31.6，Ninja 1.12.1，ccache 4.11.2，Python 3.13.5 |
| 内核参数 | `pci=realloc intel_iommu=on iommu=pt`（P2P 可用） |
| NCCL | 系统库中未见 libnccl；ninfer 本来也不用 NCCL（自带 epoch-flag 协议） |
| 构建口径 | `-DCMAKE_CUDA_ARCHITECTURES=75`，NVCC 13.4.92；`CMAKE_CUDA_FLAGS_RELEASE` 必须显式带 `-O3 -DNDEBUG`（既有文档踩过坑） |

### 2.2 GPU【实测】

| 项 | 值 | 备注 |
|---|---|---|
| 型号 | Tesla T10 16GB / GRID RTX T10-x，`10de:1e37`（TU102GL） | 四张同型号，ECC 关闭 |
| 计算能力 | 7.5 | |
| SM 数 | **56** | 不是 2080 Ti 的 68，也不是 TITAN 的 72 |
| Boost / 最大时钟 | 1,395 / 1,590 MHz | 满载 ALU 时实测 1,575–1,590 MHz |
| 显存 | 16 GB GDDR6，6,301 MHz，**256-bit** → 理论 403 GB/s | 实测 triad（读+写）**374.8–375.6 GB/s**（93%） |
| L2 / 共享内存 | L2 4 MB；smem 64 KB/SM，单块默认 48 KB、opt-in 64 KB | 限制 Marlin/attention 的 stage 数与 tile |
| 功耗 | 默认 = 最大 = **150 W**，`power.max_limit` 也是 150 W | `nvidia-smi -pl` 抬不上去；ALU 满载时 140–152 W，时钟未明显回落 |
| PCIe | Gen3 x16，Resizable BAR 生效（Region1 = 16 GB 可映射） | 空闲时 GPU3 曾报 Gen1 x16，负载下回到 Gen3 |
| 互联 | **无 NVLink**；GPU0/1 同 PIX，GPU2/3 同 PIX，跨对为 PHB | `nvidia-smi topo -m` |
| P2P | 4 卡两两 `canAccessPeer` 全部 **yes**，`cudaDeviceEnablePeerAccess` 成功 | |
| 拷贝带宽 | 固定内存 H2D **12.4 GB/s**、D2H **13.2 GB/s**；P2P 并发双向：PIX **12.7**、跨 PHB **9.9 GB/s/方向** | 256 MiB `cudaMemcpyPeerAsync` |

### 2.3 算力实测【实测】（`tc_probe.cu`，单卡，预热后 event 计时）

| 内核 | 吞吐 | 与理论值比较 |
|---|---:|---|
| FFMA（FP32） | **10.93 TFLOP/s** | ≈56×64×2×1.59 GHz = 11.4（Turing SM 只有 64 个 FP32 lane） |
| `mma.m16n8k8` f16 → **f32 累加** | **88.49 TFLOP/s** | ≈56×1024×1.59 GHz = 91 |
| `mma.m16n8k8` f16 → f16 累加 | 88.71 TFLOP/s | 与 f32 累加同速（TU102GL 上不翻倍） |
| `mma.m8n8k16` s8 → s32 | **178.32 TOP/s** | INT8 TC 恰为 FP16 的 2× |

结论：**T10 的 INT8 张量核是 FP16 的 2×，且 FP32-累加不额外掉速**——这与 TITAN 文档里 "Q4/Q5 的 verify 内核保不住 decode 比例" 并不矛盾：掉的不是指令吞吐，是解包/转换的发射槽（见 §4.3）。

---

## 3. SM75 指令边界【实测，CUDA 13.4 ptxas 逐个编译 sm_75 与 sm_80 对照】

| 机制 | sm_75 | sm_80 | 对本项目的含义 |
|---|---|---|---|
| `cp.async.ca/cg`（4/8/16B）、`commit_group`/`wait_group`、`cp.async.mbarrier.arrive` | **不可用** | 可用 | 所有依赖异步流水线的 sm80 内核（Marlin、1Cat 的 QPN、CUTLASS C2x 的 sm80 分支）搬过来必须改成 **LDG+STS 手工双/多缓冲**。vLLM 的 sm75 Marlin 正是这么做的（`marlin.cuh` 里 `__CUDA_ARCH__ < 800` 时 `cp_async*` 退化成普通存储、`cp_async_fence/wait` 是空函数） |
| `mma.m16n8k16` f16→f32 | **不可用** | 可用 | sm75 只有 **`m16n8k8`**：同一个 16×8 tile 要两条指令，累加器/A-B 寄存器映射都不同。vLLM 用 `#if __CUDA_ARCH__ == 750` 整段换 shape |
| `mma.m16n8k32` s8（INT8 大 shape） | **不可用** | 可用 | sm75 的 INT8 只有 **`m8n8k16`（带 `.satfinite`）**，一个 16×16 tile 要 4 条。`src/ops/common/sm75_mma.cuh` 已经是这个写法；但 `src/ops/linear/q4/q4_small_t_mma_i8.cuh` 与 `src/ops/common/mma.cuh` 里出现 `m16n8k32`，**这些文件在 sm75 构建下必须由 tier 宏走别的分支**，改动前要确认没有漏网的调用点 |
| `.tf32` mma | **不可用** | 可用 | 与 TITAN 文档一致：GDN 递推里 TF32 乘积在 Turing 上只有操作数全是 FP16 值时才走 FP16 TC，否则退 FP32——实测 11× 的差距就来自这里 |
| `mbarrier.init` | **不可用** | 可用 | 没有硬件 barrier：流水同步只能靠 `__syncthreads` + 计数器 |
| `redux.sync.add.s32 / .f32` | **不可用 / 不可用** | s32 可用（f32 需 sm_90） | warp 归约只能 `__shfl_xor_sync`，或 INT8 路径用 `dp4a`（sm75 可用） |
| `stmatrix` | 不可用（需 sm_90） | 不可用 | 写回只能 `ldmatrix` + 普通 store |
| `ldmatrix.sync.m8n8.x4 / .x4.trans` | **可用** | 可用 | Marlin 风格的 A/B fragment 装载在 sm75 完全没问题 |
| `dp4a`、`match.any.sync`、`vote.sync`、`red.global.add.f16x2`、`red.relaxed.sys.global.add.f32` | **可用** | 可用 | 跨卡 flag/原子（`*.sys`）可用，ninfer 的 Words 形态在此有硬件支撑 |
| `mapa`/cluster、`griddepcontrol`（PDL） | 不可用（需 sm_90） | 不可用 | 不要指望 programmatic dependent launch 掩盖 launch 间隙 |

---

## 4. roofline：现在离边界有多远

### 4.1 计算口径

模型 `Qwen3_5ForCausalLM`（artifact 内 config）：hidden 5,120，64 层（`linear_attention` 为主，每 4 层一个 `full_attention`，即 16 层全注意力），Q 头 24 / KV 头 4 / head_dim 256，GDN 为 16 个 key 头 ×128 + 48 个 value 头 ×128 + conv kernel 4，intermediate 17,408，vocab 248,320，带 MTP head。KV：BF16 **65,536 B/token**，INT8 约 33,792 B/token。
参数量约 27 B ⇒ 每 token 前向 **≈54 GFLOP**。artifact 体积：`q4_k_m-v3` 17.85 GB、`coder390-q4lynnstyle-q8mtp` 20.13 GB、`nvfp4-w4a4-mtp-dflash2` 23.77 GB。

### 4.2 上限与实测【实测上限 + 文档实测值】

| 维度 | 上限（4×T10） | 实测最好 | 达成率 |
|---|---:|---:|---:|
| 8K prefill，FP16 张量核线 | 4×88.5 TF/s ÷ 54 GF = **6,550 tok/s** | vLLM W4A16 Marlin **1,547**；NInfer q4q5+cuBLAS **1,113** | 24% / 17% |
| 8K prefill，INT8 张量核线 | **13,100 tok/s** | — | 尚未有人触碰 |
| 64K prefill（同 FP16 线） | 6,550（忽略注意力项） | vLLM **1,269**（Definitive FP8）/ 1,330（Marlin）；NInfer **1,008** | 19% / 15% |
| decode 权重带宽线（TP4，无投机） | q4lynn：375 ÷ 5.03 = **74.5**；q4_k_m：**84.1** tok/s | NInfer 4K 普通 **47.9**；TP4 64K **44.0** | 64% / 59%（64K 还要扣 KV：INT8 64K 上下文每卡 0.55 GB/token → 线降到约 67） |
| MTP3 有效 decode 线（每轮摊 2.96 token） | ≈220 tok/s | NInfer q4lynn 4K **94.8**，64K **97.0** | 43% |

要点：
- **prefill 距算力线 4–6×，距带宽线远得多**：一个 chunk 只读一遍权重（5 GB/chunk，4096 token → 1.23 MB/token → 3.3 µs/token），所以 prefill 是 **issue/kernel 效率** 问题，不是搬数问题。反例证据：TITAN 上 A8 GEMM 把一个 chunk 切成 9 个 128-token tile、每个 tile 重读整块权重，MLP-down 从 560 ms 涨到 808 ms【文档】——T10 上同样要保证"每 chunk 只读一遍权重"。
- **decode 距带宽线 1.5×**，其中 rank 0 的额外负担与通信是可以直接算的那部分。

### 4.3 已知的耗时来源（来自既有 Nsight 记录，作为选题输入而非结论）【文档】

- ninfer 四卡 8K（修复前）：Q5 A16 内核 48.9% + Q4 A16 21.0% → 69% 时间在 A16 路径；修复 PartProjection 后 prefill 从 436 → 1,113 tok/s。
- q4q5 + BF16 store 版本残留：Q4/Q5 A16 sliced-K 约 28%、GDN `output/prepare_wy_wu/state_passing` 约 15.6%、`store_fp32_epilogue` 5.3%、token int8 量化与 channel absmax 6.2%、权重反量化 int8 3.4%。
- q4lynn（GGUF）版本：时间散在多种 `mul_mat_q` 内核上，与 row-split 路线的 cuBLAS 优化互不适用。
- TITAN 侧共性：**Q4 K-split / Q5 small-T 内核只跑到显存带宽的约 1/3**；GDN 两阶段递推相对 3090 慢 11×（TF32→FP32 乘积）。

---

## 5. 多卡通信包络（本轮实测重点）

### 5.1 ninfer 在**这两类卡上自测**的 TP 交换表【实测日志，`profiles/bench/t10-optimization/**/server.log`】

| 拓扑 | 路由与门限（启动时实测） |
|---|---|
| TP2 | `peer-write words` ≤ **32 KiB（9.6 µs）** → `peer-read whole` ≤ 216 KiB（26.1 µs）→ `peer-write scatter-flags` ≤ 1 MiB（81.4 µs）→ `peer-read scatter` 1.06 MiB（87.0 µs）→ `peer-write scatter` 70 MiB（4,645 µs）；copy-engine 行传输从 64 KiB 起 |
| TP4 | `peer-write words` ≤ **15 KiB（17.7 µs）** → `peer-write whole` ≤ 29 KiB（22.5 µs）→ `scatter-flags` ≤ 54 KiB（25.6 µs）→ `scatter` 480 KiB（116.7 µs）→ `scatter-flags` 608 KiB（148.8 µs）→ `scatter` 70 MiB（15,703 µs）；copy-engine 行传输从 128 KiB 起 |
| 复现性 | 另一次 TP4 启动：words 12.5 KiB **17.0 µs**、whole 30 KiB 23.0 µs、scatter-flags 54 KiB 26.6 µs（两次相差 <4%，可作为标定基准） |

解读：
1. **TP4 的小消息延迟比 TP2 高 84%**（17.7 vs 9.6 µs），因为 4 卡里 2 个 peer 要跨 PHB（9.9 vs 12.7 GB/s），且 epoch 轮询要等最慢的一路。
2. **`peer-write` 在 T10 上压倒性胜出**（自动选择的结果全是 write），与我另写的 4-rank push/pull 探针一致：push 在 20/80/256 KiB 分别是 pull 的 2.2–2.6× 快（29.0/62.9/177 µs vs 48.4/111/322 µs，1024 线程）【实测】。
3. 64 层 × 每层 2 次 all-reduce ≈ 128 次交换：decode（hidden 5,120×FP32 = 20 KiB）落在 `whole`/`words` 区间，**每 token 约 2.3–2.9 ms**；MTP4（4 列 = 80 KiB）落在 `scatter-flags`，**约 3.3 ms**。对应 47.9 tok/s 的 **11–14%**。这不是第一瓶颈，但解释了 TP4 相对 TP2 只有 1.38× 而非 2×。

### 5.2 可以动的地方

| 方向 | 依据 | 落点 |
|---|---|---|
| rank 排序贴 PIX 亲和（把通信最重的 rank 放 0/1 与 2/3 内部） | 拓扑 PIX/PHB 差异实测 28% 带宽差 | `--devices` 顺序 + `--tensor-split`；必要时在 `src/core/tensor_parallel.cu` 的 rank 映射里按 `nvmlDeviceGetPciInfo`/topo 排序 |
| 用 `--tensor-split` 补偿 rank 0 的 head/MTP 字节 | CLI 文档：head、shared expert、MTP 都在第一张卡；`--tensor-split` 支持按卡份额 | 纯命令行实验，无需改码 |
| prompt 分片降位宽 `--tensor-parallel-prompt-partials bf16\|int8` | 文档称分别省 1/2 与 9/32 字节；本轮未见 T10 实测记录 | 只在 prefill 通道生效，需固定 NIAH 响应核对 |
| 门限本机重标定（words/whole/scatter 的边界、copy-engine 起点） | ninfer 已在启动时自测并写日志；vLLM fork 的 `custom_all_reduce.cuh` 注释写明同拓扑 "two-stage is already faster from 32 KiB onwards" | 与 `--tensor-parallel-transport` 强制对照，做 A/B |
| 混合 TP2×PP2（把 all-reduce 关在 PIX 对内） | 现在 `--devices` 是"每卡一个 pipeline stage"**或**全 TP，没有混合模式 | 需要新功能：`--stage-layers` 与 TP 组合；改动在 `src/models/qwen3_5/program/planning/startup.*` 与 `src/core/stage_link.h` |
| 通信与 sum/RMSNorm/下一层投影的融合 | 1Cat/123123213weqw 的做法（v100 调研 §4.4），ninfer 已有 Words 融合形态 | `src/ops/common/allreduce*`（v100 树）对应 ninfer-all 的 `src/core/tensor_parallel.cu` |

---

## 6. 本地已有实现与在途实验（不要重复劳动）

| 位置 | 现状 | 对调研的影响 |
|---|---|---|
| `/public/src/ninfer-all` | 未改动，`75` 构建受支持（`NINFER_ARCH_TURING`、`src/ops/common/{device_tier.h,pre_sm80.cuh,mma.cuh,sm75_mma.cuh}`） | 移植目标；tier 宏是唯一合法的 arch 分支入口 |
| `/public/src/ninfer-t10` | 5 份 `docs/performance/tesla-t10-*.md` 记录了已完成的工作：TP rank 的 cuBLAS policy 传播、FP32/BF16 partial store、`PartProjection` 共享 cuBLAS 投影、SM75 默认关 prompt pieces（`NINFER_TP_PROMPT_PIECES=1` 回退）、GDN workspace family 修正；`src/ops/linear/gguf/ggml_marlin_sm75.cu`（218 行，`mma.m16n8k8`，Q4_K/Q5_K 直接解包成 fragment，FP16 activation + 每列 2 的幂范围，FP32 累加，**不重排 GGUF**） | 已有成果与在途实验都在这里；改 Marlin 前先读 `docs/performance/tesla-t10-prefill-formats-engine-2026-11-15.md` 的 Nsight 残留表 |
| Marlin A/B（今日） | 8K NIAH、TP4、q4lynn：marlin **12.92 s**（prefill 595 tok/s，decode 47.9）；`marlin-j128` **10.12 s**；MMQ 默认 **8.42 s**（约 912 tok/s） | 当前 Marlin 比 MMQ 慢 1.11–1.53×，与"仍不及 vLLM"一致 |
| device profile | ninfer-t10 已有 `nvidia-tesla-t10-sm75`（**87 routed keys，calibration**，2026-11-15）；`ninfer-all/src/runtime/engine/device_profiles.json` **没有 T10 条目**（sm75 只有 TITAN/2060S/2070/2070S/2080Ti/Quadro 6000/Quadro 8000/Tesla T4） | 上游缺 T10：把标定结果按 `docs/device-profiles.md` 的格式提上去，是明确的小贡献 |
| `--tensor-parallel` 语义 | 文档明确：GDN 头、KV 头连同其 Q 头与 cache、稠密 FFN 行按卡切；**lm head、shared expert、MTP 留在第一张卡**；但 decode/verify 走 **divided head**（`src/models/divided_head.h`：其他 rank 把自己的 logits 通过自己的链路回给 rank 0），prefill 只投最后一列 | rank 0 串行点的准确形态要在 profile 里量化后才能定优化方向 |
| 探针 | `tools/{hbm_bandwidth_probe.cu,fma_rate_probe.cu,int8_gemm_reference_probe.cu}`；本轮另加 `t10_probe2.cu`（带宽/TP 交换）、`tc_probe.cu`（张量核吞吐）、`asm_probe.cu`+`genasm.py`（指令可用性） | 附录 A 给复现命令 |

已在 T10 上**没有**留下实测记录、但开关现成的（建议先跑，不改码）：`--tensor-parallel-prompt-partials bf16|int8`、`--tensor-attention positions`、`--tensor-split`、`--tensor-parallel-transport host|peer-read|peer-write` 强制对照、`--prefill-chunk` 在 128K 下的 8192→16384、`NINFER_LINEAR_ROUTES` 手工改路由。

---

## 7. 抄考对象：具体读哪里

### 7.1 本机就能读的对照物（最高优先）

**`/public/src/vLLM-2080Ti-Definitive`（就是要在 prefill 上打赢的对象）**

| 文件 | 值得抄的机制 | 落到 ninfer 哪里 |
|---|---|---|
| `csrc/libtorch_stable/quantization/marlin/marlin_mma.h` | `#if __CUDA_ARCH__ == 750` 把 `m16n8k16` 整段换成 `m16n8k8`，INT8 换成 4 条 `m8n8k16.satfinite`；同一模板同时覆盖 f16/f32 累加、fp8(`e4m3`)、fp4(`e2m1`)、s8/s4 | 与 `src/ops/common/sm75_mma.cuh` 的封装方式对照，统一 fragment 映射 |
| `.../marlin/dequant.h`、`marlin_template.h` | 4-bit→f16 的 lop3/prmt 常量表、per-group scale 的 shuffle-free 取法、`act-order`、split-K 小 M、stages=4 的 smem 预算（`marlin.cuh`：4 pipeline stages、`max_thread_n=256`、`max_par=16`） | `src/ops/linear/gguf/ggml_marlin_sm75.cu` 目前只有 4 warp / J=128 / stride=136，缺 stage 与 thread-config 空间 |
| `csrc/custom_all_reduce.cuh` | 1stage/2stage 选择，注释写明"**在四张 SM75 PCIe 卡上，32 KiB 起 2stage 就已经更快**"；支持 `VLLM_*_ALLREDUCE_ALGO=1stage/2stage` 强制 | 与 ninfer 的 `words/whole/scatter` 门限互校；`--tensor-parallel-transport` 的 A/B 设计 |
| `csrc/libtorch_stable/quantization/w8a8/cutlass/scaled_mm_c2x_sm75_dispatch.cuh` | CUTLASS C2x 在 sm75 上的 W8A8（INT8 TC）tile/dispatch 组合 | A8 路线（`linear_swiglu/q4a8`、`linear/t2`）的 tile 选择参考 |
| `.../moe/marlin_moe_wna16/sm75_kernel_float16_fe4m3fn_float16.cu` 等 | FP8 / NVFP4 权重在 sm75 上**在寄存器里反量化成 FP16** 再喂 TC（不落第二份权重） | 对应 ninfer 的 `nvfp4_int8` / row-FP8 尝试（后者在 SM75 上只有 210 tok/s，值得查是不是走了 host 反量化） |
| `vllm/model_executor/layers/quantization/fp8.py:443` | "direct FP8 不支持时用 BF16 反量化"——解释 Definitive 跑 FP8 checkpoint 的真实路径 | 做同权重对照时必须注明这一点 |
| `/public/llm/sm75-vllm-qwen3.8.sh` | 基线的真实参数：FP8 artifact、`--dtype float16`、`--kv-cache-dtype fp8`、TP4、MTP `num_speculative_tokens=3`、`max-num-batched-tokens 2048`（文档测到 8192 只 +2.2%）、`FULL_AND_PIECEWISE` 图捕获 `[4,8]`、`gdn_prefill_backend=flashqla_legacy`、custom all-reduce **未禁用** | 对照口径；"prefill 差异主要来自 kernel 而非 chunk" 的证据 |

### 7.2 上游项目（按 sm75 相关性排序）

| 项目 / 变更 | 借什么 | 边界 |
|---|---|---|
| vllm-project/vllm **PR #29901**「add marlin kernel support for turing (sm75)」（2025-12-16 合并）+ **#31000**（compressed-tensors 打开 Turing） | 官方 Turing Marlin 的 diff：哪些 shape/tile 被砍、repack 与 scale 布局如何适配 64 KB smem | 【待核实】具体 diff 细节本轮未读；官方文档表已写 "Marlin：Turing ✅\*"（\* = 受限） |
| ggml-org/llama.cpp **PR #7676 / #7860**（MMQ 上 INT8 张量核，q4_K/q5_K/q6_K，"Turing 或更新"） | ninfer 的 `src/ops/linear/gguf/ggml_mmq_*.cu` 已是同源 vendored 代码：核对 **Turing 的 `m8n8k16` 路径是否真的启用**、J/K tile 与 `MMQ_ITER_K` 是否按 56 SM/64 KB smem 调过 | 【实测】本机 profile 显示时间散在多种 `mul_mat_q`；需要逐内核确认走的是 MMVQ 还是 MMQ |
| llama.cpp **PR #23729**（`MMVQ_PARAMETERS_TURING`，MMVQ 在 SM75 上 89 寄存器/线程导致占用掉 61%） | 现成的 **SM75 寄存器/占用调参表**，直接指导 `gguf_linear.cpp` 里 MMVQ 的 launch 参数 | 【待核实】合并状态与最终参数需再看 |
| exllamav3 的 **Turing 分支/fork**：`weicj/exllamav3-turing`、`rafatxf/exllamav3`（`turing` 分支，README 称 "flash-attention prefill、4-bit cache flash-decoding、GDN kernels"） | 上游 issue #162 把 sm75 掉速原因写清了：**smem 少（限制 shape 与 staging）、缺 barrier/async copy、Triton 内核退回 CUDA core**；fork 的 `doc/turing.md` 与 `doc/sm75.md` 是现成的 sm75 版 GDN + 量化 KV flash-decoding 参考 | 【待核实】fork 的实现质量与许可证；模型族不同（EXL3 codec） |
| Dao-AILab/flash-attention：**FA2 需要 Ampere+**，Turing 走 FA1.x 或 README 链接的 flash-attention-turing 仓库（PR #2379，2026-03） | 提示 attention 预fill/decode 在 sm75 上没有现成的 FA2 可抄，只能自有内核 + Turing tile 调参（TITAN 文档的 16-key FP8 tile、宽度 512 的 32 列 tile 就是同类工作） | 【待核实】 |
| `wjie98/comfyui-turing-utils` | 宣称提供 sm75 的 W8A8/W4A8/W4A4 与 "exact-sm75" 精度路径，含 cubin/PTX | 【待核实】面向图像生态，但 INT8 TC GEMM 的 sm75 写法可查 |
| imisszxq/qserve（W4A8KV4） | 算法层可借（W4A8 的精度配方与 per-group scale 顺序）；**其内核面向 A100/L40S**，`m16n8k32` 与 fp8 都在 sm80+ | 【实测】指令边界已确认，不能直接编 sm75 |
| TensorRT-LLM | 不作为 sm75 参考（Turing 支持历史上就不完整，近期 release notes 无 sm75 方向） | 【待核实】 |
| 1CatAI/1Cat-vLLM、geoffwatts/ninfer-v100、plus1998/Ninfer-V100-Duo、v100-skinny、HOOLIGAN、voltattn、marlin_v100、sglang-V100、Augur-V100（v100 调研的主清单） | 仍然值得读：**GQA-packed verify、split-KV 与 SM 数匹配、K/V 同时发射、push 归约、完整轮次 A/B 方法**。但它们的 QPN/WMMA/NVLink 假设是 SM70 的，sm75 有 INT8 TC、无 bf16、smem 64 KB、纯 PCIe | 借鉴机制，不搬指令与精度边界 |
| 本机 `/public/src/fastllm`、`/public/src/Strata`、`/public/src/Turing-unlock`、`/public/src/cuda-samples` | fastllm/Strata 的**多卡层切分**是 PCIe-only 环境下与 TP 正交的另一条路（vLLM/exllama 也是模型并行优先）；Turing-unlock 若能动 vBIOS 的功耗/电压上限，是"提高整张卡上限"的物理手段；cuda-samples 里 `p2pBandwidthLatencyTest` 可作交叉验证 | 功耗上限目前驱动侧不可调（`power.max_limit = 150 W`），vBIOS 是否可改**未验证** |

---

## 8. 待引入 / 继续优化清单（按"预期收益 ÷ 成本"排序）

| # | 方向 | 依据与预期 | 判定方法 | 风险 |
|---|---|---|---|---|
| A | **零代码实验批**：`--tensor-split`（补偿 rank 0 的 head/MTP）、`--tensor-parallel-prompt-partials bf16\|int8`、`--tensor-attention positions`、`--tensor-parallel-transport` 强制三态、128K 下 chunk 8192→16384 | 每项都可能给 prefill 3–15%、decode 5–10%；成本是几小时 profile | 固定 NIAH 8K/64K/128K + MTP3 长输出，两次交错；响应正文必须仍为 `ORCHID=493817; COLOR=COBALT` | int8/bf16 partials 会改变输出与 perplexity，须单列质量影响 |
| B | **Marlin 的前提问题**：要么按 Marlin 的前提在加载期 repack Q4_K（每卡分片一次），要么放弃 Marlin | 现状 595 vs MMQ 912 tok/s；repack 后目标至少追平 MMQ，再谈 2× | 先做单算子 oracle（FP64 对照 + guard bytes），再 8K/64K 整请求 | repack 增加加载时间与每卡显存；GGUF 的 6-bit 子 scale 需要新布局约定 |
| C | **INT8 张量核覆盖 q4lynn 的 prefill**（W4A8/A8 家族扩形状、复用已标定的 T10 profile） | INT8 线是 FP16 的 2×；T10 实测 178 TOPS；Nsight 显示仍有 28% 时间在 Q4/Q5 A16 | 逐 shape 记录残留 A16 调用的 (N,K,T,dtype)，只对实测胜出的 shape 开 gate | 精度：A8 已有权衡文档；必须并行报 ppl/固定响应 |
| D | **GDN prefill 摆脱 FP32 乘积**（把递推的操作数边界改成 FP16 可表达，或按 TITAN 的两阶段法在 sm75 重标定） | TITAN 上 11× 差距；T10 上 GDN 三阶段占 prefill 约 15.6% | 独立 FP64 点积/conv oracle；两阶段 vs WY 在 56 SM 上的实测 | 状态精度（FP32 持久状态）不能降；commit/rollback 语义要保持 |
| E | **通信：本机门限标定 + rank 贴 PIX + 与 sum/RMSNorm 融合** | 交换已占 decode 的 6–14%；PIX/PHB 差 28%；vLLM 在同一拓扑上 2stage 门限是 32 KiB | 用 `t10_probe2 exchange` 与 ninfer 启动自测表交叉核对，再测整请求 committed wall | 融合会改归约顺序 → 必须固定求和顺序或明确声明非无损 |
| F | **decode 带宽侧**：head/MTP 的 rank 0 串行量化；每卡字节约束（同精度选更小的 artifact：q4_k_m 84.1 vs q4lynn 74.5 上限） | 实测 47.9 对 74.5 线（64%） | 每 rank 的 kernel 时间 + 空闲时间（Nsight，`--cuda-graph-trace=node`） | 换 artifact 会换质量，不能当纯内核收益 |
| G | **混合 TP2×PP2**（把 all-reduce 关在 PIX 对内） | 若 E 证明跨 PHB 是主因，则值得做；否则优先级低 | 与 TP4 同 fixture 对比 decode/prefill/TTFT | 新功能，改动面大；PP 的 bubble 未评估 |
| H | **上游贡献**：T10 device profile、SM75 tier 的分歧修复合回 `ninfer-all` | 减少与上游的分叉漂移 | `ctest` + 固定响应 + ppl 复核 | 上游对数值一致性的要求更严 |

### 选题时最容易走错的六件事

1. 把 sm80 代码当"改个 arch 号就能跑"：`cp.async`/`m16n8k16`/`mbarrier`/`redux` 在 sm75 上**编译期就失败**（§3 实测）。
2. 借用邻近卡型（2080 Ti / TITAN / Quadro 6000）的 device profile 或 tile 参数：T10 只有 56 SM、403 GB/s、64 KB smem、150 W，差 20–30%。
3. 把 "Marlin" 当成一个内核而不是一整套前提（repacked 权重 + 固定 fragment 布局 + 多 stage 流水 + 每 shape 的 thread config）。少了前提的实现必然输给 MMQ。
4. 拿不同 artifact / 不同 KV 精度 / 不同 draft 的数字互相换算。本仓库已有 4 组互相打架的基线数字（Definitive FP8、mainline Marlin、q4lynn、q4q5），每次比较都要写明 artifact 与口径。
5. 只测算子不测整请求：本仓库多次出现"算子快了、整机慢了"（v100 侧同样）。判定单位是完整轮次与 committed wall throughput。
6. 忘记 150 W 封顶：本轮 ALU 满载时 clocks 仍是 1,575–1,590 MHz，但**张量核密集负载下的 clock/power 轨迹本轮没测**，做算力类假设前应先补一次带 nvidia-smi 同步采样的 prefill 实测。

---

## 9. 最小有效验证

沿用 `docs/performance/methodology.md` 的口径（`--no-prefix-reuse`、greedy、CUDA Graphs、固定 fixture 与 seed），本轮建议的最小验证集：

```bash
cd /public/src/ninfer-t10
MODEL=/public/llm/models/qwen/ninfer-qwen3.8-27b/qwen3.8-27b-coder390-q4lynnstyle-q8mtp.ninfer   # decode 最优
# MODEL=/public/llm/models/qwen/ninfer-qwen3.8-27b/coder390-fp8-to-q4q5-proposal.ninfer          # prefill 最优

build/apps/ninfer-serve "$MODEL" \
  --devices 0,1,2,3 --tensor-parallel \
  --prefill-cublas --prefill-chunk 4096 \
  --embedding-residency host \
  --max-context 66560 --kv-capacity auto --kv-dtype int8 \
  --max-concurrency 1 --spec mtp --draft-tokens 3 --adaptive-mtp \
  --ngram-draft-tokens 0 --mtp-attention-window 4096 \
  --host 127.0.0.1 --port 18080
```

判定阈值（先小后大）：
1. **算子级**：新增/修改的 GEMM 必须过 FP64 oracle、guard bytes、只读输入检查（沿用 `ninfer_w4_cublas_prefill_test`、`tests/ops/linear/test_gguf.cpp` 的写法）。
2. **通信级**：`t10_probe2 exchange` 与 ninfer 启动自测表都要复测两次交错；单次改动若未在 ≥3 个尺寸上更好，就不改门限。
3. **整请求**：8K / 64K（+ 必要时 128K）× {MTP0, MTP3} × 两次交错；响应正文一致才允许宣称无损；输出 ID 变了就按"改变计算关联"记录。
4. **并发**：4K C1/C2/C4/C8 与 64K C1/C2（本轮文档已确立 C2 是 MTP3 的最佳点，C8 是普通的）。
5. 只有 1–4 都成立，才进入 profile 记录与提交。

---

## 附录 A：本轮新增测量与复现

探针源码（本地 `tmp/t10-survey/`，已上传到 `wsdev:/public/src/ninfer-t10/profiles/probe-20261115/`）：

| 文件 | 用法 | 测什么 |
|---|---|---|
| `t10_probe.cu` | `nvcc -O3 -arch=sm_75` → `./t10_probe info` | 设备属性、P2P 能力矩阵 |
| `t10_probe2.cu` | `./t10_probe2 bw` / `exchange` / `burn` | triad 带宽、P2P 单双向与 32 KiB 拷贝、TP4 push/pull 交换（1/20/80/256 KiB）、满载 clock/power |
| `tc_probe.cu` | `./tc_probe all` | FFMA / m16n8k8 f16(f32,f16 累加) / m8n8k16 s8 吞吐 |
| `asm_probe.cu` + `genasm.py` | `python3 genasm.py` | 每条 PTX 在 sm_75 与 sm_80 下的可用性对照 |
| `run_burn.sh` | `sh run_burn.sh` | 满载期间并行采样的 nvidia-smi 读数（结果在 `smi.log`；本轮 `burn.log` 未保留，峰值结论见正文：1,575–1,590 MHz @140–152 W） |

注意事项（踩过的坑，供复现者）：
- 后台任务要用 `setsid nohup ... </dev/null >log 2>&1 &` 并在脚本里等待，否则 paramiko 的 SSH 通道会被挂住；`pkill -f` 的模式若出现在同一条命令里会自杀。
- `cudaDeviceProp.clockRate/memoryClockRate` 在 CUDA 13 已移除，用 `cudaDeviceGetAttribute(cudaDevAttrClockRate/…)`。
- 跨卡 ping-pong 自旋：两个 kernel 必须在**各自设备**上创建 stream 并 launch，否则两个自旋 kernel 落在同一张卡上会假死；跨设备读的可见性要靠 `__threadfence_system()` + `ld.acquire.sys/st.release.sys`。
- 事件必须在其所属设备的当前上下文里 `cudaEventCreate`，否则 `cudaEventElapsedTime` 返回 0（本轮 P2P 单向带宽就因此失真，**只有双向数字有效**，重跑时已修）。

## 附录 B：与 V100 调研的对应关系

| 主题 | V100 调研（SM70） | 本文（SM75） |
|---|---|---|
| 互联 | 双卡 PCIe；远端四卡 **NVLink** | **纯 PCIe Gen3**，PIX/PHB 两档，P2P 全通 |
| 张量核 | WMMA/FP16，INT8 只能 `dp4a`；FP16-accum 为主 | FP16 TC 88.5 TF/s，**INT8 TC 178 TOP/s 可用**，无 bf16/fp8/tf32 |
| 主要瓶颈 | 85K/MTP4 decode 的 attention 访存与通信、QPN 调度 | prefill 的量化 GEMM/GDN 算力利用率；decode 的 rank 0 串行与每卡字节 |
| 通信结论 | push 优于 pull、packet 尺寸门限、跨层 sum/norm 融合 | 同样 push 胜出（2.2–2.6×），但门限与形状完全不同，**必须本机标定** |
| 内核移植雷区 | 缺 TC 特性（fp16-only）、WMMA 形状 | 缺 `cp.async`/`m16n8k16`/`m16n8k32`/`mbarrier`/`redux`/smem 64 KB |
| 方法 | 完整轮次 + committed wall + FP64 oracle + 两次交错 | 同左（本文 §9） |

---

## 附录 C：`ninfer-v100xN`（SM70/Volta）的可迁移性判定

用户提问：V100 架构更老，它的实现方法还值得参考吗？答案是**分层**的，而且可以用一条实测数据说清：

| 内核（单卡 T10，预热后计时）| 吞吐 | 相对最优 |
|---|---:|---:|
| `mma.m16n8k8` f16→f32（sm75 的正路 shape） | **88.9 TFLOP/s** | 1.00 |
| `mma.m8n8k4` f16→f32（**Volta 的 quarter-precision shape**，操作数取自 `ninfer-v100xN/src/ops/common/volta_mma.cuh:119-124`） | **11.3 TFLOP/s** | **0.13（慢 7.9×）** |

【实测】两个 shape 在 sm_75 上**都能编译**（`sm_75/sm_80/sm_86` 全 PASS），所以 V100 代码搬到 T10 不会"编不过"，而是**默默只用掉 1/8 的张量核**。因此判定必须按层做。

### C.1 三层判定框架

| 层 | 是否与 arch 绑定 | 对 T10 的态度 |
|---|---|---|
| 引擎 / 协议 / 调度（TP epoch 协议、图捕获、admission、投机轮次、验收方法） | 否 | **直接参考**，几乎无折扣 |
| 数值与 codec（FP8/FP4 码的位操作解码、scale 折叠、数值边界与求和顺序声明） | 否（反而更必要） | **直接参考** |
| 内核 tile / warp / 寄存器组织 | **强绑定** | **只取策略，不取常量**，且必须换成 sm75 的 shape |

### C.2 逐机制判定（源码位置均为 `wsdev:/public/src/ninfer-v100xN/`）

| 机制 | 位置 | sm75 事实 | 判定 |
|---|---|---|---|
| TP2/TP4 two-epoch push mailbox：本地贡献留在寄存器、按全局 rank 顺序 FP32 求和、一次 BF16 舍入、offset 与 epoch 域与消息大小无关 | `src/ops/kernel/allreduce.cuh:86-183` | 纯指针/volatile，无 arch 指令；本机实测 push 比 pull 快 2.2–2.6×，`peer-write` 是自动选择结果 | **直接搬机制**；`ninfer-all` 的 `src/core/tensor_parallel.cu` 已有 Words/PeerWrite/Scatter 对应物，缺的是跨层 sum/norm handoff 与本机门限 |
| 批量 projection→packet→sum/RMSNorm、channel 尺寸扩到 640 KiB 以免小消息退回多步路径 | 同上（1Cat 适配） | arch-neutral；但尺寸门限依赖链路带宽 | 搬机制，**参数按本机重标**（PIX 12.7 / 跨桥 9.9 GB/s；交换 17.7–26.6 µs） |
| `e2m1 / e4m3 → half2` 的 shift+rebias 解码；把 2⁻¹⁴ 折进 group scale（`gscale * 16384`）；e4m3 掩码必须 `0x3F80` 而非 `0x7F80`（否则负 scale 溢出成 inf） | `src/ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cuh` 头注（引 v100-skinny / TurboMind） | sm75 既没有 FP8/FP4 的 mma，也没有 sm_89 的 `cvt.e4m3x2`【实测】→ 纯位操作解码是**唯一**低成本路 | **直接参考**（数值边界按本项目 artifact 复核） |
| QPN（权重按 N 打包、warp 内 quadpair 沿 N 分工、按 shape 选 split-K、独立累加链、checkpoint-native 不预打包） | `nvfp4_volta_qpn_gemm.cuh`、`q4_volta_qpn_gemm.cuh`、`w8_volta_qpn_gemm.cuh`、`fp8_volta_qpn_gemm.cuh` | 布局与并行度思想 arch-neutral；但 quadpair 的 warp 映射是按 m8n8k4 的 fragment 分布设计的 | **换指令后搬**：sm75 要改成 `ldmatrix` + `m16n8k8`（仍无 `m16n8k16`【实测】），寄存器映射重做 |
| Volta INT8 attention 的"TC 路线" = 把量化值反量化成 FP16 再喂 `m8n8k4` | `src/ops/kernel/gqa_attention_decode_i8_tc_volta.cuh` | T10 有**真 INT8 TC**（`m8n8k16`，实测 178 TOP/s）→ 不该再走 INT8→FP16→TC | **别搬指令**；搬它的 GQA 行复用、K/V 同时发射、`B×H×splits` 的 CTA 并行度组织 |
| tile/warp 常量与"寄存器墙"推导 | 同上文件头注（明写 "Ampere 的 ldmatrix PV tile 每线程 2 行×2 列，Volta 的 m8n8k4 只有 1 行×8 列"） | sm75 有 `ldmatrix`【实测】→ 属于该注释里的 **Ampere 侧** | **别搬常量**；直接按 Ampere+ 的 tile 拓扑起步，再按 64 KB smem 与 56 SM 缩 |
| GDN verify / record 的 value tile（2 warp × 4 value 行 vs 上游 4 × 4）、周边 norm/stage/小投影融合 | `src/ops/linear_attention/gated_delta_net/recurrent.cuh`、Augur 报告 | 融合与"共用输入"机制 arch-neutral；但 V100 每 SM **2048 线程**，T10 只有 **1024**【实测】 | 搬机制，**占用与 tile 尺寸必须重调** |
| prefill 步间执行多轮 decode（`--prefill-decode-rounds`）、有 decoder 时缩小 prefill slice + admission 计费 | PR #3（90 文件 / +10751 行，**未改 `src/ops` 的 CUDA 算子**） | 纯调度层 | **值得参考**：T10 的 64K C2/C4/C8 实测显示聚合 prefill 不随并发提升（多卡执行模型不做跨请求 prefill 批处理），这类"让已 admitted 流在 prefill 窗口内继续吐字"的机制正好对症 |
| 验收方法：committed wall、完整轮次、两次交错、独立 FP64 oracle、输出 ID 分叉位置必须记录 | `AGENTS.md`、`docs/performance.md` | 完全 arch-neutral | **直接搬**（正文 §9 已采用） |

### C.3 环境侧的两个坑（本轮实测）

- `ninfer-v100xN/CMakeLists.txt` 只接受 `70 | 86 | 89`，**没有 75 tier**，且 "Volta sm_70 requires CUDA 12.8; CUDA 13 removed Volta compilation"。所以 V100 树本身不会编到 T10 上：移植 = 把机制重写到 `ninfer-all` 的 Turing tier（`NINFER_TIER_TURING` / `src/ops/common/sm75_mma.cuh`）。
- 本机 **CUDA 12.8 的 `cuda_fp16.h`/`math_functions.h` 与 Debian 13 的 glibc 头冲突**（`cospi` exception specification 报错），CUDA 13.4 正常。想在同一台机器上交叉参考 V100 路线的构建，需要绕过 12.8。

### C.4 数字换算的硬约束

V100 侧的漂亮数字几乎都带不可迁移的前提：四卡 **NVLink**（T10 是 12.7/9.9 GB/s 的 PCIe，差 20×以上）、**32 GB** 显存（T10 16 GB）、不同 artifact 与 KV 精度（E4M3/FP16 vs INT8-G64）、220 W 限功率。可用的是**机制、并行度组织、扩展趋势和负结果清单**（1Cat 记录过多个 QPN unroll / 寄存器限制 / 预取变体反而更慢），不是绝对吞吐。

### C.5 建议的读取顺序（若只有一天）

1. `src/ops/kernel/allreduce.cuh`（TP2/TP4 push 协议与不变量）→ 对照 `ninfer-all/src/core/tensor_parallel.cu` 的 Words/PeerWrite。
2. `nvfp4_volta_qpn_gemm.cuh` 的头注（e2m1/e4m3 位解码与 scale 折叠）→ 直接服务 T10 的 FP8/NVFP4 与 Marlin-like 解码路径。
3. `AGENTS.md` + `docs/performance.md` 的验收口径与负结果 → 避免在 T10 上重复踩。
4. PR #3 的 prefill/decode 交错与 admission → 对应 T10 并发场景的缺口。
5. 最后才是 attention/QPN 的 tile 常量，且只作为"为什么这样选"的论证材料。
