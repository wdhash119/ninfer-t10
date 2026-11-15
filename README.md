# NInfer-t10

C++/CUDA inference work aimed at the **NVIDIA Tesla T10 (SM75)**. The machine is a workstation with
4× Tesla T10 16 GB and nothing but PCIe between the cards. Current effort goes into Qwen3.8-27B:
prefill, plain decode, MTP speculative decoding, and how four cards split one model. Every number
here comes from a run whose report and raw logs are in this tree.

This is a **fork of [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all)**. 

The upstream documentation stays the reference for everything NInfer does; this README does not
replace it and does not claim that inherited features were checked on T10. What we maintain here is
narrower: configurations, kernels, performance reports and launch lines that have been measured on
T10.

## Goals

1. Optimize inference kernels for SM75: quantized GEMM, GGUF MMQ, NVFP4, GDN, attention, and the
   Tensor Core paths around them.
2. Optimize execution across cards: TP2/TP4, PCIe P2P, the load rank 0 carries, prompt partials,
   pipeline and scheduling.
3. Keep prefill and decode apart in the reporting. We write down prefill, plain decode, MTP decode,
   steady state and aggregate concurrent throughput as separate figures and never add them up.
4. Compare fairly on weights from the same source, with one fixture shared by the
   vLLM-2080Ti-Definitive build on this machine and the older NInfer paths.
5. Keep every experiment revertible. The default path adopts nothing that measured slower; an
   experimental kernel ships off, and we only discuss enabling it after it passes numerical
   validation.

## The machine the numbers come from

| Item | What we measured on |
|---|---|
| GPU | 4× Tesla T10 16 GB, TU102GL, SM75, 56 SM per card |
| Interconnect | PCIe Gen3, P2P available, no NVLink; bandwidth inside one bridge differs from bandwidth across bridges |
| Host | Debian 13, Xeon E5-2698 v3, 48 GB RAM |
| Build | CUDA 13.4 / NVCC 13.4.92, CMake, Ninja, GCC 14 |
| Main model | Qwen3.8-27B Coder390 q4lynn; QUASAR NVFP4 as the experimental comparison |
| Main runtime | `build/apps/ninfer-serve`, with no NCCL dependency |

SM75 has none of the SM80 machinery: no `cp.async`, no `m16n8k16` or `m16n8k32`, no TF32 MMA, no
mbarrier. The T10 kernels run on `m16n8k8` FP16 Tensor Core, `m8n8k16` INT8 Tensor Core, `ldmatrix`,
DP4A and hand-built pipelines. An SM80+ kernel does not move over as it is.

## What runs today

### The production route

We recommend **q4lynn NInfer + TP4 + INT8 KV + MTP3**.

- 8K prefill at C1: about **1200 tok/s**
- 64K prefill at C1: about **1060-1080 tok/s**
- 4K MTP3 decode: about **96.6 tok/s**
- the fixed NIAH answer stays `ORCHID=493817; COLOR=COBALT`

The flags that carry it:

```text
--tensor-parallel-prompt-partials int8
--tensor-split 0.8,1,1,1
--spec mtp --draft-tokens 3 --adaptive-mtp
--ngram-draft-tokens 0 --mtp-attention-window 4096
```

An uneven `--tensor-split` is validated at C1. Under C4 with CUDA Graph the split has to be the even
`1,1,1,1`, otherwise GGUF shapes can mismatch. The long configuration (262K context, C4, vision,
CUDA Graph) starts and passes an API smoke test with little memory left over.

### The conservative quality tier

INT8 and BF16 prompt partials change how the reduction associates. A matching fixed NIAH answer is
not a claim of lossless math, and full perplexity has not been rerun. Where output quality is the
constraint we start from:

```text
--tensor-parallel-prompt-partials bf16
```

which measured about **1060 / 992 tok/s** for 8K and 64K prefill.

### Experimental routes

| Route | Where it stands | Switch |
|---|---|---|
| GGUF SM75 Marlin-like | correct numerically; about 759 tok/s at 8K against roughly 912 for MMQ | `NINFER_GGUF_MARLIN=1` |
| NVFP4 SM75 QPN | about 60.5 tok/s once the codec is vectorized, against 67.6 for W4A8 | `NINFER_NVFP4_QPN=1` |
| QUASAR NVFP4 W4A8 | good for comparing INT8 kernels, no replacement for q4lynn | `--nvfp4-w4a8-int` |
| row-FP8 | regresses on SM75 | not recommended |

The production defaults stay on **GGUF MMQ and the existing W4A8 paths**. Marlin, QPN and the other
numeric paths do not turn themselves on.

## Quick start

### Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DCMAKE_CUDA_FLAGS_RELEASE="-O3 -DNDEBUG"
cmake --build build --target ninfer-serve -j8
```

### Start the C1 optimized configuration

```bash
MODEL=/path/to/qwen3.8-27b-coder390-q4lynnstyle-q8mtp.ninfer   # from the nerkyor release, see Weights

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

For C4, change `--tensor-split` to `1,1,1,1`. 

## Weights and baselines

### The main artifact

[nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2](https://huggingface.co/nerkyor/Qwen3.8-27B-Coder390-EfficientThink-Opus5.5-GPT6Astra-Grok4.7-DSV4Pro-K3-SFT-RLOO-MTP-DFlash2)
`Qwen3.8-27B-Coder390-EfficientThink-Q4LynnStyle-Q8MTP.ninfer`, under `GGUF-NInfer/`. The same release carries the other
LynnStyle tiers, their GGUF twins, and the INT8/INT4/NVFP4 trees.

### The QUASAR experimental artifact

```text
qwen3_8_27b_nvfp4-quasar-proposal.ninfer
```
[wallawalla47/Qwen3.8-27B-Quasar-NVFP4-NInferV3](https://huggingface.co/wallawalla47/Qwen3.8-27B-Quasar-NVFP4-NInferV3)

Its INT8 comparison route:

```bash
build/apps/ninfer-serve /path/to/qwen3_8_27b_nvfp4-quasar-proposal.ninfer \
  --devices 0,1,2,3 --tensor-parallel \
  --nvfp4-w4a8-int \
  --spec mtp --draft-tokens 4 --kv-dtype int8
```

## Where this comes from

### Direct lineage

| Project | Relation |
|---|---|
| [Neroued/ninfer](https://github.com/Neroued/ninfer) | the original NInfer project |
| [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) | the direct upstream of this fork, supplying model execution, serving, quant formats, conversion tools and the test framework |

### Implementations we read

| Reference | What it contributed |
|---|---|
| vLLM-2080Ti-Definitive | SM75 Marlin, FP8/W4A16, TP all-reduce, FlashQLA, performance baselines |
| [vllm-project/vllm](https://github.com/vllm-project/vllm) | vLLM's architecture and its Turing Marlin work |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | the GGUF format, MMQ/MMVQ, how quantization kernels are organized |
| ninfer-v100xN | NVFP4 QPN, the T9-64 packed projection, MTP, TP push, sum/norm fusion, and how its results were accepted |
| [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) | TurboMind/QPN, E2M1/E4M3 codec, communication and reduction mechanics |
| [dnv2003/v100-skinny](https://github.com/dnv2003/v100-skinny) | QPN layout, K partitioning, separate accumulation chains, fragment organization |
| `geoffwatts/ninfer-v100`, `plus1998/Ninfer-V100-Duo` | study references for V100 scheduling, communication and optimization |
| CUTLASS / ggml / FlashQLA-SM70-SM75 | Tensor Core tiles, GGUF kernels, GDN prefill |

Reading a reference is not porting. Volta's `m8n8k4`, its tile shapes and occupancy, the NVLink
thresholds and SM80's async pipeline all stop being true on T10.

## Performance reports

Recent T10 reports live in `docs/performance/`:

- `tesla-t10-2026-11-14.md`: the first pass at SM75 TP prefill optimization
- `tesla-t10-part-projections-2026-11-14.md`: part projections and the cuBLAS paths
- `tesla-t10-mtp-concurrency-2026-11-15.md`: MTP against a concurrency matrix
- `tesla-t10-same-hf-comparison-2026-11-15.md`: comparison on weights from the same source
- `tesla-t10-definitive-baseline-2026-11-15.md`: the corrected Definitive baseline
- `tesla-t10-prefill-formats-engine-2026-11-15.md`: weight formats and the prefill engine
- `tesla-t10-sm75-marlin-prototype-2026-11-15.md`: the Marlin-like prototype
- `tesla-t10-survey-quasar-prompt-partials-2026-11-15.md`: prompt partials and tensor-split
- `tesla-t10-v100xN-quasar-portability-2026-11-15.md`: what ports from V100
- `tesla-t10-turing-qpn-first-stage-2026-11-15.md`: the first QPN stage
- `tesla-t10-turing-qpn-stage2-2026-11-15.md`: the vectorized codec and paired SwiGLU
- `tesla-t10-turing-qpn-stage3-2026-11-15.md`: K split and fragment sharing, which regressed

Upstream documentation starts at [docs/README.md](docs/README.md).

## License and thanks

The upstream `LICENSE`, `CONTRIBUTORS.md` and the notices under `third_party` stay where they are.
Code, algorithm notes and comments taken from reference implementations keep their original
attribution, and the license for a given file follows the project that file came from.

NInfer and this fork are licensed under Apache-2.0. See NOTICE for required attribution; 
third-party dependencies retain their own license files.