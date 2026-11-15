// Experimental SM75 QPN for QUASAR NVFP4 decode/verify widths. The codec and activation-supply
// organization follow ninfer-v100xN, while the warp computes m16n8k8 FP16 MMA with FP32 accumulators.
// The original BlockScaleK16M128x4 weight image is read directly; there is no second weight copy.

#include "nvfp4_launch.h"
#include "ops/common/device_tier.h"
#include "ops/common/math.cuh"
#include "core/device_launch.h"
#include "core/layout.h"
#include "ninfer/ops/silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

inline constexpr int kQpnWarps = 8;
inline constexpr int kQpnThreads = 32 * kQpnWarps;
inline constexpr int kQpnKSplits = 4;

// Eight E2M1 codes are decoded with shifts and a 2^14 rebias. Output pair j contains codes
// (j,j+4), so qpn_stage_pairs applies the same permutation to activations in registers.
__device__ __forceinline__ void qpn_decode_e2m1(std::uint32_t q, half2 rebias,
                                                 half2 (&out)[4]) {
    constexpr std::uint32_t kSign = 0x80008000u;
    constexpr std::uint32_t kExpM = 0x0E000E00u;
    const std::uint32_t v0 = ((q << 12) & kSign) | ((q << 9) & kExpM);
    const std::uint32_t v1 = ((q << 8) & kSign) | ((q << 5) & kExpM);
    const std::uint32_t v2 = ((q << 4) & kSign) | ((q << 1) & kExpM);
    const std::uint32_t v3 = (q & kSign) | ((q >> 3) & kExpM);
    out[0] = __hmul2(*reinterpret_cast<const half2*>(&v0), rebias);
    out[1] = __hmul2(*reinterpret_cast<const half2*>(&v1), rebias);
    out[2] = __hmul2(*reinterpret_cast<const half2*>(&v2), rebias);
    out[3] = __hmul2(*reinterpret_cast<const half2*>(&v3), rebias);
}

__device__ __forceinline__ void qpn_stage_pairs(const __half* seq, half2* out) {
    const auto* r = reinterpret_cast<const std::uint32_t*>(seq);
    const std::uint32_t o0 = __byte_perm(r[0], r[2], 0x5410);
    const std::uint32_t o1 = __byte_perm(r[0], r[2], 0x7632);
    const std::uint32_t o2 = __byte_perm(r[1], r[3], 0x5410);
    const std::uint32_t o3 = __byte_perm(r[1], r[3], 0x7632);
    out[0] = *reinterpret_cast<const half2*>(&o0);
    out[1] = *reinterpret_cast<const half2*>(&o1);
    out[2] = *reinterpret_cast<const half2*>(&o2);
    out[3] = *reinterpret_cast<const half2*>(&o3);
}

__device__ __forceinline__ void qpn_mma(float (&c)[4], unsigned a0, unsigned a1, unsigned b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    (void)c; (void)a0; (void)a1; (void)b0;
#endif
}

__device__ __forceinline__ float qpn_e4m3_scale(std::uint8_t value) {
    const std::uint16_t bits = static_cast<std::uint16_t>((value & 0x80U) << 8) |
                               static_cast<std::uint16_t>((value & 0x7FU) << 7);
    return __half2float(__ushort_as_half(bits)) * 256.0F;
}

__device__ __forceinline__ std::int64_t qpn_scale_offset(int row, int group, int k) {
    const int tile = row / 128;
    const int inner = row - tile * 128;
    return (static_cast<std::int64_t>(tile) * (k / 64) + group / 4) * 512 +
           (inner & 31) * 16 + (inner >> 5) * 4 + (group & 3);
}

__global__ void qpn_pack_activation(const __nv_bfloat16* __restrict__ input,
                                    __half* __restrict__ output, std::int64_t count) {
    const std::int64_t i = std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { output[i] = __float2half_rn(__bfloat162float(input[i])); }
}

struct QpnFp32Output {
    float* data;
    int rows;
    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[std::int64_t(token) * rows + row] = value;
    }
};

struct QpnBf16Output {
    __nv_bfloat16* data;
    int rows;
    __device__ __forceinline__ void store(int row, int token, float value) const {
        data[std::int64_t(token) * rows + row] = __float2bfloat16_rn(value);
    }
};

struct QpnAddOutput {
    __nv_bfloat16* data;
    int rows;
    __device__ __forceinline__ void store(int row, int token, float value) const {
        __nv_bfloat16* slot = data + std::int64_t(token) * rows + row;
        *slot = __float2bfloat16_rn(__bfloat162float(*slot) + value);
    }
};

template <int Tiles, class Output>
__global__ void __launch_bounds__(kQpnThreads)
    qpn_kernel(const std::uint8_t* __restrict__ codes, const std::uint8_t* __restrict__ scales,
               const __half* __restrict__ x, int weight_row0, int output_rows, int k, int tokens,
               float inverse_divisor, Output output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    __shared__ float partial[kQpnKSplits][Tiles][32][8];
    const int tid = int(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_group = warp / kQpnKSplits;
    const int ksplit = warp % kQpnKSplits;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int row0 = weight_row0 + int(blockIdx.x) * 32 + row_group * 16 + group;
    const int row1 = row0 + 8;
    const int token0 = int(blockIdx.y) * Tiles * 8;
    float acc[Tiles][4];
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[tile][i] = 0.0F; }
    }
    const int groups = k / 16;
    for (int g = ksplit; g < groups; g += kQpnKSplits) {
        const float scale0 = qpn_e4m3_scale(scales[qpn_scale_offset(row0, g, k)]) * inverse_divisor;
        const float scale1 = qpn_e4m3_scale(scales[qpn_scale_offset(row1, g, k)]) * inverse_divisor;
        const std::uint8_t* code0 = codes + std::int64_t(row0) * (k / 2) + g * 8;
        const std::uint8_t* code1 = codes + std::int64_t(row1) * (k / 2) + g * 8;
        const half2 rebias = __float2half2_rn(16384.0F);
        const half2 scale0_h = __half2half2(__float2half_rn(scale0));
        const half2 scale1_h = __half2half2(__float2half_rn(scale1));
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            half2 weights0[4], weights1[4];
            qpn_decode_e2m1(*reinterpret_cast<const std::uint32_t*>(code0 + half * 4), rebias,
                            weights0);
            qpn_decode_e2m1(*reinterpret_cast<const std::uint32_t*>(code1 + half * 4), rebias,
                            weights1);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                weights0[j] = __hmul2(weights0[j], scale0_h);
                weights1[j] = __hmul2(weights1[j], scale1_h);
            }
            const unsigned a0 = reinterpret_cast<const unsigned*>(weights0)[tig];
            const unsigned a1 = reinterpret_cast<const unsigned*>(weights1)[tig];
#pragma unroll
            for (int tile = 0; tile < Tiles; ++tile) {
                const int token = token0 + tile * 8 + group;
                unsigned b0 = 0;
                if (token < tokens) {
                    const __half* source = x + std::int64_t(token) * k + g * 16 + half * 8;
                    half2 staged[4];
                    qpn_stage_pairs(source, staged);
                    b0 = reinterpret_cast<const unsigned*>(staged)[tig];
                }
                qpn_mma(acc[tile], a0, a1, b0);
            }
        }
    }
#pragma unroll
    for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = (i >> 1) * 8 + group;
            const int c = tig * 2 + (i & 1);
            partial[ksplit][tile][row_group * 16 + r][c] = acc[tile][i];
        }
    }
    __syncthreads();
    for (int i = tid; i < Tiles * 32 * 8; i += kQpnThreads) {
        const int tile = i / (32 * 8);
        const int rem = i % (32 * 8);
        const int r = rem / 8;
        const int c = rem % 8;
        const int token = token0 + tile * 8 + c;
        if (token >= tokens) { continue; }
        float value = 0.0F;
#pragma unroll
        for (int split = 0; split < kQpnKSplits; ++split) {
            value += partial[split][tile][r][c];
        }
        output.store(int(blockIdx.x) * 32 + r, token, value);
    }
#else
    (void)codes; (void)scales; (void)x; (void)weight_row0; (void)output_rows; (void)k; (void)tokens;
    (void)inverse_divisor; (void)output;
#endif
}

template <class Output>
void qpn_launch(const __half* x, const Weight& weight, int weight_row0, int output_rows, int k,
                int tokens, Output output, cudaStream_t stream) {
    const auto launch = [&]<int Tiles>() {
        const int token_tiles = (tokens + Tiles * 8 - 1) / (Tiles * 8);
        const dim3 grid(output_rows / 32, token_tiles, 1);
        qpn_kernel<Tiles><<<grid, kQpnThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), x, weight_row0, output_rows, k,
            tokens, 1.0F / weight.weight_scale_divisor, output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (tokens <= 8) { launch.template operator()<1>(); }
    else if (tokens <= 16) { launch.template operator()<2>(); }
    else if (tokens <= 32) { launch.template operator()<4>(); }
    else if (tokens <= 40) { launch.template operator()<5>(); }
    else { launch.template operator()<8>(); }
}

template <int Tiles>
__global__ void __launch_bounds__(kQpnThreads)
    qpn_paired_kernel(const std::uint8_t* __restrict__ gate_codes,
                      const std::uint8_t* __restrict__ gate_scales,
                      const std::uint8_t* __restrict__ up_codes,
                      const std::uint8_t* __restrict__ up_scales, const __half* __restrict__ x,
                      int output_rows, int k, int tokens, float gate_inverse, float up_inverse,
                      int up_row_offset, QpnBf16Output output) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    __shared__ float partial[kQpnKSplits][Tiles][32][8];
    __shared__ float gate_values[Tiles][32][8];
    const int tid = int(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int row_group = warp / kQpnKSplits;
    const int ksplit = warp % kQpnKSplits;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int token0 = int(blockIdx.y) * Tiles * 8;
    for (int projection = 0; projection < 2; ++projection) {
        const auto* codes = projection == 0 ? gate_codes : up_codes;
        const auto* scales = projection == 0 ? gate_scales : up_scales;
        const float inverse = projection == 0 ? gate_inverse : up_inverse;
        const int row0 = (projection == 0 ? 0 : up_row_offset) +
                         int(blockIdx.x) * 32 + row_group * 16 + group;
        const int row1 = row0 + 8;
        float acc[Tiles][4];
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[tile][i] = 0.0F; }
        }
        const int groups = k / 16;
        for (int g = ksplit; g < groups; g += kQpnKSplits) {
            const float scale0 = qpn_e4m3_scale(scales[qpn_scale_offset(row0, g, k)]) * inverse;
            const float scale1 = qpn_e4m3_scale(scales[qpn_scale_offset(row1, g, k)]) * inverse;
            const std::uint8_t* code0 = codes + std::int64_t(row0) * (k / 2) + g * 8;
            const std::uint8_t* code1 = codes + std::int64_t(row1) * (k / 2) + g * 8;
            const half2 rebias = __float2half2_rn(16384.0F);
            const half2 scale0_h = __half2half2(__float2half_rn(scale0));
            const half2 scale1_h = __half2half2(__float2half_rn(scale1));
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                half2 weights0[4], weights1[4];
                qpn_decode_e2m1(*reinterpret_cast<const std::uint32_t*>(code0 + half * 4), rebias,
                                weights0);
                qpn_decode_e2m1(*reinterpret_cast<const std::uint32_t*>(code1 + half * 4), rebias,
                                weights1);
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    weights0[j] = __hmul2(weights0[j], scale0_h);
                    weights1[j] = __hmul2(weights1[j], scale1_h);
                }
                const unsigned a0 = reinterpret_cast<const unsigned*>(weights0)[tig];
                const unsigned a1 = reinterpret_cast<const unsigned*>(weights1)[tig];
#pragma unroll
                for (int tile = 0; tile < Tiles; ++tile) {
                    const int token = token0 + tile * 8 + group;
                    unsigned b0 = 0;
                    if (token < tokens) {
                        const __half* source = x + std::int64_t(token) * k + g * 16 + half * 8;
                        half2 staged[4];
                        qpn_stage_pairs(source, staged);
                        b0 = reinterpret_cast<const unsigned*>(staged)[tig];
                    }
                    qpn_mma(acc[tile], a0, a1, b0);
                }
            }
        }
#pragma unroll
        for (int tile = 0; tile < Tiles; ++tile) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r = (i >> 1) * 8 + group;
                const int c = tig * 2 + (i & 1);
                partial[ksplit][tile][row_group * 16 + r][c] = acc[tile][i];
            }
        }
        __syncthreads();
        for (int i = tid; i < Tiles * 32 * 8; i += kQpnThreads) {
            const int tile = i / (32 * 8);
            const int rem = i % (32 * 8);
            const int r = rem / 8;
            const int c = rem % 8;
            const int token = token0 + tile * 8 + c;
            if (token >= tokens) { continue; }
            float value = 0.0F;
#pragma unroll
            for (int split = 0; split < kQpnKSplits; ++split) {
                value += partial[split][tile][r][c];
            }
            if (projection == 0) {
                gate_values[tile][r][c] = value;
            } else {
                output.store(int(blockIdx.x) * 32 + r, token,
                             silu(gate_values[tile][r][c]) * value);
            }
        }
        __syncthreads();
    }
#else
    (void)gate_codes; (void)gate_scales; (void)up_codes; (void)up_scales; (void)x;
    (void)output_rows; (void)k; (void)tokens; (void)gate_inverse; (void)up_inverse; (void)output;
#endif
}

template <class Output>
void qpn_paired_launch(const __half* x, const Weight& gate, const Weight& up, int tokens,
                       int up_row_offset, Output output, cudaStream_t stream) {
    const auto launch = [&]<int Tiles>() {
        const int token_tiles = (tokens + Tiles * 8 - 1) / (Tiles * 8);
        const int output_rows = up_row_offset > 0 ? up_row_offset : gate.n;
        const dim3 grid(output_rows / 32, token_tiles, 1);
        qpn_paired_kernel<Tiles><<<grid, kQpnThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(gate.qdata),
            static_cast<const std::uint8_t*>(gate.scales),
            static_cast<const std::uint8_t*>(up.qdata),
            static_cast<const std::uint8_t*>(up.scales), x, gate.n, gate.k, tokens,
            1.0F / gate.weight_scale_divisor, 1.0F / up.weight_scale_divisor, up_row_offset,
            output);
        CUDA_CHECK(cudaGetLastError());
    };
    if (tokens <= 8) { launch.template operator()<1>(); }
    else if (tokens <= 16) { launch.template operator()<2>(); }
    else if (tokens <= 32) { launch.template operator()<4>(); }
    else if (tokens <= 40) { launch.template operator()<5>(); }
    else { launch.template operator()<8>(); }
}

std::size_t round256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

void qpn_prepare_activation(const Tensor& x, __half* out, cudaStream_t stream) {
    const std::int64_t count = std::int64_t(x.ne[0]) * x.ne[1];
    qpn_pack_activation<<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), out, count);
    CUDA_CHECK(cudaGetLastError());
}

bool qpn_tokens(int tokens) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_NVFP4_QPN");
        return value != nullptr && std::atoi(value) == 1;
    }();
    return enabled && current_device_tier() == DeviceTier::Turing && tokens >= 1 && tokens <= 64;
}

bool qpn_shape(const Weight& weight, int tokens) {
    if (!qpn_tokens(tokens)) { return false; }
    return weight.qtype == QType::NVFP4 && weight.layout == QuantLayout::BlockScaleK16M128x4 &&
           weight.group == 16 && weight.qdata != nullptr && weight.scales != nullptr &&
           (weight.n == 34816 || weight.n == 17408 || weight.n == 8704 || weight.n == 5120) &&
           (weight.k == 5120 || weight.k == 6144 || weight.k == 4352 || weight.k == 17408) &&
           weight.weight_scale_divisor > 0.0F && std::isfinite(weight.weight_scale_divisor);
}

} // namespace

bool nvfp4_turing_qpn_enabled_tokens(std::int32_t tokens) {
    return qpn_tokens(tokens);
}

bool nvfp4_turing_qpn_supported(const Weight& weight, std::int32_t tokens) {
    return qpn_shape(weight, tokens);
}

std::size_t nvfp4_turing_qpn_activation_bytes(std::int32_t k, std::int32_t tokens) {
    return round256(std::size_t(k) * tokens * sizeof(__half));
}

std::size_t nvfp4_turing_qpn_swiglu_workspace_bytes(std::int32_t k, std::int32_t tokens) {
    return nvfp4_turing_qpn_activation_bytes(k, tokens);
}

void nvfp4_turing_qpn_linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out,
                                    WorkspaceArena& workspace, cudaStream_t stream) {
    if (!qpn_shape(weight, x.ne[1]) || (weight.n != 34816 && weight.n != 8704) ||
        weight.k != 5120) {
        throw std::invalid_argument("SM75 QPN SwiGLU: unsupported geometry");
    }
    const int half = weight.n / 2;
    auto scope = workspace.scope();
    auto* activation = static_cast<__half*>(
        workspace.alloc_bytes(nvfp4_turing_qpn_activation_bytes(weight.k, x.ne[1])).data);
    qpn_prepare_activation(x, activation, stream);
    qpn_paired_launch(activation, weight, weight, x.ne[1], half,
                      QpnBf16Output{static_cast<__nv_bfloat16*>(out.data), half}, stream);
}

void nvfp4_turing_qpn_linear_swiglu_pair(const Tensor& x, const Weight& gate, const Weight& up,
                                         Tensor& out, WorkspaceArena& workspace,
                                         cudaStream_t stream) {
    if (!qpn_shape(gate, x.ne[1]) || !qpn_shape(up, x.ne[1]) || gate.n != up.n || gate.k != up.k) {
        throw std::invalid_argument("SM75 QPN paired SwiGLU: unsupported geometry");
    }
    auto scope = workspace.scope();
    auto* activation = static_cast<__half*>(
        workspace.alloc_bytes(nvfp4_turing_qpn_activation_bytes(gate.k, x.ne[1])).data);
    qpn_prepare_activation(x, activation, stream);
    qpn_paired_launch(activation, gate, up, x.ne[1], 0,
                      QpnBf16Output{static_cast<__nv_bfloat16*>(out.data), gate.n}, stream);
}

void nvfp4_turing_qpn_linear_add(const Tensor& x, const Weight& weight, Tensor& residual,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    if (!qpn_shape(weight, x.ne[1]) || weight.n != 5120 ||
        (weight.k != 6144 && weight.k != 4352 && weight.k != 17408)) {
        throw std::invalid_argument("SM75 QPN LinearAdd: unsupported geometry");
    }
    auto scope = workspace.scope();
    auto* activation = static_cast<__half*>(
        workspace.alloc_bytes(nvfp4_turing_qpn_activation_bytes(weight.k, x.ne[1])).data);
    qpn_prepare_activation(x, activation, stream);
    qpn_launch(activation, weight, 0, weight.n, weight.k, x.ne[1],
               QpnAddOutput{static_cast<__nv_bfloat16*>(residual.data), weight.n}, stream);
}

void nvfp4_turing_qpn_linear(const Tensor& x, const Weight& weight, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream) {
    if (!qpn_shape(weight, x.ne[1])) {
        throw std::invalid_argument("SM75 QPN Linear: unsupported geometry");
    }
    auto scope = workspace.scope();
    auto* activation = static_cast<__half*>(
        workspace.alloc_bytes(nvfp4_turing_qpn_activation_bytes(weight.k, x.ne[1])).data);
    qpn_prepare_activation(x, activation, stream);
    if (out.dtype == DType::FP32) {
        qpn_launch(activation, weight, 0, weight.n, weight.k, x.ne[1],
                   QpnFp32Output{static_cast<float*>(out.data), weight.n}, stream);
    } else {
        qpn_launch(activation, weight, 0, weight.n, weight.k, x.ne[1],
                   QpnBf16Output{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
    }
}

} // namespace ninfer::ops::detail
