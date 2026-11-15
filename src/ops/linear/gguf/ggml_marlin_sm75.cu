// Experimental dense SM75 Marlin-like route for GGUF Q4_K/Q5_K. Weights are decoded directly from
// their packed blocks into m16n8k8 fragments; activations are FP16 with a per-column power-of-two
// range factor and products accumulate in FP32. No GGUF repacking or whole-weight dequantization.

#include "ggml_bridge_internal.cuh"
#include "ggml_bridge_vec.cuh"

#include <algorithm>
#include <cstdlib>

namespace ninfer::ops::gguf {
namespace {

inline constexpr int kMarlinStep = 128;
inline constexpr int kMarlinRows = 64;
inline constexpr int kMarlinWarps = 4;
inline constexpr int kMarlinThreads = 32 * kMarlinWarps;
inline constexpr int kMarlinJ = 128;
inline constexpr int kMarlinStride = kMarlinStep + 8;
inline constexpr int kActivationThreads = 256;

__device__ __forceinline__ unsigned marlin_half2_word(float lo, float hi) {
    const __half2 pair = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&pair);
}

__device__ __forceinline__ unsigned marlin_load_half2(const __half* p) {
    const __half2 pair = *reinterpret_cast<const __half2*>(p);
    return *reinterpret_cast<const unsigned*>(&pair);
}

__device__ __forceinline__ void marlin_mma1688(float (&c)[4], unsigned a0, unsigned a1,
                                                unsigned b0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
#else
    (void)c; (void)a0; (void)a1; (void)b0;
#endif
}

__global__ void __launch_bounds__(kActivationThreads)
    marlin_activation_kernel(const __nv_bfloat16* __restrict__ x, int k,
                             const std::int32_t* __restrict__ input_columns,
                             __half* __restrict__ plane, float* __restrict__ factors) {
    const int c = int(blockIdx.x);
    const __nv_bfloat16* column = x + std::int64_t(c) * k;
    float largest = 0.0F;
    for (int i = int(threadIdx.x); i < k; i += kActivationThreads) {
        const int from = input_columns != nullptr ? input_columns[i] : i;
        largest = fmaxf(largest, fabsf(__bfloat162float(column[from])));
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
        largest = fmaxf(largest, __shfl_down_sync(0xffffffffu, largest, offset));
    }
    __shared__ float warp_max[kActivationThreads / 32];
    if ((threadIdx.x & 31) == 0) { warp_max[threadIdx.x / 32] = largest; }
    __syncthreads();
    largest = warp_max[0];
#pragma unroll
    for (int w = 1; w < kActivationThreads / 32; ++w) {
        largest = fmaxf(largest, warp_max[w]);
    }
    const int e = isfinite(largest) && largest >= 32768.0F ? ilogbf(largest) - 14 : 0;
    const float down = ldexpf(1.0F, -e);
    __half* out = plane + std::int64_t(c) * k;
    for (int i = int(threadIdx.x); i < k; i += kActivationThreads) {
        const int from = input_columns != nullptr ? input_columns[i] : i;
        out[i] = __float2half_rn(__bfloat162float(column[from]) * down);
    }
    if (threadIdx.x == 0) { factors[c] = ldexpf(1.0F, e); }
}

template <ggml_type type, int J>
__global__ void __launch_bounds__(kMarlinThreads)
    marlin_kernel(const std::uint8_t* __restrict__ weight, std::int64_t row_bytes, int rows, int k,
                  const __half* __restrict__ activation, const float* __restrict__ factors,
                  int columns, float* __restrict__ out, std::int64_t out_stride) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    using D = detail::vec::Decoder<type>;
    constexpr int kPerBlock = D::kBlockElems / 32;
    __shared__ __align__(16) __half stage[J * kMarlinStride];
    const int tid = int(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int group = lane >> 2;
    const int tig = lane & 3;
    const int row0 = int(blockIdx.x) * kMarlinRows + warp * 16 + group;
    const bool active0 = row0 < rows;
    const bool active1 = row0 + 8 < rows;
    const std::uint8_t* w0 = weight + std::int64_t(active0 ? row0 : 0) * row_bytes;
    const std::uint8_t* w1 = weight + std::int64_t(active1 ? row0 + 8 : 0) * row_bytes;
    float acc[J / 8][4];
#pragma unroll
    for (int g = 0; g < J / 8; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[g][i] = 0.0F; }
    }
    const int column0 = int(blockIdx.y) * J;
    for (int step = 0; step * kMarlinStep < k; ++step) {
        __syncthreads();
        for (int i = tid; i < J * (kMarlinStep / 8); i += kMarlinThreads) {
            const int c = i / (kMarlinStep / 8);
            const int q = i % (kMarlinStep / 8);
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (column0 + c < columns) {
                v = *reinterpret_cast<const uint4*>(activation +
                    std::int64_t(column0 + c) * k + step * kMarlinStep + q * 8);
            }
            *reinterpret_cast<uint4*>(stage + c * kMarlinStride + q * 8) = v;
        }
        __syncthreads();
        const int end = min(k / 32, (step + 1) * (kMarlinStep / 32));
        for (int s = step * (kMarlinStep / 32); s < end; ++s) {
            detail::vec::Slice d0, d1;
            const std::int64_t block = std::int64_t(s / kPerBlock) * D::kBlockBytes;
            D::decode(w0 + block, s % kPerBlock, nullptr, d0);
            D::decode(w1 + block, s % kPerBlock, nullptr, d1);
            const int local = s - step * (kMarlinStep / 32);
#pragma unroll
            for (int quarter = 0; quarter < 4; ++quarter) {
                const int base = quarter * 8 + 2 * tig;
                const unsigned a0 = marlin_half2_word(
                    detail::vec::slice_value<type>(d0, (base + 0) / 4, (base + 0) % 4),
                    detail::vec::slice_value<type>(d0, (base + 1) / 4, (base + 1) % 4));
                const unsigned a1 = marlin_half2_word(
                    detail::vec::slice_value<type>(d1, (base + 0) / 4, (base + 0) % 4),
                    detail::vec::slice_value<type>(d1, (base + 1) / 4, (base + 1) % 4));
#pragma unroll
                for (int g = 0; g < J / 8; ++g) {
                    const __half* b = stage + (g * 8 + group) * kMarlinStride +
                                      local * 32 + quarter * 8 + 2 * tig;
                    marlin_mma1688(acc[g], a0, a1, marlin_load_half2(b));
                }
            }
        }
    }
#pragma unroll
    for (int g = 0; g < J / 8; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int r = row0 + (i >> 1) * 8;
            const int c = column0 + g * 8 + tig * 2 + (i & 1);
            if (r < rows && c < columns) {
                out[std::int64_t(c) * out_stride + r] = acc[g][i] * factors[c];
            }
        }
    }
#else
    (void)weight; (void)row_bytes; (void)rows; (void)k; (void)activation; (void)factors;
    (void)columns; (void)out; (void)out_stride;
#endif
}

template <ggml_type type>
void marlin_launch(const void* weight, std::int64_t row_bytes, int rows, int k,
                   const void* activation, int columns, float* out, std::int64_t out_stride,
                   cudaStream_t stream) {
    const auto* plane = static_cast<const __half*>(activation);
    const float* factors = reinterpret_cast<const float*>(plane + std::int64_t(columns) * k);
    const dim3 grid((rows + kMarlinRows - 1) / kMarlinRows,
                    (columns + kMarlinJ - 1) / kMarlinJ, 1);
    marlin_kernel<type, kMarlinJ><<<grid, kMarlinThreads, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight), row_bytes, rows, k, plane, factors, columns, out,
        out_stride);
    detail::check(cudaGetLastError(), "SM75 Marlin-like product launch");
}

} // namespace

bool marlin_sm75_supported(GgmlType type, int rows, int k, int columns) {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_GGUF_MARLIN");
        return value != nullptr && std::atoi(value) == 1;
    }();
    // A kernel unit holds GgmlType opaque (ggml_bridge_args.h); its values are the ggml_type ids
    // a GGUF tensor directory carries, so the type test goes through those ids.
    const int type_id = static_cast<int>(type);
    return enabled && detail::device_facts().cc == 750 &&
           (type_id == GGML_TYPE_Q4_K || type_id == GGML_TYPE_Q5_K) && rows > 0 && k > 0 &&
           k % 256 == 0 && columns >= 128;
}

std::size_t marlin_sm75_activation_bytes(int k, int columns) {
    const std::size_t plane = (std::size_t(k) * columns * sizeof(__half) + 255) / 256 * 256;
    return plane + std::size_t(columns) * sizeof(float);
}

void marlin_sm75_activation(const __nv_bfloat16* x, int k, int columns,
                            const std::int32_t* input_columns, void* out, cudaStream_t stream) {
    auto* plane = static_cast<__half*>(out);
    auto* factors = reinterpret_cast<float*>(plane + std::int64_t(columns) * k);
    marlin_activation_kernel<<<columns, kActivationThreads, 0, stream>>>(
        x, k, input_columns, plane, factors);
    detail::check(cudaGetLastError(), "SM75 Marlin-like activation launch");
}

void marlin_sm75_product(GgmlType type, const void* weight, std::int64_t row_bytes, int rows,
                         int k, const void* activation, int columns, float* out,
                         std::int64_t out_stride, cudaStream_t stream) {
    if (!marlin_sm75_supported(type, rows, k, columns)) {
        throw std::invalid_argument("SM75 Marlin-like product: unsupported geometry or type");
    }
    switch (static_cast<int>(type)) {
    case GGML_TYPE_Q4_K:
        marlin_launch<GGML_TYPE_Q4_K>(weight, row_bytes, rows, k, activation, columns, out,
                                      out_stride, stream);
        return;
    case GGML_TYPE_Q5_K:
        marlin_launch<GGML_TYPE_Q5_K>(weight, row_bytes, rows, k, activation, columns, out,
                                      out_stride, stream);
        return;
    default:
        throw std::invalid_argument("SM75 Marlin-like product: unsupported GGUF type");
    }
}

} // namespace ninfer::ops::gguf
