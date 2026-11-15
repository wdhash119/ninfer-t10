#include "ops/linear/part_projection.h"

#include "core/device_launch.h"
#include "ops/common/device_tier.h"
#include "ops/linear_swiglu/q4cublas/cublas_prefill_gate.h"
#include "ops/linear_swiglu/q4cublas/w4_cublas_prefill.h"
#include "core/layout.h"
#include "core/qtype.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/gguf/gguf_linear.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace ninfer::ops::detail {
namespace {

bool linear_part(QType format) { return !is_gguf(format) && linear_writes_fp32(format); }

// Whether a part's rows of its output are one contiguous block: one column, or the whole output.
bool contiguous_rows(const Tensor& out, const ProjectionPart& part) {
    return out.ne[1] == 1 || (part.row == 0 && part.weight.n == out.ne[0]);
}

} // namespace

bool gguf_parts(const PartProjectionWeights& weights) {
    if (weights.parts.empty()) { throw std::invalid_argument("a part projection needs parts"); }
    const bool gguf = is_gguf(weights.parts.front().weight.qtype);
    for (const auto& part : weights.parts) {
        if (is_gguf(part.weight.qtype) != gguf || (!gguf && !linear_part(part.weight.qtype))) {
            throw std::invalid_argument("a part projection takes all GGUF parts, or all BF16, "
                                        "row-split, FP8 and NVFP4 parts");
        }
    }
    return gguf;
}

void project_parts(const Tensor& x, const PartProjectionWeights& weights,
                   std::span<Tensor* const> outputs, WorkspaceArena& workspace,
                   cudaStream_t stream, const CublasPrefillFamily* cublas_family) {
    if (gguf_parts(weights)) {
        std::vector<GgufProduct> products;
        products.reserve(weights.parts.size());
        for (const auto& part : weights.parts) {
            GgufProduct product;
            product.weight = &part.weight;
            product.out    = outputs[static_cast<std::size_t>(part.output)];
            product.row    = part.row;
            products.push_back(product);
        }
        gguf_project(x, products, workspace, stream);
        return;
    }
    const std::int32_t columns = x.ne[1];
    if (cublas_family != nullptr && current_device_tier() == DeviceTier::Turing &&
        allows_cublas_prefill(weights.policy) && cublas_prefill_takes(*cublas_family, columns)) {
        std::vector<CublasProjection> parents;
        std::vector<CublasProjectionDestination> destinations;
        parents.reserve(weights.parts.size());
        destinations.reserve(weights.parts.size());
        for (const auto& part : weights.parts) {
            Tensor& out = *outputs[static_cast<std::size_t>(part.output)];
            destinations.push_back({out.data, 0, part.weight.n, out.ne[0], part.row});
            parents.push_back({&part.weight, &destinations.back(), 1});
        }
        if (w4_cublas_projection_supported(parents.data(), static_cast<int>(parents.size()),
                                           columns)) {
            w4_cublas_projection_launch(x, parents.data(), static_cast<int>(parents.size()),
                                        workspace, stream);
            return;
        }
    }
    for (const auto& part : weights.parts) {
        Tensor& out = *outputs[static_cast<std::size_t>(part.output)];
        if (contiguous_rows(out, part)) {
            Tensor rows(static_cast<std::byte*>(out.data) + std::size_t(part.row) * sizeof(std::uint16_t),
                        DType::BF16, {part.weight.n, columns});
            linear(x, part.weight, rows, weights.policy, workspace, stream);
            continue;
        }
        auto scope   = workspace.scope();
        Tensor block = workspace.alloc(DType::BF16, {part.weight.n, columns});
        linear(x, part.weight, block, weights.policy, workspace, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(
            static_cast<std::byte*>(out.data) + std::size_t(part.row) * sizeof(std::uint16_t),
            std::size_t(out.ne[0]) * sizeof(std::uint16_t), block.data,
            std::size_t(part.weight.n) * sizeof(std::uint16_t),
            std::size_t(part.weight.n) * sizeof(std::uint16_t), std::size_t(columns),
            cudaMemcpyDeviceToDevice, stream));
    }
}

std::size_t project_parts_workspace_bytes(const PartProjectionWeights& weights,
                                          std::int32_t min_tokens, std::int32_t max_tokens,
                                          const CublasPrefillFamily* cublas_family) {
    if (gguf_parts(weights)) {
        std::vector<GgufShape> shapes;
        for (const auto& part : weights.parts) {
            shapes.push_back({part.weight.qtype, part.weight.n, part.weight.k});
        }
        return gguf_project_workspace_bytes(shapes, min_tokens, max_tokens);
    }
    std::size_t widest = 0;
    for (const auto& part : weights.parts) {
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::BF16, {part.weight.n, max_tokens});
        (void)layout.alloc_bytes(linear_workspace_capacity_bytes(
            part.weight.qtype, part.weight.n, part.weight.k, weights.policy, min_tokens,
            max_tokens));
        widest = std::max(widest, layout.peak_bytes(1));
    }
    if (cublas_family != nullptr && current_device_tier() == DeviceTier::Turing &&
        allows_cublas_prefill(weights.policy)) {
        const std::int32_t first =
            cublas_prefill_first_width(*cublas_family, min_tokens, max_tokens);
        if (first != 0) {
            std::int32_t max_rows = 0;
            for (const auto& part : weights.parts) { max_rows = std::max(max_rows, part.weight.n); }
            widest = std::max(widest, w4_cublas_projection_workspace_capacity_bytes(
                                          max_rows, weights.parts.front().weight.k, first,
                                          max_tokens));
        }
    }
    return widest;
}

} // namespace ninfer::ops::detail
