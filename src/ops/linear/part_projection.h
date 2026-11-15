#pragma once

// ninfer::ops::detail - the projection of a PartProjectionWeights (ninfer/ops/weight_input.h) into
// its outputs, shared by the GDN and attention input projections.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/weight_input.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// Whether every part is a GGUF matrix (projected over one q8_1 activation by gguf_project), or
// every part a format Linear projects at any shape: BF16 or row-split Q4/Q5/Q6/Q8. Throws for a
// mix or another format.
[[nodiscard]] bool gguf_parts(const PartProjectionWeights& weights);

// Rows [row, row + n) of outputs[part.output] (each a BF16 [rows, T] matrix) take part's product.
struct CublasPrefillFamily;

// `cublas_family` is an opt-in SM75 path for Qwen TP shares. Null preserves the original per-part
// linear route and all non-Turing behavior.
void project_parts(const Tensor& x, const PartProjectionWeights& weights,
                   std::span<Tensor* const> outputs, WorkspaceArena& workspace,
                   cudaStream_t stream, const CublasPrefillFamily* cublas_family = nullptr);

[[nodiscard]] std::size_t project_parts_workspace_bytes(const PartProjectionWeights& weights,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens,
                                                        const CublasPrefillFamily* cublas_family = nullptr);

} // namespace ninfer::ops::detail
