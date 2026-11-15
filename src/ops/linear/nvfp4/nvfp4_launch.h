#pragma once
#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
using Nvfp4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// Experimental SM75 QPN decode/verify route, opt-in with NINFER_NVFP4_QPN=1.
[[nodiscard]] bool nvfp4_turing_qpn_enabled_tokens(std::int32_t tokens);
[[nodiscard]] bool nvfp4_turing_qpn_supported(const Weight& weight, std::int32_t tokens);
[[nodiscard]] std::size_t nvfp4_turing_qpn_activation_bytes(std::int32_t k, std::int32_t tokens);
[[nodiscard]] std::size_t nvfp4_turing_qpn_swiglu_workspace_bytes(std::int32_t k,
                                                                  std::int32_t tokens);
void nvfp4_turing_qpn_linear_swiglu(const Tensor& x, const Weight& weight, Tensor& out,
                                    WorkspaceArena& workspace, cudaStream_t stream);
void nvfp4_turing_qpn_linear_swiglu_pair(const Tensor& x, const Weight& gate, const Weight& up,
                                         Tensor& out, WorkspaceArena& workspace,
                                         cudaStream_t stream);
void nvfp4_turing_qpn_linear_add(const Tensor& x, const Weight& weight, Tensor& residual,
                                 WorkspaceArena& workspace, cudaStream_t stream);
void nvfp4_turing_qpn_linear(const Tensor& x, const Weight& weight, Tensor& out,
                             WorkspaceArena& workspace, cudaStream_t stream);
}
