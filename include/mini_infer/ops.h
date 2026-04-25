#pragma once

#include "mini_infer/tensor.h"

namespace mini_infer {
namespace ops {

// C = A @ B.  A: [M, K],  B: [K, N]  →  C: [M, N]
Tensor matmul(const Tensor& a, const Tensor& b);

// out = x + bias.  x: [M, N],  bias: [N]  →  out: [M, N]
Tensor bias_add(const Tensor& x, const Tensor& bias);

// out = max(0, x)  (element-wise)
Tensor relu(const Tensor& x);

}  // namespace ops
}  // namespace mini_infer
