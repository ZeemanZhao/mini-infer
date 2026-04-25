#include "mini_infer/ops.h"

#include <stdexcept>

namespace mini_infer {
namespace ops {

Tensor matmul(const Tensor& a, const Tensor& b) {
  if (a.ndim() != 2 || b.ndim() != 2)
    throw std::invalid_argument("matmul: both inputs must be 2-D");
  int M = a.shape()[0], K = a.shape()[1];
  if (b.shape()[0] != K)
    throw std::invalid_argument("matmul: inner dimensions must match");
  int N = b.shape()[1];

  Tensor c({M, N});
  const float* A = a.data();
  const float* B = b.data();
  float* C = c.data();

  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) acc += A[i * K + k] * B[k * N + j];
      C[i * N + j] = acc;
    }
  }
  return c;
}

Tensor bias_add(const Tensor& x, const Tensor& bias) {
  if (x.ndim() != 2 || bias.ndim() != 1)
    throw std::invalid_argument("bias_add: x must be 2-D, bias must be 1-D");
  if (x.shape()[1] != bias.shape()[0])
    throw std::invalid_argument("bias_add: bias length must equal x columns");
  int M = x.shape()[0], N = x.shape()[1];

  Tensor out = x.clone();
  float* O = out.data();
  const float* B = bias.data();

  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) O[i * N + j] += B[j];

  return out;
}

}  // namespace ops
}  // namespace mini_infer
