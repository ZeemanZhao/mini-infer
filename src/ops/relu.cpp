#include "mini_infer/ops.h"

namespace mini_infer {
namespace ops {

Tensor relu(const Tensor& x) {
  Tensor out = x.clone();
  float* d = out.data();
  for (int i = 0; i < out.size(); ++i)
    if (d[i] < 0.0f) d[i] = 0.0f;
  return out;
}

}  // namespace ops
}  // namespace mini_infer
