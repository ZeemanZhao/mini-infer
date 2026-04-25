#include "mini_infer/tensor.h"

#include <algorithm>
#include <stdexcept>

namespace mini_infer {

Tensor::Tensor(std::vector<int> shape) : shape_(std::move(shape)) {
  size_ = 1;
  for (int d : shape_) {
    if (d <= 0) throw std::invalid_argument("all tensor dimensions must be positive");
    size_ *= d;
  }
  // make_unique<float[]> value-initializes → zero-fills the buffer
  data_ = std::make_unique<float[]>(size_);
}

Tensor Tensor::clone() const {
  Tensor t(shape_);
  std::copy(data_.get(), data_.get() + size_, t.data_.get());
  return t;
}

}  // namespace mini_infer
