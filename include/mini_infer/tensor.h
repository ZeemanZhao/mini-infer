#pragma once

#include <memory>
#include <vector>

namespace mini_infer {

class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<int> shape);

  // Explicit move; copy disabled — single owner via unique_ptr
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;

  ~Tensor() = default;

  const std::vector<int>& shape() const { return shape_; }
  int ndim() const { return static_cast<int>(shape_.size()); }
  int size() const { return size_; }

  float* data() { return data_.get(); }
  const float* data() const { return data_.get(); }

  Tensor clone() const;

 private:
  std::vector<int> shape_;
  int size_ = 0;
  std::unique_ptr<float[]> data_;
};

}  // namespace mini_infer
