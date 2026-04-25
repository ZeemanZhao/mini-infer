#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mini_infer/ops.h"
#include "mini_infer/tensor.h"

namespace {

mini_infer::Tensor load_tensor(const std::string& path, std::vector<int> shape) {
  mini_infer::Tensor t(std::move(shape));
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open: " + path);
  f.read(reinterpret_cast<char*>(t.data()), static_cast<std::streamsize>(t.size() * sizeof(float)));
  if (!f) throw std::runtime_error("short read: " + path);
  return t;
}

}  // namespace

int main() {
  const std::string dir = DATA_DIR;

  // Weights exported transposed from PyTorch: shape = [in_features, out_features]
  auto w1 = load_tensor(dir + "/fc1_weight.bin", {784, 128});
  auto b1 = load_tensor(dir + "/fc1_bias.bin",   {128});
  auto w2 = load_tensor(dir + "/fc2_weight.bin", {128, 10});
  auto b2 = load_tensor(dir + "/fc2_bias.bin",   {10});

  // Blank 28×28 image (all zeros) — smoke test only
  mini_infer::Tensor x({1, 784});

  using namespace mini_infer::ops;
  auto out = bias_add(matmul(relu(bias_add(matmul(x, w1), b1)), w2), b2);

  const float* logits = out.data();
  int predicted = 0;
  for (int i = 1; i < 10; ++i)
    if (logits[i] > logits[predicted]) predicted = i;

  std::printf("Smoke test — blank image predicted class: %d\n", predicted);
  std::printf("Logits:");
  for (int i = 0; i < 10; ++i) std::printf(" %.3f", logits[i]);
  std::printf("\n");
  return 0;
}
