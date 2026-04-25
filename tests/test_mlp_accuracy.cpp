#include <gtest/gtest.h>

#include <algorithm>
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

std::vector<int> load_labels(const std::string& path, int n) {
  std::vector<int> v(n);
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open: " + path);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(int)));
  if (!f) throw std::runtime_error("short read: " + path);
  return v;
}

}  // namespace

TEST(MLPAccuracy, MNIST95) {
  const std::string dir = DATA_DIR;
  constexpr int N = 10000;

  auto w1 = load_tensor(dir + "/fc1_weight.bin", {784, 128});
  auto b1 = load_tensor(dir + "/fc1_bias.bin",   {128});
  auto w2 = load_tensor(dir + "/fc2_weight.bin", {128, 10});
  auto b2 = load_tensor(dir + "/fc2_bias.bin",   {10});

  auto images = load_tensor(dir + "/mnist_test_images.bin", {N, 784});
  auto labels = load_labels(dir + "/mnist_test_labels.bin", N);

  using namespace mini_infer::ops;

  int correct = 0;
  const float* img_ptr = images.data();

  for (int n = 0; n < N; ++n) {
    mini_infer::Tensor x({1, 784});
    std::copy(img_ptr + n * 784, img_ptr + (n + 1) * 784, x.data());

    auto out = bias_add(matmul(relu(bias_add(matmul(x, w1), b1)), w2), b2);

    const float* logits = out.data();
    int predicted = static_cast<int>(std::max_element(logits, logits + 10) - logits);
    if (predicted == labels[n]) ++correct;
  }

  double accuracy = static_cast<double>(correct) / N;
  std::printf("MNIST test accuracy: %.4f  (%d/%d)\n", accuracy, correct, N);
  EXPECT_GE(accuracy, 0.95) << "accuracy " << accuracy << " < 0.95";
}
