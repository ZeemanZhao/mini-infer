# mini-infer

A minimal C++ inference engine for both vision and LLM workloads, built incrementally to demonstrate end-to-end deep learning deployment systems engineering.

> **Status**: Phase 0 (MVP) — bootstrapping repo, CMake, GoogleTest, CI. See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the 9-phase plan.

[![CI](https://github.com/ZeemanZhao/mini-infer/actions/workflows/ci.yml/badge.svg)](https://github.com/ZeemanZhao/mini-infer/actions)

## Goals

- Load ONNX models and run inference end-to-end (target: ResNet-18 in Phase 5; Qwen2-0.5B / TinyLlama in Phase 7)
- Hand-written graph IR with Kahn topological scheduling
- Memory planner — tensor lifetime analysis + Greedy Best-Fit buffer reuse (vision); KV cache management (LLM)
- Graph optimization passes — operator fusion (Conv+BN+ReLU), BN fold, constant folding, dead-node elimination
- Multi-backend execution — CPU first, CUDA later
- Python interface via pybind11 from Phase 1 onward
- Engineering rigor from day one — modern C++17, smart-pointer ownership, GoogleTest, GitHub Actions CI, clang-format

Designed as a learning artifact for inference-engine systems engineering, grown via **Spiral MVP** — every phase ships a runnable artifact; modules are added incrementally rather than designed up front.

## Roadmap

| Phase | Capability | Status |
|-------|------------|--------|
| 0 | MVP: hardcoded MLP + MNIST validation + engineering scaffolding | ✅ done |
| 1 | Graph IR + topological sort + Python binding (pybind11) | ⬜ |
| 2 | Memory planner (lifetime analysis + Greedy Best-Fit) | ⬜ |
| 3 | ONNX loading (protobuf → internal IR) | ⬜ |
| 4 | Operator library (Conv2D im2col, BN, Pool, MatMul, Add) | ⬜ |
| 5 | End-to-end ResNet-18 vs PyTorch | ⬜ |
| 6 | Graph optimization passes (Conv-BN-ReLU fusion, BN fold) | ⬜ |
| 7 | **Tiny LLM** (Qwen2-0.5B / TinyLlama) — KV cache, Attention, sampling | ⬜ |
| 8 | CUDA backend (bonus) | ⬜ |
| 9 | INT8 quantization (bonus) | ⬜ |

Detailed scope per phase: [`docs/ROADMAP.md`](docs/ROADMAP.md).

## Build

```bash
# 1. Generate weights + test data (requires PyTorch)
pip install torch torchvision numpy
python scripts/export_mlp.py

# 2. Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# 3. Test (asserts MNIST accuracy ≥ 95%)
cd build && ctest --output-on-failure

# 4. Demo
./build/mini_infer
```

## License

[MIT](LICENSE).
