# mini-infer Roadmap

A minimal C++ inference engine, grown incrementally via Spiral MVP. Each phase produces a runnable artifact; modules are added when their phase needs them, not designed up front. Engineering scaffolding (CMake / GoogleTest / GitHub Actions CI / clang-format) is set up in Phase 0 even though functional surface is tiny — these are the signals recruiters look for.

## Architecture (target — end of Phase 7)

```
PyTorch model (vision OR LLM)
     ↓ (export)
ONNX file
     ↓ (Phase 3: parse)
Internal Graph IR ─→ (Phase 6: opt passes — Conv-BN-ReLU fusion, BN fold, const folding)
     ↓ (Phase 1: topological sort)
Execution order
     ↓ (Phase 2: memory planner — lifetime + Greedy Best-Fit; Phase 7 adds KV cache)
Buffer plan
     ↓ (Phase 4 / Phase 7: operator library — Conv2D im2col, BN, MatMul, Attention, LayerNorm, ...)
Inference output
     ├─→ (Phase 5: validate against PyTorch on ResNet-18)
     └─→ (Phase 7: validate text generation against HuggingFace on Qwen2-0.5B)
```

Phase 8/9 are bonus: CUDA backend and INT8 quantization.

## Phases

| Phase | Goal | Deliverable | Time |
|-------|------|-------------|------|
| **0 — MVP + scaffolding** | End-to-end MLP runs in C++; professional engineering scaffolding in place | Hardcoded `Linear→ReLU→Linear` matches PyTorch on MNIST; CMake + GoogleTest + GitHub Actions CI + clang-format; CI badge green on README | 2–3 weeks |
| **1 — Graph IR + Python binding** | Replace hardcode with a general DAG; expose minimal Python interface | `Graph`/`Node`/`Tensor` classes (shared_ptr where appropriate); Kahn topological sort; executor walks the graph; pybind11 bindings for `Tensor`, `Graph`, `Engine.run()` | 2 weeks |
| **2 — Memory planner** ⭐ | Reuse buffers across non-overlapping tensor lifetimes (vision-style, dense static graphs) | Lifetime analysis + Greedy Best-Fit allocator (`std::map + lower_bound`); README prints "without reuse: X MB / with reuse: Y MB"; allocator unit tests | 2 weeks |
| **3 — ONNX loading** | Real models can be loaded | protobuf-based ONNX parser → internal Graph IR; supports the op set needed for ResNet-18 | 1–2 weeks |
| **4 — Operator library** | Cover ResNet-18 op set | MatMul (cleanup), Conv2D (im2col, mandatory), BatchNorm, MaxPool, GlobalAvgPool, Add — all naive correct, all unit-tested. **No BLAS/Eigen/cuBLAS substitutions.** | 2 weeks |
| **5 — End-to-end ResNet-18** | Real model runs and matches PyTorch | Export PyTorch ResNet-18 → ONNX → run in mini-infer; logits match PyTorch to 1e-4; performance numbers in README | 1 week |
| **6 — Graph optimization passes** ⭐ | Demonstrate inference-engine differentiation | Pass manager; Conv+BN+ReLU **fused kernel** (the kernel itself, not just graph rewrite); BN fold into Conv weights; constant folding; dead node elimination | 2 weeks |
| **7 — Tiny LLM support** ⭐ | Modern LLM inference capability — covers AI-unicorn / LLM-ecosystem hiring | Qwen2-0.5B (or GPT-2 small / TinyLlama-1.1B) loaded and generates text; KV cache management; Attention op (naive scaled-dot-product + causal mask + softmax); LayerNorm; Embedding lookup; greedy + top-k sampling; Python `generate(prompt)` via pybind11 | 3–4 weeks |
| **8 — CUDA backend** (bonus) | Multi-backend abstraction | `OpKernel` interface with CPU and CUDA implementations; tiled MatMul kernel ported from `cuda-kernels` sandbox; Nsight numbers in README | flexible |
| **9 — INT8 quantization** (bonus) | Quantization engineering | Per-tensor INT8 quantize/dequantize ops; INT8 MatMul; accuracy delta on ResNet-18 | flexible |

## Phase 0 — MVP scope detail

**Goal (one-line acceptance test)**: load PyTorch-trained MLP weights in C++, run inference on MNIST test set, GoogleTest assertion passes (accuracy ≥ 95%), GitHub Actions CI green.

**Functional scope (in)**:
- `Tensor` class: shape, owned data via `std::unique_ptr<float[]>`, float32 only, explicit move semantics (`= default`), copy `= delete`
- `MatMul` op: naive triple-loop, no optimization
- `ReLU` op: element-wise, single loop
- Hardcoded forward: `Linear(784,128) → ReLU → Linear(128,10)` written as function calls in `main()`
- Weight export: PyTorch trains MLP, dumps weights as raw binary (or `.npy`)
- Weight loading: C++ reads with `ifstream` into `Tensor`

**Engineering scaffolding scope (in)**:
- CMake (≥3.20), C++17, out-of-source build, `option(BUILD_TESTS ON)`, `option(BUILD_PYTHON_BINDING OFF)` reserved
- GoogleTest integrated via `FetchContent` (no manual install)
- One integration test (`tests/test_mlp_accuracy.cpp`): assert accuracy ≥ 95%
- GitHub Actions workflow: build + run tests on push, badge in README
- `.clang-format` (Google or LLVM style — pick one and commit)
- pybind11 CMake hooks reserved (no actual binding code yet)

**Out of scope (do not add)**:
- Graph IR, topological sort, memory planner — Phases 1–2
- ONNX, protobuf — Phase 3
- Conv2D, BN, Pool — Phase 4
- LLM, Attention, KV cache — Phase 7
- CUDA, multi-threading, SIMD — Phase 8
- Unit tests beyond the 1 integration test
- Actual pybind11 bindings (Phase 1)

**Budget**: ~600–800 lines of C++ (functional + scaffolding) + ~30 lines of Python. 2–3 weeks of evening work. Exceeding this signals scope creep.

## Operator policy (Phases 4, 6, 7 — locked)

- **Every op is hand-written naive correct.** No SIMD, no tiling, no library substitutions (no OpenBLAS / Eigen / cuBLAS).
- **Conv2D uses im2col** (Phase 4) — non-negotiable. Interview gold.
- **Fused kernels are real kernels** (Phase 6), not graph rewrites composed of non-fused ops. Conv+BN+ReLU fused = one kernel that does Conv, applies BN affine inline, applies ReLU on store. Memory passes drop from 3 to 1 — that is the entire point of fusion.
- **Phase 7 LLM ops** (Attention / LayerNorm / Embedding) are naive. Do not attempt FlashAttention.

## Why this project (interview narrative)

Two parallel tracks:
- **Internship** (Cambricon, NPU Runtime layer): graph scheduling DAG optimization, async memory pool, multi-backend Runtime wrapper. Targets *Runtime* job direction.
- **Open-source mini-infer** (this project): graph IR, memory planning, op fusion, end-to-end vision (ResNet-18) AND LLM (Qwen2-0.5B). Targets *推理引擎 / 部署* job direction (primary).

The combined narrative covers Runtime + inference-engine layers AND static-shape vision + dynamic-shape LLM workloads — a full upstream/downstream + cross-domain view that single-track candidates lack.

**Target companies (混投)**:
- NPU vendors — Cambricon (实习公司转正), Horizon, HiSilicon Ascend, Black Sesame
- LLM inference ecosystem — NVIDIA TRT-LLM, vLLM team, Volc Engine, Qianfan, LLM startups (硅基流动 etc.)
- Internet majors — ByteDance, Alibaba, Tencent, Baidu, Meituan
- AI unicorns — Zhipu, Moonshot, MiniMax, Stepfun, 01.AI

## License

MIT.
