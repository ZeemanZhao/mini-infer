# mini-infer — Claude Project Guidance

> Parent `CLAUDE.md` (at `~/Desktop/CFP-Study/CLAUDE.md`) defines the tutoring philosophy and is still in effect. This file adds project-specific conventions only.

## What this project is

A minimal C++ inference engine, built **incrementally** to demonstrate end-to-end deep learning deployment systems engineering. Long-term target: load ONNX models, run optimized inference for **both vision (ResNet-18) and LLM (Qwen2-0.5B with KV cache)** on CPU + CUDA, with hand-written graph IR / memory planner / op fusion. Career goal: 推理引擎 / 部署工程师 (主攻方向).

## Build philosophy: Spiral MVP

**Core rule**: every phase produces a *running* artifact. Modules grow incrementally; we don't design the full architecture up front. **Do not over-engineer for future phases.**

But MVP ≠ skipping engineering basics. Phase 0 must ship with **professional engineering scaffolding** (CMake / GoogleTest / GitHub Actions CI / clang-format / smart pointers) even when the functional surface is tiny. Recruiters scan repos for these signals; missing them is a silent reject.

**See `docs/ROADMAP.md` for the full 9-phase plan and what each phase delivers.**

## Current status

- **Phase**: 0 (MVP — MLP inference, CPU only, hardcoded graph)
- **What's done**: repo + LICENSE + README + this guidance + roadmap
- **Next task**: CMake skeleton (with FetchContent gtest + `.clang-format` + pybind11 hooks reserved + GitHub Actions CI) → Tensor class
- **Then**: MatMul + ReLU naive ops → PyTorch MLP weight export script → MNIST gtest assertion → demo `main`

## MVP scope (Phase 0) — strict

**Functional IN scope**:
- `Tensor` class: shape, owned data buffer via `std::unique_ptr<float[]>`, float32 only, explicit move semantics, copy disabled
- `MatMul` op (naive triple-loop, no optimization)
- `ReLU` op (element-wise, single loop)
- Hardcoded MLP: `Linear(784,128) → ReLU → Linear(128,10)` as function calls in `main()`
- Weight loading: PyTorch trains MLP, exports raw binary (or `.npy`); C++ reads with `ifstream`
- MNIST accuracy validation ≥ 95% (asserted via gtest)

**Engineering scaffolding IN scope** (added based on mid-session revision — recruiter signals):
- CMake ≥ 3.20, C++17, out-of-source build (`build/` gitignored)
- `option(BUILD_TESTS ON)` and `option(BUILD_PYTHON_BINDING OFF)` reserved
- GoogleTest integrated via `FetchContent` (no manual install)
- One integration test (`tests/test_mlp_accuracy.cpp`): MLP accuracy ≥ 95%
- GitHub Actions workflow: build + test on every push; CI badge in README
- `.clang-format` (Google or LLVM style — pick one and commit)
- pybind11 CMake hooks reserved (no actual binding code yet — Phase 1)

**Functional OUT of scope** (do NOT add until later phases):
- ❌ ONNX parsing / protobuf — Phase 3
- ❌ Graph / Node / Tensor IR abstraction — Phase 1
- ❌ Topological sort — Phase 1
- ❌ Memory planner — Phase 2
- ❌ Conv2D / BN / Pool — Phase 4
- ❌ LLM / KV cache / Attention — Phase 7
- ❌ CUDA backend — Phase 8
- ❌ Multi-threading / SIMD / cache tiling / any optimization
- ❌ Unit tests beyond the 1 integration test (Phase 1+ adds per-module unit tests)
- ❌ Actual pybind11 binding code (Phase 1)

**Budget**: ~600–800 lines of C++ (functional + scaffolding) + ~30 lines of Python. **2–3 weeks** of evening work. Exceeding this signals scope creep — push back.

## Coding conventions (locked)

- **C++ standard**: **C++17 only**. Do not use C++20. No real use case in this project; using C++20 for resume signaling is reverse signal — interviewers will ask "why" and "for what" and find no good answer.
- **CMake**: minimum 3.20, out-of-source build, prefer `target_*` commands (not directory-level `include_directories`)
- **Header layout**: public headers in `include/mini_infer/<name>.h`, sources in `src/<name>.cpp`
- **Namespace**: everything in `mini_infer::`
- **Naming**: `CamelCase` for classes (`Tensor`, `Graph`), `snake_case` for functions/variables (`get_shape`, `data_ptr`)
- **No `using namespace std`** in headers; OK in `.cpp` only when scoped to a function
- **Style**: `.clang-format` enforced. Run before commit.

### Memory ownership rules (no exceptions)

- **No raw `new` or `delete` anywhere** — period. Phase 0 through Phase 9.
- **Phase 0 Tensor**: owns its data buffer via `std::unique_ptr<float[]>`. Single owner, move semantics, copy disabled.
- **Phase 1 Graph**: tensors shared between nodes upgrade to `std::shared_ptr<Tensor>`. The deliberate `unique_ptr → shared_ptr` transition is **interview-quality material** — be ready to explain why each phase chose what it did.
- **Tensor must declare**:
  ```cpp
  Tensor(Tensor&&) noexcept = default;
  Tensor& operator=(Tensor&&) noexcept = default;
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  ```
  Explicit move, no accidental copy. If deep copy is later needed, add `Tensor clone() const`.
- **No `shared_ptr` until needed**: Phase 0 has no shared ownership scenario.

### Operator implementation policy (locked)

- **Hand-write naive correct version of every op.** No exceptions for "I'll just call a library."
- **No SIMD / multi-threading / cache tiling** in Phase 0–7. Naive triple-loop MatMul is the standard. Optimization is Phase 8+ (CUDA).
- **Conv2D must be implemented via im2col** (Phase 4) — non-negotiable. This is interview gold.
- **Fused kernels** (Phase 6) are written by hand as real kernels, not composed from non-fused ops. The whole point of fusion is the fused kernel runs fewer memory passes (Conv+BN+ReLU goes from 3 mem passes to 1).
- **Do NOT use OpenBLAS / Eigen / cuBLAS** as substitutes for hand-written ops. BLAS comparison benchmarks were considered and explicitly rejected (no Phase 4 BLAS comparison work).
- **Phase 7 LLM ops**: Attention is naive scaled-dot-product + causal mask + softmax. **Do not attempt FlashAttention** — out of scope.

## Git workflow

- **Branch**: work directly on `main` (single contributor; branching adds friction without payoff for an MVP)
- **Commit message**: Conventional Commits — `<type>: <description>`
  - Types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`
  - Examples: `feat: add Tensor class with unique_ptr ownership`, `ci: add GitHub Actions build + test workflow`, `build: integrate GoogleTest via FetchContent`
- **Before every commit**: `git status` then `git diff` — never blind `git add .`
- **Stage specific files**: `git add include/mini_infer/tensor.h src/tensor.cpp`
- **Push cadence**: push after each meaningful commit (don't accumulate locally)

## Teaching mode reminders (for Claude)

This project is a learning vehicle. The user is competent in C/C++ but new to inference-engine design and modern C++ idioms. When introducing new code:

1. **Explain every line** of new files: `#include`s, namespaces, RAII patterns, move semantics, `FetchContent`, `target_link_libraries`, GitHub Actions YAML — nothing is "self-explanatory".
2. **Connect to interview value**: after each module, note "this is how it'd be asked in 推理引擎 interviews".
3. **Don't over-design**: refuse to add abstractions Phase 0 doesn't need, even if "we'll need them later". **Exception**: engineering scaffolding (CMake hooks, gtest scaffold, CI) IS Phase 0 work — see scaffolding list above.
4. **Refer to ROADMAP.md before suggesting features** — if it's listed as Phase 3+, defer it.
5. **Honesty over flattery**: if user proposes something inconsistent with the locked plan (e.g., "let me skip ops and just do fusion"), say no with reasoning. Per parent CLAUDE.md: 严格纠错，不迎合不奉承.

## Repo layout (target after Phase 0)

```
mini-infer/
├── CMakeLists.txt
├── CLAUDE.md, README.md, LICENSE, .clang-format
├── .github/workflows/ci.yml         # build + test
├── docs/ROADMAP.md
├── include/mini_infer/
│   ├── tensor.h
│   └── ops.h
├── src/
│   ├── tensor.cpp
│   ├── ops/
│   │   ├── matmul.cpp
│   │   └── relu.cpp
│   └── main.cpp                     # hardcoded MLP forward (demo)
├── scripts/
│   └── export_mlp.py                # PyTorch trains + dumps weights
└── tests/
    └── test_mlp_accuracy.cpp        # gtest: MNIST accuracy ≥ 95%
```

Files appear as their phase needs them — don't pre-create empty stubs.

## Dev environment notes

- **Primary dev machine**: macOS (this repo lives at `~/Desktop/CFP-Study/projects/mini-infer/`)
- **CUDA dev** (Phase 8 only): Windows + RTX 4060 + WSL2 (Ubuntu, ext4 filesystem, NOT `/mnt/c/...`)
- **GitHub auth**: SSH over port 443 (`~/.ssh/config` configures `Host github.com → Hostname ssh.github.com Port 443`) — required because of TUN-mode proxy on user's machine. Do not change to HTTPS or port 22.
