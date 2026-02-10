# GPU Chess Engine

GPU-accelerated engine based on PUCT/MCTS (CUDA).

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- CMake 3.18+
- MSVC or another C++17 compiler supported by your CUDA toolkit

## Build

From the repository root, generate build files and compile:

```bash
cmake -S tests -B tests/build
cmake --build tests/build --config Release
```

### Notes

- Ensure `nvcc` is on your PATH (provided by the CUDA Toolkit).
- If your GPU architecture is not covered by the default CMake settings, update `CUDA_ARCHITECTURES` in `tests/CMakeLists.txt`.
