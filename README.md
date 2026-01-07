# Chess Engine - CPU & GPU Parallelization

This project contains chess engine implementations optimized for parallelization on both CPU and GPU.

## Project Structure

- **CPU version** (`cpu/`): Minimax/Negamax algorithm with OpenMP parallelization
- **GPU version** (`gpu/`): Monte Carlo Tree Search (MCTS) implementation with CUDA

## CPU Version - Quick Start

### Prerequisites
- Meson build system
- C++ compiler with OpenMP support

### Building
```bash
cd cpu
meson setup build
meson compile -C build
```

### Running
```bash
# Run tests
build/chess-tests

# Run engine with depth 10
build/test_suite_parallel --mode=depth --depth=10 --level=easy
```

## GPU Version - Quick Start

### Prerequisites
- NVIDIA GPU (Compute Capability 7.5+)
- CUDA Toolkit 12.6 or 13.1+
- Windows with MSVC compiler

### Building
```bash
cd gpu
scripts\build.bat        # Build MCTS engine
scripts\build_tests.bat  # Build test suite
```

### Running
```bash
# Run MCTS engine
build\main.exe

# Run test suite
build\test_runner.exe --all
build\test_runner.exe --easy      # Tactical tests (mate-in-1)
build\test_runner.exe --perft     # Move generation tests
```

See [gpu/README.md](gpu/README.md) for more details.

## CPU vs GPU Comparison

| Aspect | CPU (Minimax) | GPU (MCTS) |
|--------|---------------|------------|
| Algorithm | Deterministic search | Stochastic simulations |
| Parallelization | OpenMP (threads) | CUDA (thousands of threads) |
| Evaluation | Heuristics + piece-square | Win/loss statistics |
| Speed | ~1M positions/s | ~50K simulations/s |
| Accuracy | High at shallow depth | Increases with simulations |

## License

MIT License
