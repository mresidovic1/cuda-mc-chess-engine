# GPU Chess Engine - Quick Start

GPU-accelerated chess engine using Monte Carlo Tree Search (MCTS) with CUDA.

## Prerequisites

- NVIDIA GPU with CUDA support (Compute Capability 7.5+)
- CUDA Toolkit 12.6 or 13.1+
- Windows with MSVC compiler

## Building

```bash
cd gpu

# Build MCTS engine
scripts\build.bat

# Build test suite
scripts\build_tests.bat
```

**Note**: Adjust `-arch=sm_75` in build scripts to match your GPU:
- RTX 20xx: `sm_75`
- RTX 30xx: `sm_86`
- RTX 40xx: `sm_89`

## Running

### MCTS Engine

```bash
build\main.exe
```

### Test Suite

```bash
# Run all tests
build\test_runner.exe --all

# Run specific test suites
build\test_runner.exe --perft     # Move generation validation
build\test_runner.exe --easy      # Tactical tests (mate in 1)
build\test_runner.exe --medium    # Tactical tests (mate in 4-5)
build\test_runner.exe --hard      # Tactical tests (mate in 8-12)
```

## Current Limitations

- **Tactical solver depth**: Fixed at 2 plies (solves mate-in-1 only)
- Medium/hard tactical tests require deeper search (not yet implemented)

## Test Results

| Test Suite | Pass Rate |
|------------|-----------|
| Perft (move generation) | 100% (16/16) |
| Easy tactical (mate-in-1) | 100% (5/5) |
| Medium tactical (mate-in-4/5) | 0-20% |
| Hard tactical (mate-in-8/12) | 0-20% |
