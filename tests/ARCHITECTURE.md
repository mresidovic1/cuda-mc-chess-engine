# Test Framework Architecture

## Overview

This directory contains a comprehensive benchmarking and evaluation framework for comparing the CPU (Negamax) and GPU (MCTS/PUCT) chess engines.

## Design Principles

1. **Black-box testing**: Tests interact with engines only through defined interfaces
2. **Minimal modifications**: No changes to `/cpu` or `/gpu` source code
3. **CSV output**: All results in machine-readable format for analysis
4. **Reproducibility**: Fixed seeds, standardized test positions
5. **Academic rigor**: Publication-quality benchmarking methodology

## Architecture Components

### 1. Engine Interfaces (`engine_interface.h`)

Abstract base class for unified engine interaction:

```cpp
class EngineInterface {
public:
    virtual Move search(const Position& pos, SearchParams& params) = 0;
    virtual void reset() = 0;
    virtual std::string get_name() const = 0;
};
```

Concrete implementations:
- `CPUEngine`: Wraps `find_best_move()` from `/cpu`
- `GPUEngine`: Wraps `PUCTEngine::search()` from `/gpu`

### 2. Test Suites

#### A. Throughput Benchmark (`benchmark_throughput.cpp`)
- Measures raw search performance
- CPU: nodes per second
- GPU: playouts/simulations per second
- Multiple positions, varying complexity
- CSV format: `engine,position,time_ms,nodes/playouts,throughput`

#### B. Fixed-Time Quality (`benchmark_fixed_time.cpp`)
- Time budgets: 50ms, 100ms, 500ms, 1s, 5s
- Records: move chosen, evaluation score, depth/sims
- Tactical positions from test suites
- CSV format: `engine,position,time_budget,move,eval,depth`

#### C. Stockfish Agreement (`benchmark_stockfish.cpp`)
- External reference engine comparison
- Top-1 move agreement rate
- Top-3 move agreement rate
- Evaluation correlation
- CSV format: `engine,position,stockfish_move,engine_move,match,eval_diff`

#### D. Head-to-Head Matches (`benchmark_matches.cpp`)
- Play GPU vs CPU games
- Opening book or standard positions
- Record wins/draws/losses
- Compute Elo rating difference (±confidence interval)
- CSV format: `game_id,white,black,result,moves,termination`

### 3. Supporting Infrastructure

#### Position Sets (`test_positions.h`)
- Standard tactical test suites (BratkoKopec, WAC, etc.)
- Positions of varying complexity:
  - Simple: midgame, clear best move
  - Complex: tactical puzzles
  - Endgame: tablebase positions

#### CSV Writers (`csv_writer.h`)
- Type-safe CSV output
- Automatic escaping
- Buffered writes for performance

#### Timing Utilities (`benchmark_utils.h`)
- High-precision timing
- Cross-platform compatibility
- Time budget enforcement

## Build System

CMake-based build with two targets:
1. **CPU-only benchmarks**: Link against `/cpu` engine
2. **Full benchmarks**: Link against both `/cpu` and `/gpu` (requires CUDA)

## Data Flow

```
Test Position → Engine Wrapper → Raw Engine → Result
     ↓              ↓                           ↓
  test_positions   CPUEngine/GPUEngine      CSV File
                   (adapters)                  ↓
                                           Analysis
```

## Usage Workflow

1. Compile engines: `cd cpu && make` and `cd gpu && make`
2. Compile tests: `cd tests && cmake .. && make`
3. Run benchmarks:
   - `./benchmark_throughput --output results_throughput.csv`
   - `./benchmark_fixed_time --time 1000 --output results_quality.csv`
   - `./benchmark_stockfish --stockfish-path /usr/bin/stockfish --output results_agree.csv`
   - `./benchmark_matches --games 100 --output results_matches.csv`
4. Analyze results: CSV files ready for Python/R analysis

## Google Colab Integration

All benchmarks designed to run in Colab notebooks:
- GPU availability detection
- Automated dependency installation
- Result download helpers
- Visualization notebooks

## File Structure

```
tests/
├── ARCHITECTURE.md              # This file
├── README.md                     # User documentation
├── CMakeLists.txt                # Build system
├── include/
│   ├── engine_interface.h        # Abstract engine wrapper
│   ├── test_positions.h          # Position databases
│   ├── csv_writer.h              # CSV output utilities
│   └── benchmark_utils.h         # Timing & helpers
├── src/
│   ├── benchmark_throughput.cpp  # Raw performance test
│   ├── benchmark_fixed_time.cpp  # Quality vs time test
│   ├── benchmark_stockfish.cpp   # External reference test
│   ├── benchmark_matches.cpp     # Head-to-head games
│   ├── cpu_engine_wrapper.cpp    # CPU adapter
│   └── gpu_engine_wrapper.cpp    # GPU adapter
└── scripts/
    ├── run_all_benchmarks.sh     # Convenience runner
    └── colab_setup.sh            # Colab environment setup
```

## Extension Points

Future enhancements:
- Opening book integration
- Endgame tablebase verification
- Multi-threaded batch testing
- Real-time visualization dashboard
- Automated regression detection
