# Chess Engine Benchmarking Framework

Comprehensive benchmarking and evaluation suite for comparing CPU (Negamax) and GPU (PUCT/MCTS) chess engines.

## Overview

This testing framework provides production-quality benchmarking tools for chess engine evaluation:

- **Throughput benchmarks**: Measure raw performance (nodes/sec, playouts/sec)
- **Fixed-time quality tests**: Evaluate decision quality under time pressure
- **Stockfish agreement**: Compare against external reference engine
- **Head-to-head matches**: Direct engine competition with Elo estimation

## Architecture

The framework follows these design principles:

1. **Black-box testing**: Engines accessed only through defined interfaces
2. **No source modifications**: All code isolated in `/tests`
3. **CSV output**: Machine-readable results for analysis
4. **Reproducible**: Fixed seeds, standardized positions
5. **Academic rigor**: Publication-quality methodology

## Directory Structure

```
tests/
├── ARCHITECTURE.md          # Detailed design documentation
├── README.md                # This file
├── CMakeLists.txt           # Build system
├── include/                 # Headers
│   ├── engine_interface.h   # Abstract engine wrapper
│   ├── test_positions.h     # Position databases
│   ├── csv_writer.h         # CSV output utilities
│   └── benchmark_utils.h    # Timing & statistics
└── src/                     # Implementations
    ├── cpu_engine_wrapper.cpp
    ├── gpu_engine_wrapper.cpp
    ├── benchmark_throughput.cpp
    ├── benchmark_fixed_time.cpp
    ├── benchmark_stockfish.cpp
    └── benchmark_matches.cpp
```

## Building

### Prerequisites

- CMake 3.18+
- C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- OpenMP support
- CUDA Toolkit 11.0+ (for GPU benchmarks, optional)

### Build Commands

#### CPU-only benchmarks:

```bash
cd tests
mkdir build && cd build
cmake .. -DBUILD_GPU_BENCHMARKS=OFF
make -j$(nproc)
```

#### Full benchmarks (CPU + GPU):

```bash
cd tests
mkdir build && cd build
cmake .. -DBUILD_GPU_BENCHMARKS=ON
make -j$(nproc)
```

This will produce executables:
- `benchmark_throughput` (or `benchmark_throughput_gpu`)
- `benchmark_fixed_time` (or `benchmark_fixed_time_gpu`)
- `benchmark_stockfish` (or `benchmark_stockfish_gpu`)
- `benchmark_matches` (or `benchmark_matches_gpu`)

## Usage

### 1. Throughput Benchmark

Measures raw search performance across various positions.

```bash
./benchmark_throughput --help

# Run with default settings
./benchmark_throughput --output results_throughput.csv

# Custom configuration
./benchmark_throughput \
    --output my_results.csv \
    --cpu-depth 18 \
    --gpu-sims 10000 \
    --time 10000 \
    --difficulty medium
```

**Options:**
- `--output FILE`: Output CSV file
- `--difficulty LEVEL`: `easy`, `medium`, `hard`, or `all`
- `--cpu-depth N`: CPU search depth (default: 15)
- `--gpu-sims N`: GPU simulation count (default: 5000)
- `--time N`: Max time per position in ms (default: 5000)
- `--cpu-only` / `--gpu-only`: Test specific engine
- `--verbose`: Detailed output

**Output format:**
```csv
engine,position_name,fen,time_ms,nodes_or_playouts,throughput,depth
CPU-Negamax,BK01,1k1r4/...,1234,567890,460123,15
GPU-PUCT-MCTS,BK01,1k1r4/...,2345,50000,21321,0
```

### 2. Fixed-Time Quality Benchmark

Evaluates move quality under various time constraints.

```bash
./benchmark_fixed_time --help

# Run Bratko-Kopec suite with standard time budgets
./benchmark_fixed_time --output results_quality.csv

# Custom time budgets
./benchmark_fixed_time \
    --times 100,500,2000,5000 \
    --suite wac \
    --output wac_quality.csv
```

**Options:**
- `--output FILE`: Output CSV file
- `--times MS,MS,...`: Comma-separated time budgets in ms
- `--suite NAME`: Test suite (`bratko-kopec`, `wac`, `performance`, `all`)
- `--cpu-only` / `--gpu-only`: Test specific engine
- `--verbose`: Show move details

**Output format:**
```csv
engine,position_name,fen,time_budget_ms,actual_time_ms,move_uci,eval_cp,depth,nodes
CPU-Negamax,BK01,1k1r4/...,1000,987,d6d1,325,14,234567
```

### 3. Stockfish Agreement Benchmark

Compares engine decisions against Stockfish reference.

**Note:** Requires Stockfish installed and in PATH, or specify path with `--stockfish`.

```bash
./benchmark_stockfish --help

# Run with Stockfish in PATH
./benchmark_stockfish --output results_agreement.csv

# Specify Stockfish path
./benchmark_stockfish \
    --stockfish /usr/local/bin/stockfish \
    --stockfish-depth 22 \
    --engine-time 5000 \
    --suite bratko-kopec
```

**Options:**
- `--output FILE`: Output CSV file
- `--stockfish PATH`: Path to Stockfish binary
- `--suite NAME`: Test suite
- `--stockfish-depth N`: Stockfish search depth (default: 20)
- `--engine-depth N`: Test engine depth (default: 20)
- `--engine-time MS`: Test engine time limit (default: 5000)

**Output format:**
```csv
engine,position_name,fen,stockfish_move,engine_move,top1_match,stockfish_eval,engine_eval,eval_diff
CPU-Negamax,BK01,1k1r4/...,d6d1,d6d1,true,325,310,15
```

**Important:** Current implementation includes a placeholder Stockfish interface. For production use, implement proper UCI protocol communication (see source comments for details).

### 4. Head-to-Head Matches

Plays games between CPU and GPU engines.

```bash
./benchmark_matches --help

# Play 100 games with alternating colors
./benchmark_matches --games 100 --output matches.csv

# Custom configuration
./benchmark_matches \
    --games 200 \
    --time 2000 \
    --max-moves 150 \
    --output long_matches.csv \
    --verbose
```

**Options:**
- `--output FILE`: Output CSV file
- `--games N`: Number of games (default: 100)
- `--time MS`: Time per move in ms (default: 1000)
- `--max-moves N`: Max moves before draw (default: 200)
- `--no-alternate`: Don't alternate colors
- `--opening FEN`: Custom starting position
- `--verbose`: Show move details

**Output format:**
```csv
game_id,white_engine,black_engine,result,moves,termination_reason,final_fen
1,CPU-Negamax,GPU-PUCT-MCTS,1-0,45,checkmate,8/8/...
2,GPU-PUCT-MCTS,CPU-Negamax,1/2-1/2,120,repetition,rnbqk...
```

**Note:** Current implementation uses simplified game management. For full functionality, integrate with a chess library (e.g., the one used by the CPU engine).

## Interpreting Results

### Throughput Analysis

- **Higher = Better**: Compare `throughput` column
- **CPU engines**: Typically measured in millions of nodes/sec
- **GPU engines**: Measured in thousands of playouts/sec
- **Complexity matters**: Branching factor affects performance

### Quality Metrics

- **Accuracy**: % of positions where engine finds best move
- **Time efficiency**: How quickly does quality plateau?
- **Consistency**: Variance across difficulty levels

### Stockfish Agreement

- **Top-1 match rate**: % agreement on best move
- **Eval correlation**: How close are evaluations?
- **Typical results**: Strong engines achieve 70-90% agreement

### Elo Estimation

- **Match results**: Win rate → Elo difference
- **Confidence intervals**: Require ~100+ games for ±50 Elo
- **Interpretation**: ±100 Elo ≈ 64% expected score

## Analyzing Results

### Using Python/Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load throughput results
df = pd.read_csv('results_throughput.csv')

# Compare engines
engine_stats = df.groupby('engine')['throughput'].agg(['mean', 'std', 'min', 'max'])
print(engine_stats)

# Plot throughput distribution
df.boxplot(column='throughput', by='engine')
plt.ylabel('Throughput (ops/sec)')
plt.title('Engine Throughput Comparison')
plt.show()
```

### Using R

```r
library(tidyverse)

# Load fixed-time results
df <- read_csv('results_fixed_time.csv')

# Accuracy by time budget
accuracy <- df %>%
  filter(!is.na(best_move)) %>%
  group_by(engine, time_budget_ms) %>%
  summarize(
    accuracy = mean(move_uci == best_move),
    avg_depth = mean(depth)
  )

# Plot
ggplot(accuracy, aes(x = time_budget_ms, y = accuracy, color = engine)) +
  geom_line() +
  geom_point() +
  scale_x_log10() +
  labs(title = "Move Accuracy vs Time Budget",
       x = "Time (ms)", y = "Accuracy")
```

## Test Suites

### Bratko-Kopec (24 positions)
Classic engine testing positions with known best moves. Mixed tactical and positional.

### Win At Chess (WAC)
Tactical puzzles requiring concrete calculation.

### Performance Suite
Varied complexity positions for throughput testing (opening, middlegame, endgame).

## Troubleshooting

### Build Issues

**CUDA not found:**
```bash
cmake .. -DBUILD_GPU_BENCHMARKS=OFF
```

**OpenMP not available:**
Install development package:
```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS
brew install libomp
```

### Runtime Issues

**GPU benchmark crashes:**
- Check CUDA installation: `nvidia-smi`
- Verify compute capability matches CMake settings
- Try reducing batch size in GPU engine config

**Slow performance:**
- Reduce time budgets or depth limits
- Use smaller test suites for quick validation
- Check CPU/GPU utilization

### CSV Output Issues

**Corrupted CSV:**
- Ensure write permissions in output directory
- Check disk space
- Verify benchmark completed (not killed mid-execution)

## Google Colab Integration

See `COLAB_TUTORIAL.md` for complete step-by-step guide to running these benchmarks in Google Colab, including:

- CUDA environment setup
- Dependency installation
- Compilation instructions
- Result visualization
- Downloading results

## Contributing

When adding new benchmarks:

1. Follow existing code structure
2. Use CSV output for consistency
3. Add command-line help
4. Update this README
5. Test on both CPU and GPU configurations

## License

Same as parent project (MIT License).

## Citation

If you use this benchmarking framework in academic work, please cite:

```bibtex
@software{chess_benchmark_2024,
  author = {Your Name},
  title = {Chess Engine Benchmarking Framework},
  year = {2024},
  url = {https://github.com/yourusername/chess-engine-benchmarks}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
