# Chess Engine Benchmarks - Quick Start Guide

Get started in 5 minutes!

## TL;DR - Fastest Path

### Local Machine (Linux/Mac)

```bash
# 1. Navigate to tests directory
cd tests

# 2. Build
mkdir build && cd build
cmake .. -DBUILD_GPU_BENCHMARKS=OFF  # Or ON if you have CUDA
make -j$(nproc)

# 3. Run
./benchmark_throughput --output results.csv --time 3000
```

### Google Colab

```python
# Cell 1: Setup
!nvidia-smi
!apt-get update -qq && apt-get install -qq cmake build-essential libomp-dev

# Cell 2: Get code
!git clone YOUR_REPO_URL
%cd PROJECT_NAME/tests

# Cell 3: Build
!mkdir -p build && cd build
!cmake .. -DCMAKE_CUDA_ARCHITECTURES="75" && make -j$(nproc)

# Cell 4: Run
!./build/benchmark_throughput_gpu --output results.csv --time 3000
```

## 30-Second Decision Tree

**Q: Do you have a GPU?**

├─ **NO** → Use CPU-only benchmarks
│  ```bash
│  cmake .. -DBUILD_GPU_BENCHMARKS=OFF
│  make
│  ./benchmark_throughput --cpu-only
│  ```
│
└─ **YES** → Is it NVIDIA with CUDA?
   │
   ├─ **NO** → Use CPU-only benchmarks (same as above)
   │
   └─ **YES** → Full GPU benchmarks
      ```bash
      cmake .. -DBUILD_GPU_BENCHMARKS=ON
      make
      ./benchmark_throughput_gpu
      ```

## First-Time Setup Checklist

### Prerequisites

- [ ] CMake 3.18+ installed (`cmake --version`)
- [ ] C++17 compiler (GCC 9+, Clang 10+, MSVC 2019+)
- [ ] OpenMP support (`echo | g++ -fopenmp -x c++ -c -`)
- [ ] CUDA Toolkit 11+ (for GPU, `nvcc --version`)

### Install Missing Tools

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake build-essential libomp-dev
```

**macOS:**
```bash
brew install cmake libomp
```

**Windows:**
- Install Visual Studio 2019+ with C++ support
- Install CMake from https://cmake.org/download/
- (Optional) Install CUDA Toolkit from NVIDIA

## Running Your First Benchmark

### Step 1: Build

```bash
cd /path/to/cuda-mc-chess-engine/tests
mkdir build && cd build

# Choose ONE:
cmake .. -DBUILD_GPU_BENCHMARKS=OFF   # CPU only
cmake .. -DBUILD_GPU_BENCHMARKS=ON    # CPU + GPU

make -j$(nproc)
```

**Expected output:**
```
[100%] Built target benchmark_throughput
[100%] Built target benchmark_fixed_time
[100%] Built target benchmark_stockfish
[100%] Built target benchmark_matches
```

### Step 2: Run Simple Test

```bash
# Quick test (30 seconds)
./benchmark_throughput --cpu-only --difficulty easy --time 1000
```

**Expected output:**
```
========================================
Chess Engine Throughput Benchmark
========================================

Testing 8 positions
Output file: results_throughput.csv

CPU engine initialized

Testing CPU-Negamax...

Benchmark Progress: 8/8 (100.0%) Elapsed: 12.3s ETA: 0.0s   

========================================
Benchmark Complete!
Results saved to: results_throughput.csv
========================================
```

### Step 3: View Results

```bash
cat results_throughput.csv
```

or

```python
import pandas as pd
df = pd.read_csv('results_throughput.csv')
print(df.head())
```

## Common First-Run Issues

### Issue 1: "CMake version too old"

```bash
# Install newer CMake
pip install cmake --upgrade
# Or download from https://cmake.org/download/
```

### Issue 2: "OpenMP not found"

```bash
# Ubuntu/Debian
sudo apt-get install libomp-dev

# macOS
brew install libomp
```

### Issue 3: "CUDA not found" (when building GPU)

**Option A**: Disable GPU
```bash
cmake .. -DBUILD_GPU_BENCHMARKS=OFF
```

**Option B**: Install CUDA
- Download from https://developer.nvidia.com/cuda-downloads
- Or use package manager: `sudo apt-get install nvidia-cuda-toolkit`

### Issue 4: Compilation fails

```bash
# Clean build
rm -rf build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j1  # Single-threaded for better error messages
```

### Issue 5: "engine_interface.h: No such file"

**You're in wrong directory!**
```bash
# Should be in tests/build
pwd  # Should show: .../cuda-mc-chess-engine/tests/build

# If not:
cd /path/to/cuda-mc-chess-engine/tests/build
```

## Quick Benchmark Reference

### 1. Throughput (Raw Performance)

**What**: How fast can engines search?

```bash
./benchmark_throughput --output throughput.csv --time 5000
```

**Time**: ~5-10 minutes for full suite

### 2. Fixed-Time Quality (Best Move Under Time Pressure)

**What**: Move quality at different time budgets

```bash
./benchmark_fixed_time --output quality.csv --times 100,500,1000
```

**Time**: ~10-15 minutes for Bratko-Kopec suite

### 3. Stockfish Agreement (Compare to Reference)

**What**: How often do engines agree with Stockfish?

**Requires**: Stockfish installed (`sudo apt-get install stockfish`)

```bash
./benchmark_stockfish --output agreement.csv --stockfish /usr/bin/stockfish
```

**Time**: ~20-30 minutes (depends on depth)

### 4. Head-to-Head Matches (Engine vs Engine)

**What**: Direct competition for Elo rating

**Requires**: Both CPU and GPU engines

```bash
./benchmark_matches --output matches.csv --games 50 --time 1000
```

**Time**: ~30-60 minutes for 50 games

## Command-Line Cheat Sheet

### Universal Options

```bash
--output FILE      # Where to save CSV (default: results_*.csv)
--verbose          # Detailed output
--help             # Show all options
```

### Throughput-Specific

```bash
--cpu-depth N      # CPU search depth (default: 15)
--gpu-sims N       # GPU simulations (default: 5000)
--time MS          # Max time per position (default: 5000)
--difficulty LEVEL # easy/medium/hard/all (default: all)
--cpu-only         # Test only CPU engine
--gpu-only         # Test only GPU engine
```

### Fixed-Time-Specific

```bash
--times MS,MS,...  # Time budgets (default: 50,100,500,1000,5000)
--suite NAME       # bratko-kopec/wac/performance/all
```

### Stockfish-Specific

```bash
--stockfish PATH   # Path to Stockfish binary
--stockfish-depth N # Stockfish depth (default: 20)
--engine-time MS   # Engine time limit (default: 5000)
```

### Matches-Specific

```bash
--games N          # Number of games (default: 100)
--time MS          # Time per move (default: 1000)
--max-moves N      # Max moves before draw (default: 200)
--no-alternate     # Don't swap colors
```

## Analyzing Results

### Quick View (Terminal)

```bash
# View CSV
cat results_throughput.csv | column -t -s,

# Count rows
wc -l results_throughput.csv

# Summary stats (if you have csvkit)
csvstat results_throughput.csv
```

### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results_throughput.csv')

# Summary statistics
print(df.groupby('engine')['throughput'].describe())

# Simple plot
df.boxplot(column='throughput', by='engine')
plt.ylabel('Throughput (ops/sec)')
plt.title('Engine Comparison')
plt.savefig('comparison.png')
plt.show()
```

### R Analysis

```r
library(tidyverse)

# Load results
df <- read_csv('results_throughput.csv')

# Summary
df %>% 
  group_by(engine) %>% 
  summarize(
    mean_throughput = mean(throughput),
    sd_throughput = sd(throughput)
  )

# Plot
ggplot(df, aes(x=engine, y=throughput)) +
  geom_boxplot() +
  theme_minimal()
```

## What to Try Next

After your first successful run:

1. **Run all benchmarks**:
   ```bash
   cd tests/scripts
   chmod +x run_all_benchmarks.sh
   ./run_all_benchmarks.sh my_results
   ```

2. **Try Google Colab**:
   - Open COLAB_TUTORIAL.md
   - Follow step-by-step guide
   - Get free GPU access

3. **Customize test positions**:
   - Edit `include/test_positions.h`
   - Add your own positions
   - Rebuild and run

4. **Tune engine parameters**:
   - Adjust depth limits
   - Change time budgets
   - Compare configurations

5. **Generate publication plots**:
   - See README.md for Python/R examples
   - Create camera-ready figures
   - Export to PDF/PNG

## Getting Help

**Problem**: Build errors
**Solution**: Check ARCHITECTURE.md → Troubleshooting section

**Problem**: Runtime crashes
**Solution**: Try verbose mode (`--verbose`) for details

**Problem**: Weird results
**Solution**: Verify engines work standalone first

**Problem**: Google Colab issues
**Solution**: See COLAB_TUTORIAL.md → Common Issues section

**Problem**: Something else
**Solution**: Check README.md or open GitHub issue

## Success Indicators

You know it's working when:

✅ CMake finds all dependencies
✅ `make` completes without errors
✅ Executables exist in `build/` directory
✅ First benchmark runs and produces CSV
✅ CSV file has expected columns
✅ No "Failed" or "Error" messages
✅ Results look reasonable (throughput > 0, etc.)

## Next Steps

- [ ] Read [README.md](README.md) for detailed documentation
- [ ] Try [COLAB_TUTORIAL.md](COLAB_TUTORIAL.md) for cloud testing
- [ ] Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
- [ ] Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) for complete overview

---

**Happy benchmarking!** 🚀♟️

Questions? Open an issue on GitHub or check the documentation.
