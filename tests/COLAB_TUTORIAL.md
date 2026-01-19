# Running Chess Engine Benchmarks in Google Colab

Complete step-by-step guide for running CPU and GPU chess engine benchmarks in Google Colab.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Colab Setup](#colab-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Getting the Code](#getting-the-code)
5. [Compiling the Engines](#compiling-the-engines)
6. [Running Benchmarks](#running-benchmarks)
7. [Visualizing Results](#visualizing-results)
8. [Downloading Results](#downloading-results)
9. [Common Issues & Solutions](#common-issues--solutions)
10. [GPU-Specific Considerations](#gpu-specific-considerations)

---

## Prerequisites

- Google account
- Basic familiarity with Jupyter notebooks
- Optional: GitHub repository with your chess engine code

## Colab Setup

### Step 1: Create New Notebook

1. Go to [https://colab.research.google.com](https://colab.research.google.com)
2. Click "File" → "New notebook"
3. **Important**: Enable GPU runtime
   - Click "Runtime" → "Change runtime type"
   - Set "Hardware accelerator" to "GPU"
   - Choose "T4 GPU" or "A100 GPU" (if available)
   - Click "Save"

### Step 2: Verify GPU Availability

Run this cell to verify GPU is active:

```python
!nvidia-smi
```

**Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.x   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:04.0 Off |                    0 |
...
```

**If you see "NVIDIA-SMI has failed":**
- Runtime is not set to GPU
- Go back to "Runtime" → "Change runtime type" → Select GPU
- Restart runtime

### Step 3: Check CUDA Toolkit

```python
!nvcc --version
```

**Expected output:**
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.x, ...
```

---

## Installing Dependencies

### Step 4: Install Build Tools

Run each cell separately to avoid timeout:

```python
# Update package lists
!apt-get update -qq
```

```python
# Install CMake (need 3.18+)
!apt-get install -qq cmake
!cmake --version
```

```python
# Install build essentials (GCC, G++, Make)
!apt-get install -qq build-essential
!g++ --version
```

```python
# Install OpenMP development files
!apt-get install -qq libomp-dev
```

**Verify installation:**

```python
# Test OpenMP
import subprocess
result = subprocess.run(['bash', '-c', 'echo "#include <omp.h>" | g++ -fopenmp -x c++ - -o /dev/null'], 
                       capture_output=True)
print("OpenMP:", "✓ Available" if result.returncode == 0 else "✗ Failed")
```

---

## Getting the Code

### Option A: Clone from GitHub

```python
# Clone repository (replace with your repo URL)
!git clone https://github.com/yourusername/cuda-mc-chess-engine.git
%cd cuda-mc-chess-engine
```

### Option B: Upload Files

If you don't have a GitHub repo:

```python
from google.colab import files
import zipfile
import os

# Upload your project as a ZIP file
print("Please upload your project ZIP file:")
uploaded = files.upload()

# Extract
zip_name = list(uploaded.keys())[0]
with zipfile.ZipFile(zip_name, 'r') as zip_ref:
    zip_ref.extractall('.')

# Navigate to project directory
project_dir = zip_name.replace('.zip', '')
%cd {project_dir}
```

### Verify Project Structure

```python
# List project structure
!ls -la
!ls cpu/
!ls gpu/
!ls tests/
```

**Expected output:**
```
total XX
drwxr-xr-x  5 root root   cpu/
drwxr-xr-x  5 root root   gpu/
drwxr-xr-x  3 root root   tests/
-rw-r--r--  1 root root   README.md
...
```

---

## Compiling the Engines

### Step 5: Build CPU Engine (Standalone Test)

```python
# The CPU engine uses Meson build system
# Install Meson if not available
!pip install meson ninja

# Configure CPU engine build
!meson setup cpu/build cpu --buildtype=release

# Build CPU tests
!meson compile -C cpu/build

# Test CPU engine
!./cpu/build/test_suite_parallel --mode=depth --depth=10 --level=easy | head -20
```

**Alternative: Manual compilation (if Meson fails):**

```python
# Compile directly with proper paths
!g++ -O3 -march=native -fopenmp \
    -I cpu/include \
    -o cpu_test \
    cpu/src/chess_engine_parallelized.cpp \
    cpu/tests/test_suite.cpp \
    -DUNIT_TESTS

# Test
!./cpu_test --mode=depth --depth=10 --level=easy | head -20
```

### Step 6: Build GPU Engine (Standalone Test)

```python
# Navigate to GPU directory
%cd gpu

# Check for build script
!ls build*.sh build*.bat 2>/dev/null || echo "Using manual build"

# If build script exists, use it:
!chmod +x build_and_test.sh 2>/dev/null
!bash build_and_test.sh 2>/dev/null || echo "Build script not available, using manual build"

# Manual build (if script doesn't exist):
!mkdir -p build
%cd build

# Detect GPU architecture
import subprocess
gpu_cap = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                        capture_output=True, text=True)
arch = gpu_cap.stdout.strip().replace('.', '') if gpu_cap.returncode == 0 else "75"
print(f"Using CUDA architecture: {arch}")

# Configure with CMake
!cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="{arch}" \
    -DCMAKE_BUILD_TYPE=Release

# Compile (this may take 3-5 minutes)
!make -j$(nproc)

# Test GPU engine
%cd ../..
!./gpu/build/test_puct_mcts | head -30
```

**Important GPU Notes:**
- CUDA architecture must match your GPU
- T4 GPU → use `75`
- A100 GPU → use `80`
- Check with: `nvidia-smi --query-gpu=compute_cap --format=csv`

```python
# Detect GPU compute capability
gpu_info = !nvidia-smi --query-gpu=compute_cap --format=csv,noheader
if gpu_info:
    cap = gpu_info[0].strip()
    print(f"GPU Compute Capability: {cap}")
    arch = cap.replace('.', '')
    print(f"Use CMAKE_CUDA_ARCHITECTURES={arch}")
```

### Step 7: Build Benchmark Framework

```python
# Navigate to tests directory
%cd /content/cuda-mc-chess-engine/tests
!mkdir -p build
%cd build

# Auto-detect GPU architecture
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                       capture_output=True, text=True)
if result.returncode == 0:
    arch = result.stdout.strip().replace('.', '')
    print(f"Detected GPU architecture: {arch}")
else:
    arch = "75"  # Default to T4
    print(f"Using default architecture: {arch}")

# Configure benchmarks with GPU support
cmake_cmd = f"""cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES="{arch}" \
    -DBUILD_GPU_BENCHMARKS=ON \
    -DBUILD_CPU_BENCHMARKS=ON \
    -DCMAKE_BUILD_TYPE=Release"""

print(f"Running: {cmake_cmd}")
!{cmake_cmd}

# Compile (this may take 5-10 minutes)
# Use VERBOSE=1 to see detailed compilation steps
!make -j$(nproc) VERBOSE=1
```

**Expected output:**
```
[ 10%] Building CXX object CMakeFiles/benchmark_throughput_gpu.dir/src/benchmark_throughput.cpp.o
[ 20%] Building CXX object CMakeFiles/benchmark_throughput_gpu.dir/src/cpu_engine_wrapper.cpp.o
[ 30%] Building CXX object CMakeFiles/benchmark_throughput_gpu.dir/src/gpu_engine_wrapper.cpp.o
[ 40%] Building CUDA object CMakeFiles/benchmark_throughput_gpu.dir/__/gpu/src/init_tables.cu.o
...
[100%] Linking CXX executable benchmark_throughput_gpu
[100%] Built target benchmark_throughput_gpu
```

### Verify Binaries

```python
# List built executables
!ls -lh benchmark_* 2>/dev/null || echo "No benchmarks built"

# Check which ones include GPU support
!ldd benchmark_throughput_gpu 2>/dev/null | grep cuda || echo "GPU binary not found"
```

---

## Running Benchmarks

### Step 8: Throughput Benchmark

```python
%cd /content/cuda-mc-chess-engine/tests/build

# Create output directory
!mkdir -p results

# Run throughput benchmark
!./benchmark_throughput_gpu \
    --output results/throughput.csv \
    --difficulty easy \
    --time 3000 \
    --cpu-depth 12 \
    --gpu-sims 3000 \
    --verbose
```

**Monitor progress:**
```python
import time
import subprocess
import sys

# Run with real-time output
proc = subprocess.Popen(
    ['./benchmark_throughput_gpu', '--output', 'results/throughput.csv', 
     '--difficulty', 'easy', '--time', '3000'],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    universal_newlines=True
)

for line in proc.stdout:
    print(line, end='')
    sys.stdout.flush()

proc.wait()
print(f"\nExit code: {proc.returncode}")
```

### Step 9: Fixed-Time Quality Benchmark

```python
# Run fixed-time benchmark with multiple time budgets
!./benchmark_fixed_time_gpu \
    --output results/fixed_time.csv \
    --times 50,100,500,1000 \
    --suite bratko-kopec \
    --verbose
```

### Step 10: Stockfish Agreement (Optional)

**First, install Stockfish:**

```python
# Install Stockfish
!apt-get install -qq stockfish

# Verify installation
!stockfish --version || echo "Stockfish not installed"
```

**Then run benchmark:**

```python
!./benchmark_stockfish_gpu \
    --output results/stockfish_agreement.csv \
    --stockfish /usr/games/stockfish \
    --suite bratko-kopec \
    --stockfish-depth 15 \
    --engine-time 3000 \
    --verbose
```

### Step 11: Head-to-Head Matches

```python
# Run matches (this can take a while!)
!./benchmark_matches_gpu \
    --output results/matches.csv \
    --games 20 \
    --time 1000 \
    --max-moves 150 \
    --verbose
```

**For longer runs, use smaller parameters:**
```python
# Quick test: 10 games, 500ms per move
!./benchmark_matches_gpu \
    --games 10 \
    --time 500 \
    --max-moves 100 \
    --output results/quick_matches.csv
```

---

## Visualizing Results

### Step 12: Load and Visualize CSV Results

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
```

#### Throughput Analysis

```python
# Load throughput results
df_throughput = pd.read_csv('results/throughput.csv')

print("Throughput Statistics:")
print(df_throughput.groupby('engine')['throughput'].describe())

# Plot throughput comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Box plot
df_throughput.boxplot(column='throughput', by='engine', ax=axes[0])
axes[0].set_ylabel('Throughput (operations/sec)')
axes[0].set_title('Throughput Distribution by Engine')
axes[0].set_xlabel('Engine')

# Bar plot - mean with std error
throughput_stats = df_throughput.groupby('engine')['throughput'].agg(['mean', 'std'])
throughput_stats.plot(kind='bar', y='mean', yerr='std', ax=axes[1], legend=False)
axes[1].set_ylabel('Mean Throughput (ops/sec)')
axes[1].set_title('Average Throughput with Std Dev')
axes[1].set_xlabel('Engine')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results/throughput_comparison.png', dpi=150)
plt.show()
```

#### Fixed-Time Quality Analysis

```python
# Load fixed-time results
df_quality = pd.read_csv('results/fixed_time.csv')

# Calculate accuracy if best moves are known
# (Assuming test_positions.h has best_move field populated)
if 'best_move' in df_quality.columns:
    df_quality['correct'] = df_quality['move_uci'] == df_quality['best_move']
    
    # Accuracy by time budget
    accuracy = df_quality.groupby(['engine', 'time_budget_ms'])['correct'].mean().reset_index()
    accuracy['accuracy'] = accuracy['correct'] * 100
    
    plt.figure(figsize=(12, 6))
    for engine in accuracy['engine'].unique():
        engine_data = accuracy[accuracy['engine'] == engine]
        plt.plot(engine_data['time_budget_ms'], engine_data['accuracy'], 
                marker='o', label=engine, linewidth=2)
    
    plt.xscale('log')
    plt::xlabel('Time Budget (ms)')
    plt.ylabel('Accuracy (%)')
    plt.title('Move Accuracy vs Time Budget')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/accuracy_vs_time.png', dpi=150)
    plt.show()

# Depth reached analysis
plt.figure(figsize=(12, 6))
depth_stats = df_quality.groupby(['engine', 'time_budget_ms'])['depth'].mean().reset_index()

for engine in depth_stats['engine'].unique():
    engine_data = depth_stats[depth_stats['engine'] == engine]
    plt.plot(engine_data['time_budget_ms'], engine_data['depth'],
            marker='s', label=engine, linewidth=2)

plt.xscale('log')
plt.xlabel('Time Budget (ms)')
plt.ylabel('Average Search Depth')
plt.title('Search Depth vs Time Budget')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/depth_vs_time.png', dpi=150)
plt.show()
```

#### Match Results Analysis

```python
# Load match results
df_matches = pd.read_csv('results/matches.csv')

# Count results
results = df_matches['result'].value_counts()
print("\nMatch Results:")
print(results)

# Determine wins for each engine
white_wins = len(df_matches[df_matches['result'] == '1-0'])
black_wins = len(df_matches[df_matches['result'] == '0-1'])
draws = len(df_matches[df_matches['result'].str.contains('1/2')])

# Separate by engine (assuming alternating colors or specific naming)
cpu_name = df_matches['white_engine'].iloc[0]  # Get first white engine
gpu_name = df_matches['black_engine'].iloc[0]

cpu_wins = len(df_matches[
    ((df_matches['white_engine'] == cpu_name) & (df_matches['result'] == '1-0')) |
    ((df_matches['black_engine'] == cpu_name) & (df_matches['result'] == '0-1'))
])

gpu_wins = len(df_matches[
    ((df_matches['white_engine'] == gpu_name) & (df_matches['result'] == '1-0')) |
    ((df_matches['black_engine'] == gpu_name) & (df_matches['result'] == '0-1'))
])

# Pie chart
fig, ax = plt.subplots(figsize=(8, 8))
sizes = [cpu_wins, gpu_wins, draws]
labels = [f'{cpu_name} Wins', f'{gpu_name} Wins', 'Draws']
colors = ['#ff9999', '#66b3ff', '#99ff99']
explode = (0.05, 0.05, 0)

ax.pie(sizes, explode=explode, labels=labels, colors=colors,
       autopct='%1.1f%%', shadow=True, startangle=90)
ax.set_title('Match Results Distribution')
plt.savefig('results/match_results_pie.png', dpi=150)
plt.show()

# Calculate Elo
total = cpu_wins + gpu_wins + draws
score = (cpu_wins + 0.5 * draws) / total
elo_diff = -400 * np.log10(1 / max(score, 0.01) - 1) if score > 0 else 0

print(f"\n{cpu_name} Score: {score:.1%}")
print(f"Estimated Elo Difference: {elo_diff:+.0f}")
print(f"({cpu_name} is {abs(elo_diff):.0f} Elo {'higher' if elo_diff > 0 else 'lower'})")
```

---

## Downloading Results

### Step 13: Download CSV Files

```python
from google.colab import files
import os
import zipfile

# Option 1: Download individual CSVs
files.download('results/throughput.csv')
files.download('results/fixed_time.csv')
files.download('results/matches.csv')
```

```python
# Option 2: Download all results as ZIP
def create_results_zip():
    zip_name = 'benchmark_results.zip'
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        for root, dirs, files_list in os.walk('results'):
            for file in files_list:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
    return zip_name

zip_path = create_results_zip()
files.download(zip_path)
print(f"Downloaded: {zip_path}")
```

### Step 14: Download Visualizations

```python
# Download all PNG plots
for plot_file in ['throughput_comparison.png', 'accuracy_vs_time.png', 
                  'depth_vs_time.png', 'match_results_pie.png']:
    plot_path = f'results/{plot_file}'
    if os.path.exists(plot_path):
        files.download(plot_path)
```

---

## Common Issues & Solutions

### Issue 1: "CUDA out of memory"

**Symptoms:** Benchmark crashes with OOM error

**Solutions:**
```python
# Reduce batch size in GPU engine
# Edit gpu/include/search_config.h or pass smaller parameters
!./benchmark_throughput_gpu --gpu-sims 1000  # Reduce from 5000
```

### Issue 2: "CMake version too old"

**Solution:**
```python
# Install newer CMake from pip
!pip install cmake --upgrade
!cmake --version
```

### Issue 3: Compilation fails with "unsupported GPU architecture"

**Solution:**
```python
# Find your GPU architecture
!nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

# Rebuild with correct architecture (example for T4 = 75)
!cmake .. -DCMAKE_CUDA_ARCHITECTURES="75"
!make clean && make -j$(nproc)
```

### Issue 4: OpenMP not found

**Solution:**
```python
# Reinstall OpenMP
!apt-get install --reinstall libomp-dev gcc g++

# Verify
!echo | g++ -fopenmp -x c++ -c - || echo "OpenMP failed"
```

### Issue 5: Colab disconnects during long benchmarks

**Solutions:**
```python
# 1. Run in chunks
!./benchmark_throughput_gpu --difficulty easy  # Run easy first
!./benchmark_throughput_gpu --difficulty medium  # Then medium

# 2. Keep Colab alive with JavaScript console (F12):
function ClickConnect() {
    console.log("Keeping Colab alive");
    document.querySelector("colab-toolbar-button#connect").click();
}
setInterval(ClickConnect, 60000);  # Click every 60 seconds

# 3. Use Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')
# Then save results to /content/drive/MyDrive/results/
```

---

## GPU-Specific Considerations

### Verifying GPU is Actually Used

```python
# Monitor GPU usage during benchmark
import subprocess
import threading
import time

def monitor_gpu(duration=30):
    """Monitor GPU utilization for specified duration"""
    end_time = time.time() + duration
    while time.time() < end_time:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True)
        util, mem = result.stdout.strip().split(', ')
        print(f"\rGPU Util: {util}% | Memory: {mem} MB", end='', flush=True)
        time.sleep(1)
    print()

# Start monitor in background
monitor_thread = threading.Thread(target=monitor_gpu, args=(30,))
monitor_thread.start()

# Run benchmark
!./benchmark_throughput_gpu --gpu-only --time 3000

monitor_thread.join()
```

### GPU Performance Tuning

```python
# Check GPU clock speeds
!nvidia-smi -q -d CLOCK | grep -A 5 "Max Clocks"

# Set persistence mode (helps with initialization latency)
!nvidia-smi -pm 1

# Query GPU details
!nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,compute_cap --format=csv
```

### Optimal Batch Sizes for Different GPUs

```python
# T4 (16GB): Use batch_size=256, simulations=5000
# A100 (40GB): Use batch_size=512, simulations=10000
# V100 (16GB): Use batch_size=256, simulations=5000

gpu_name = !nvidia-smi --query-gpu=name --format=csv,noheader
mem_total = !nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits

print(f"GPU: {gpu_name[0]}")
print(f"Memory: {mem_total[0]} MB")

# Adjust parameters accordingly
if int(mem_total[0]) > 30000:  # >30GB
    print("Recommended: --gpu-sims 10000")
else:
    print("Recommended: --gpu-sims 5000")
```

---

## Complete Example Workflow

Here's a complete notebook cell sequence:

```python
# === CELL 1: Setup ===
!nvidia-smi
!nvcc --version
!apt-get update -qq && apt-get install -qq cmake build-essential libomp-dev

# === CELL 2: Get Code ===
!git clone https://github.com/yourusername/cuda-mc-chess-engine.git
%cd cuda-mc-chess-engine

# === CELL 3: Build ===
%cd tests
!mkdir -p build && cd build
!cmake .. -DCMAKE_CUDA_ARCHITECTURES="75" -DBUILD_GPU_BENCHMARKS=ON
!make -j$(nproc)

# === CELL 4: Run Benchmarks ===
!mkdir -p results
!./benchmark_throughput_gpu --output results/throughput.csv --time 3000
!./benchmark_fixed_time_gpu --output results/fixed_time.csv --times 100,500,1000
!./benchmark_matches_gpu --output results/matches.csv --games 20 --time 1000

# === CELL 5: Visualize ===
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/throughput.csv')
df.groupby('engine')['throughput'].describe()
df.boxplot(column='throughput', by='engine')
plt.savefig('results/throughput.png')

# === CELL 6: Download ===
from google.colab import files
files.download('results/throughput.csv')
files.download('results/throughput.png')
```

---

## Tips for Research/Publication

1. **Reproducibility:**
   - Note Colab GPU type in your paper
   - Save full benchmark output logs
   - Record CUDA and driver versions

```python
# Save environment info
with open('results/environment.txt', 'w') as f:
    import subprocess
    f.write("=== GPU Info ===\n")
    f.write(subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout)
    f.write("\n=== CUDA Version ===\n")
    f.write(subprocess.run(['nvcc', '--version'], capture_output=True, text=True).stdout)
    f.write("\n=== GCC Version ===\n")
    f.write(subprocess.run(['g++', '--version'], capture_output=True, text=True).stdout)
```

2. **Statistical Significance:**
   - Run multiple trials (change random seeds)
   - Report confidence intervals
   - Use appropriate sample sizes

3. **Fair Comparisons:**
   - Same hardware for all engines
   - Same time budgets
   - Document all hyperparameters

---

## Next Steps

After running benchmarks successfully:

1. **Analyze trade-offs**: Throughput vs. quality, time vs. accuracy
2. **Tune engines**: Adjust search parameters based on results
3. **Extend tests**: Add custom positions, opening books
4. **Automate**: Create Python scripts for batch runs
5. **Publish**: Share results, compare with literature

Happy benchmarking! 🚀♟️
