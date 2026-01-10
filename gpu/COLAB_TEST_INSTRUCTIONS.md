# PUCT MCTS Chess Engine - Google Colab Test Instructions

## 📋 Prerequisites

Your Colab needs:
- NVIDIA GPU (T4, V100, A100)
- CUDA Toolkit
- Git access to your repository

## 🚀 Step-by-Step Colab Setup

### 1. Check GPU Availability

```python
!nvidia-smi
```

Expected output: Shows GPU name (T4, V100, etc.)

### 2. Clone Repository

```python
!git clone https://YOUR_GITHUB_TOKEN@github.com/YOUR_USERNAME/cuda-mc-chess-engine.git
%cd cuda-mc-chess-engine
```

### 3. Checkout Latest Branch

```python
!git fetch
!git checkout monte-carlo-v3-sejtanluci
!git pull origin monte-carlo-v3-sejtanluci
```

### 4. Navigate to GPU Directory

```python
%cd gpu
!ls -la
```

You should see:
- `include/` folder
- `src/` folder
- `tests/` folder
- `Makefile`

### 5. Compile PUCT Test Suite

```python
# Compile test_puct_mcts.cpp
!nvcc -std=c++17 -arch=sm_75 -O3 -Iinclude \
  tests/test_puct_mcts.cpp \
  src/gpu_kernels.cu \
  src/init_tables.cu \
  src/mcts.cpp \
  src/puct_mcts.cpp \
  -o test_puct_mcts \
  -lcudart -lcurand
```

**Note**: If you have T4 GPU, use `-arch=sm_75`. For V100, use `-arch=sm_70`. For A100, use `-arch=sm_80`.

### 6. Run PUCT Tests

```python
!./test_puct_mcts
```

Expected output:
```
========================================
  PUCT MCTS COMPREHENSIVE TEST SUITE
  Heuristic AlphaZero (NO Neural Nets)
========================================

Initializing GPU and attack tables...
✓ Initialization complete

[TEST 1] PUCT Engine Initialization
------------------------------------
✓ Engine initialized successfully
...

========================================
PUCT MCTS Test Summary
========================================
Total:  10
Passed: 10 (100.0%)
Failed: 0
Time:   X.XX seconds
========================================

🎉 ALL TESTS PASSED! Engine is working correctly.
```

### 7. Build Main Engine

```python
# Build complete PUCT chess engine
!make clean
!make
```

### 8. Run Engine Benchmark

```python
# PUCT MCTS benchmark (1600 simulations)
!./puct_chess --puct --benchmark --sims 1600
```

Expected output:
```
=== PUCT MCTS Engine (Heuristic AlphaZero-style) ===
NO Neural Networks - Pure tactical heuristics

Best move: e2e4
Total visits: 2456
Root value: 0.08
Time: 0.150 s
Sims/sec: 10666
PV: e2e4 e7e5 g1f3 b8c6 f1c4
```

### 9. Compare PUCT vs Original MCTS

```python
# PUCT (400 simulations)
!./puct_chess --puct --benchmark --sims 400

# Original UCB1 (400 simulations)
!./puct_chess --original --benchmark --sims 400
```

Compare speed and quality!

### 10. Run Self-Play Game

```python
# PUCT self-play (50 moves)
!./puct_chess --puct --play --sims 800 --moves 50
```

---

## 🔧 Troubleshooting

### Issue: "nvcc not found"

```python
# Install CUDA toolkit
!apt-get update
!apt-get install -y cuda-toolkit-11-8
```

### Issue: "Compute capability mismatch"

Check your GPU:
```python
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Compute capability: {torch.cuda.get_device_capability(0)}")
```

Update `-arch` flag:
- sm_70 for V100
- sm_75 for T4
- sm_80 for A100

### Issue: "Out of memory"

Reduce batch size:
```python
# In test file, reduce batch_size in configs
# Or pass smaller batch via command line
!./puct_chess --puct --benchmark --sims 400 --batch 128
```

---

## 📊 Expected Performance (T4 GPU)

| Test | Time | Simulations/sec |
|------|------|-----------------|
| Initialization | <1s | N/A |
| PUCT Formula | <1s | N/A |
| Starting Position (400 sims) | ~40ms | ~10,000 |
| Mate in 1 (800 sims) | ~80ms | ~10,000 |
| Batch Performance (512) | ~150ms | ~10,000 |

---

## 🎯 Test Coverage

The test suite verifies:

1. ✅ PUCT engine initialization
2. ✅ PUCT selection formula correctness
3. ✅ Virtual loss mechanism (thread safety)
4. ✅ Heuristic policy prior computation
5. ✅ Starting position search
6. ✅ PUCT vs Original MCTS comparison
7. ✅ Tactical solving (Mate in 1)
8. ✅ Dirichlet noise exploration
9. ✅ GPU batch evaluation performance
10. ✅ Move probability distribution

---

## 📝 Complete Colab Notebook Code

Copy-paste this into a new Colab notebook:

```python
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Clone and setup
!git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/cuda-mc-chess-engine.git
%cd cuda-mc-chess-engine
!git checkout monte-carlo-v3-sejtanluci
!git pull origin monte-carlo-v3-sejtanluci
%cd gpu

# Cell 3: Compile tests
!nvcc -std=c++17 -arch=sm_75 -O3 -Iinclude \
  tests/test_puct_mcts.cpp \
  src/gpu_kernels.cu \
  src/init_tables.cu \
  src/mcts.cpp \
  src/puct_mcts.cpp \
  -o test_puct_mcts \
  -lcudart -lcurand

# Cell 4: Run tests
!./test_puct_mcts

# Cell 5: Build main engine
!make clean
!make

# Cell 6: PUCT benchmark
!./puct_chess --puct --benchmark --sims 1600

# Cell 7: Comparison test
print("=== PUCT MCTS ===")
!./puct_chess --puct --benchmark --sims 400
print("\n=== Original MCTS ===")
!./puct_chess --original --benchmark --sims 400

# Cell 8: Self-play
!./puct_chess --puct --play --sims 800 --moves 30
```

---

## 🎓 What Each Test Verifies

### TEST 1: Initialization
- GPU memory allocation
- Attack table loading
- Configuration setup

### TEST 2: PUCT Formula
- Exploitation term (Q value)
- Exploration term (U value)
- Prior probability weighting

### TEST 3: Virtual Loss
- Atomic operations
- Thread-safe updates
- Parallel search correctness

### TEST 4: Heuristic Priors
- MVV-LVA scoring
- Tactical bonuses (checks, promotions)
- Killer moves and history heuristic

### TEST 5: Starting Position
- Full PUCT search
- Move selection
- Performance metrics

### TEST 6: PUCT vs Original
- Speed comparison
- Search quality
- Algorithmic differences

### TEST 7: Mate in 1
- Tactical strength
- Quiescence search
- Position evaluation

### TEST 8: Dirichlet Noise
- Root exploration
- Noise generation
- Prior mixing

### TEST 9: GPU Batching
- Batch size optimization
- Throughput measurement
- Scalability

### TEST 10: Move Probabilities
- Visit count distribution
- Temperature scaling
- Probability normalization

---

## 🔍 Debugging Tips

### View detailed test output:

```python
!./test_puct_mcts 2>&1 | tee test_output.log
!cat test_output.log
```

### Check for memory leaks:

```python
!nvidia-smi
# Run tests
!./test_puct_mcts
!nvidia-smi
# GPU memory should be freed
```

### Profile performance:

```python
!nvprof ./test_puct_mcts
```

---

## ✅ Success Criteria

Your implementation is correct if:

1. All 10 tests pass ✓
2. PUCT finds reasonable moves in starting position
3. PUCT solves Mate in 1 correctly
4. Simulations/sec > 5000 on T4 GPU
5. No CUDA errors or memory leaks

---

## 📞 Support

If tests fail:
1. Check GPU compatibility (`nvidia-smi`)
2. Verify CUDA version (`nvcc --version`)
3. Review compilation flags (arch, std, optimization)
4. Check branch is up to date (`git status`)

Happy testing! 🎉
