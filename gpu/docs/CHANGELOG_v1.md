# Changelog v1.0 - GPU MCTS Chess Engine Improvements

## Summary

This release implements all High-Priority (P1) and Medium-Priority (P2) improvements from the improvement roadmap. All changes are marked with `//N - v1` comments in the source code.

---

## New Files

| File | Description |
|------|-------------|
| `monte_carlo_advanced_kernel_v1.cuh` | New header with compact structures and MCTS support |
| `monte_carlo_advanced_kernel_v1.cu` | Implementation with all P1/P2 improvements |
| `monte_carlo_advanced_v1.hpp` | Updated C++ interface header |
| `monte_carlo_advanced_v1.cpp` | C++ wrapper with evaluation mode selection |
| `main_advanced_v1.cpp` | Test driver with benchmarking support |
| `docs/REUSABILITY_ANALYSIS.md` | Analysis of CPU component reuse |
| `docs/CHANGELOG_v1.md` | This file |

---

## P1 (High-Priority) Improvements

### P1-5: Parallel Move Evaluation with CUDA Streams

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`, `monte_carlo_advanced_v1.cpp`

**Changes**:
- Added `monte_carlo_simulate_batch_kernel()` - single kernel evaluates ALL moves
- Grid layout: `dim3(blocks_per_move, num_moves)` - 2D grid for parallelism
- Added CUDA streams support in `evaluate_all_moves_streams()`
- Three evaluation modes: `LEGACY`, `BATCHED`, `STREAMS`

**Expected Performance Impact**: 5-10x speedup

```cpp
// New batched kernel launch
dim3 grid(blocks_per_move, num_moves);
monte_carlo_simulate_batch_kernel<<<grid, block>>>(...);
```

---

### P1-6: Transposition Detection in Playouts

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`, `monte_carlo_advanced_kernel_v1.cuh`

**Changes**:
- Implemented Zobrist hashing with constants in device memory
- Added `GPUTTEntry` structure (16 bytes per entry)
- TT size: 1M entries (~16MB)
- Functions: `compute_hash()`, `update_hash()`, `tt_probe()`, `tt_store()`
- Repetition detection during playouts

**Expected Performance Impact**: 10-30% quality improvement

```cpp
struct GPUTTEntry {
    uint64_t key;       // Zobrist hash
    int16_t score;      // Evaluation
    uint8_t depth;
    uint8_t flag;
    uint8_t best_from;
    uint8_t best_to;
    uint16_t generation;
};  // 16 bytes
```

---

### P1-7: Batched Simulation Results with Reduction

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`

**Changes**:
- Added `warp_reduce_sum()` using `__shfl_down_sync`
- Added `reduce_simulation_results()` kernel with shared memory
- Results accumulated with `atomicAdd()` in batched kernel
- Eliminated host-side reduction loop

**Expected Performance Impact**: 20-50% reduction in memory traffic

```cpp
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}
```

---

### P1-8: Reduce Position Struct Size

**Files Modified**: `monte_carlo_advanced_kernel_v1.cuh`

**Changes**:
- New `CompactPosition` struct: 72 bytes (was 280 bytes)
- New `CompactMove` struct: 8 bytes (was 24 bytes)
- Accessor macros: `POS_SIDE_TO_MOVE()`, `POS_CASTLING()`, etc.
- Legacy structs retained for host compatibility

**Expected Performance Impact**: 30-50% reduction in register pressure

```cpp
struct CompactPosition {
    int8_t board[64];       // 64 bytes
    uint8_t flags;          // 1 byte
    int8_t en_passant;      // 1 byte
    uint8_t halfmove_clock; // 1 byte
    uint8_t fullmove_low;   // 1 byte
    uint32_t padding;       // 4 bytes
};  // Total: 72 bytes
```

---

## P2 (Medium-Priority) Improvements

### P2-9: True MCTS with UCB Tree Policy

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`, `monte_carlo_advanced_kernel_v1.cuh`

**Changes**:
- Added `MCTSNode` structure (32 bytes)
- Implemented `calculate_ucb()` for selection
- Added `mcts_tree_kernel()` with:
  - **Selection**: Walk tree using UCB scores
  - **Expansion**: Add child nodes atomically
  - **Simulation**: Run playout from leaf
  - **Backpropagation**: Update values up the tree
- Configurable exploration constant: `MCTS_EXPLORATION_CONSTANT = 1.414`

**Expected Performance Impact**: Major strength improvement

```cpp
struct MCTSNode {
    int32_t parent, first_child, num_children;
    int32_t visits;
    float total_value;
    int32_t virtual_loss;
    uint8_t move_from, move_to, move_piece, move_promotion;
};  // 32 bytes
```

---

### P2-10: Virtual Loss for Parallel MCTS

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`

**Changes**:
- Added `virtual_loss` field to `MCTSNode`
- Atomic increment on selection: `atomicAdd(&node->virtual_loss, 1)`
- Atomic decrement after backpropagation: `atomicSub(&node->virtual_loss, 1)`
- UCB calculation accounts for virtual losses

**Expected Performance Impact**: Better parallelization efficiency

```cpp
// In selection phase
atomicAdd(&tree_nodes[current_node].virtual_loss, MCTS_VIRTUAL_LOSS_VALUE);

// After backpropagation
atomicSub(&tree_nodes[node_idx].virtual_loss, MCTS_VIRTUAL_LOSS_VALUE);
```

---

### P2-11: Quiescence Extension for Captures

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`

**Changes**:
- Added `generate_captures()` for capture-only move generation
- Implemented `quiescence_extension()` with alpha-beta bounds
- Called at end of playouts for stable evaluation
- Extension depth: `MCTS_QUIESCENCE_EXTENSION = 8`

**Expected Performance Impact**: 20-40% improvement in tactical positions

```cpp
__device__ int quiescence_extension(Position pos, int alpha, int beta,
                                    int depth, curandState* rand_state) {
    // Stand pat evaluation
    // Generate and evaluate captures only
    // Recursive search with alpha-beta pruning
}
```

---

### P2-12: Move History Learning

**Files Modified**: `monte_carlo_advanced_kernel_v1.cu`, `monte_carlo_advanced_kernel_v1.cuh`

**Changes**:
- Added `d_history_table[2][64][64]` in device memory
- Implemented `history_update()` with atomic operations
- History scores integrated into `score_moves()`
- Clamping to prevent overflow: `[-32768, 32767]`

**Expected Performance Impact**: 15-25% improvement in quiet positions

```cpp
__device__ void history_update(int color, int from, int to, int bonus) {
    bonus = max(-2000, min(2000, bonus));
    int new_val = max(HISTORY_MIN, min(HISTORY_MAX, old_val + bonus));
    atomicExch(&d_history_table[color][from][to], new_val);
}
```

---

## Resource Management

### Initialization
```cpp
initialize_gpu_resources();  // Called automatically on first use
```

Allocates:
- Transposition table: ~16MB
- MCTS nodes: ~2MB
- Zobrist keys in constant memory

### Cleanup
```cpp
monte_carlo_advanced_v1::shutdown();  // Call at program end
monte_carlo_advanced_v1::new_game();  // Clear caches between games
```

---

## API Changes

### New Evaluation Modes
```cpp
enum class EvaluationMode {
    LEGACY,     // Original sequential (backward compatible)
    BATCHED,    // Recommended - all moves in one kernel
    STREAMS     // Async with multiple CUDA streams
};
```

### Updated Function Signatures
```cpp
chess::Move find_best_move(
    const chess::Board& board,
    int simulations_per_move = 10000,
    int threads_per_move = 256,
    bool verbose = true,
    EvaluationMode mode = EvaluationMode::BATCHED  // NEW
);
```

---

## Building

### Compile v1 Version
```bash
nvcc -O3 -arch=sm_75 -c monte_carlo_advanced_kernel_v1.cu -o monte_carlo_advanced_kernel_v1.o
g++ -O3 -std=c++17 -I.. -c monte_carlo_advanced_v1.cpp -o monte_carlo_advanced_v1.o
g++ -O3 -std=c++17 -I.. -c main_advanced_v1.cpp -o main_advanced_v1.o
g++ monte_carlo_advanced_kernel_v1.o monte_carlo_advanced_v1.o main_advanced_v1.o \
    -L/usr/local/cuda/lib64 -lcudart -lcurand -o monte_carlo_v1
```

### Run with Benchmarks
```bash
./monte_carlo_v1 --benchmark --sims 50000
./monte_carlo_v1 --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
```

---

## Testing Recommendations

### Correctness Testing
1. Compare move selection on tactical puzzles against known solutions
2. Verify mate-in-N detection on standard positions
3. Check repetition detection with repetitive test positions
4. Validate history table updates with debug output

### Performance Testing
1. Run `--benchmark` to compare evaluation modes
2. Measure simulations/second with varying workloads
3. Profile GPU occupancy with `nvprof` or Nsight
4. Compare against original implementation on same positions

### Regression Testing
1. Run original test suite with `EvaluationMode::LEGACY`
2. Verify identical move selection at same random seed
3. Check memory usage doesn't exceed GPU limits

---

## Known Limitations

1. **Castling/En Passant**: Still not implemented in move generator (P0 items)
2. **MCTS Tree**: Limited to 64K nodes per kernel launch
3. **Streams Mode**: Currently uses synchronous fallback pending kernel refactor
4. **Memory**: Requires ~20MB GPU memory for TT and MCTS nodes

---

## Performance Expectations

| Configuration | Simulations/sec | vs Original |
|--------------|-----------------|-------------|
| Original (Legacy) | ~50K | 1.0x |
| Batched Mode | ~200-400K | 4-8x |
| With TT + History | +20-30% quality | - |
| Full MCTS Tree | Better strength | - |

---

## Future Work (P3/P4)

- [ ] Implement castling and en passant (P0 - critical)
- [ ] Port full bitboard-based move generation
- [ ] Add shared memory optimization for move arrays
- [ ] Implement proper CUDA streams with async kernels
- [ ] Add neural network evaluation integration
