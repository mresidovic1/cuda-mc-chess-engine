# GPU MCTS Chess Engine - Improvement Roadmap

## Executive Summary

This document provides a prioritized improvement roadmap to make the GPU Monte Carlo implementation competitive with the CPU Lazy SMP minimax engine. The current GPU implementation has significant performance gaps due to missing features, algorithmic limitations, and suboptimal GPU utilization.

**Current State**: ~50K simulations/sec, weak tactical play, missing chess rules
**Target State**: >500K simulations/sec, competitive tactical strength with CPU at depth 10-12

---

## Critical Issues (Highest Priority)

These are bugs and correctness problems that must be fixed first.

---

### 1. Missing Castling Implementation

**Problem**:
The GPU move generator does not generate castling moves. This is a fundamental chess rule violation that causes the engine to miss critical defensive and developmental moves.

**Location**: `monte_carlo_advanced_kernel.cu:330-351` (`generate_all_moves`)

**Current Code**:
```cpp
__device__ int generate_all_moves(const Position& pos, Move* moves) {
    int move_count = 0;
    generate_pawn_moves(pos, moves, move_count);
    generate_knight_moves(pos, moves, move_count);
    // ... sliding pieces and king
    generate_king_moves(pos, moves, move_count);
    return move_count;
    // NO CASTLING GENERATION!
}
```

**Implementation Approach**:
1. Add castling rights tracking to Position struct (already present but unused)
2. Implement `generate_castling_moves()`:
   - Check kingside/queenside rights
   - Verify king and rook haven't moved (use rights flags)
   - Check that squares between king and rook are empty
   - Check that king doesn't pass through or end in check
3. Update `make_move()` to handle castling (move both king and rook)
4. Update castling rights after king/rook moves or captures

**Implementation Complexity**: Medium
**Expected Impact**: Correctness - Required for legal chess play

---

### 2. Missing En Passant Implementation

**Problem**:
En passant captures are not implemented. The `en_passant` field in Position exists but is always -1.

**Location**:
- `monte_carlo_advanced_kernel.cu:159-213` (`generate_pawn_moves`)
- `monte_carlo_advanced_kernel.cu:357-370` (`make_move`)

**Current Code**:
```cpp
pos.en_passant = -1;  // Always reset, never set
```

**Implementation Approach**:
1. In `make_move()`: When a pawn moves 2 squares, set `en_passant` to the skipped square
2. In `generate_pawn_moves()`: Add check for en passant capture:
   ```cpp
   if (pos.en_passant >= 0) {
       int ep_sq = pos.en_passant;
       // Check if current pawn can capture en passant
       if (can_capture_en_passant(from, ep_sq, color)) {
           moves[move_count++] = {from, ep_sq, 0, enemy_pawn, piece, 100.0f};
       }
   }
   ```
3. In `make_move()`: Handle en passant capture (remove the captured pawn on different square)

**Implementation Complexity**: Low-Medium
**Expected Impact**: Correctness - Required for legal chess play

---

### 3. King Legality Not Fully Checked

**Problem**:
The move generator doesn't filter out moves that leave the king in check. This can lead to illegal positions during playouts.

**Location**: `monte_carlo_advanced_kernel.cu:330-351`

**Current Code**:
The functions generate pseudo-legal moves, and the playout checks for game-over after moves, but illegal moves (leaving king in check) can still be played and evaluated.

**Implementation Approach**:
1. After generating all moves, filter using legality check:
   ```cpp
   __device__ bool is_legal_move(const Position& pos, const Move& move) {
       Position test = pos;
       make_move(test, move);
       int our_king_sq = find_king(test, pos.side_to_move);
       return !is_square_attacked(test, our_king_sq, 1 - pos.side_to_move);
   }
   ```
2. Or, more efficiently, only check king moves and moves that could expose the king

**Implementation Complexity**: Medium
**Expected Impact**: Correctness - Prevents invalid game states

---

### 4. Promotion Piece Type Mapping Error

**Problem**:
In the GPU kernel, promotion values are inconsistent. The move generator uses 5 for queen, but the piece encoding uses 5 for W_QUEEN.

**Location**:
- `monte_carlo_advanced_kernel.cu:177-181`
- `monte_carlo_advanced.cpp:69-75`

**Analysis**:
```cpp
// In kernel:
moves[move_count++] = {from, to, 5, EMPTY, piece, 150.0f}; // Queen

// In make_move:
pos.board[move.to] = (move.promotion > 0) ?
    ((pos.side_to_move == GPU_WHITE) ? move.promotion : (move.promotion + 8)) :
    move.piece;
```

The promotion value 5 directly becomes the piece type, which works for white (W_QUEEN=5), but the logic needs verification for all promotion types.

**Implementation Approach**:
1. Add explicit mapping in `make_move()`:
   ```cpp
   int promoted_piece;
   switch(move.promotion) {
       case 2: promoted_piece = W_KNIGHT; break;
       case 3: promoted_piece = W_BISHOP; break;
       case 4: promoted_piece = W_ROOK; break;
       case 5: promoted_piece = W_QUEEN; break;
   }
   if (pos.side_to_move == GPU_BLACK) promoted_piece += 8;
   ```

**Implementation Complexity**: Low
**Expected Impact**: Correctness - Proper piece promotion

---

## High-Priority Optimizations

Changes that will significantly close the performance gap with the CPU implementation.

---

### 5. Parallel Move Evaluation (Critical Performance Fix)

**Problem**:
Currently, each root move is evaluated sequentially with a separate kernel launch. This causes massive GPU idle time due to kernel launch overhead.

**Location**: `monte_carlo_advanced.cpp:102-166` (`evaluate_all_moves`)

**Current Code**:
```cpp
for (const auto& move : movelist) {
    // ... convert move ...
    launch_monte_carlo_simulate_kernel(...);  // BLOCKING CALL
    cudaMemcpy(h_results.data(), d_results, ...);  // BLOCKING COPY
    // ... calculate average ...
}
```

**Implementation Approach**:
Two options:

**Option A: CUDA Streams (Recommended)**
```cpp
cudaStream_t streams[MAX_STREAMS];
for (int i = 0; i < num_moves; i++) {
    int stream_idx = i % MAX_STREAMS;
    launch_kernel_async(streams[stream_idx], move[i], results[i]);
}
cudaDeviceSynchronize();
```

**Option B: Single Mega-Kernel**
Launch one kernel that evaluates all moves:
```cpp
__global__ void evaluate_all_moves_kernel(
    Position root,
    Move* all_moves,
    int num_moves,
    float* results
) {
    int move_idx = blockIdx.y;
    int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each block-row handles one move
}
```

**Implementation Complexity**: Medium-High
**Expected Performance Impact**: 5-10x speedup (reduces kernel launch overhead from O(n) to O(1))

---

### 6. Add Transposition Detection in Playouts

**Problem**:
The same positions are re-evaluated multiple times during playouts with no caching.

**Location**: `monte_carlo_advanced_kernel.cu:559-626` (`monte_carlo_playout`)

**Implementation Approach**:
1. Use a simple position hash (Zobrist-style) computed incrementally
2. Store hash in thread-local small table (16-32 entries)
3. On position repeat → return draw (0 score)

**Device Code**:
```cpp
__device__ uint64_t compute_hash(const Position& pos) {
    uint64_t hash = 0;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] != EMPTY) {
            hash ^= zobrist_table[sq][pos.board[sq]];
        }
    }
    hash ^= pos.side_to_move ? side_hash : 0;
    return hash;
}
```

**Implementation Complexity**: Medium
**Expected Performance Impact**: 10-30% quality improvement (draws detected correctly)

---

### 7. Batched Simulation Results with Reduction

**Problem**:
Each thread writes to global memory, then CPU sums all results. This is inefficient.

**Location**: `monte_carlo_advanced.cpp:151-160`

**Current Code**:
```cpp
float total_score = 0.0f;
for (int i = 0; i < total_threads; i++) {
    total_score += h_results[i];
}
```

**Implementation Approach**:
1. Use warp-level reduction in kernel
2. Use block-level reduction with shared memory
3. Only write one result per block to global memory

**Device Code**:
```cpp
__shared__ float shared_scores[256];
// ... each thread computes its score ...
shared_scores[threadIdx.x] = my_score;
__syncthreads();

// Reduction
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
        shared_scores[threadIdx.x] += shared_scores[threadIdx.x + s];
    }
    __syncthreads();
}

if (threadIdx.x == 0) {
    atomicAdd(&global_results[move_idx], shared_scores[0]);
}
```

**Implementation Complexity**: Low-Medium
**Expected Performance Impact**: 20-50% reduction in memory traffic

---

### 8. Reduce Position Struct Size

**Problem**:
The Position struct is ~280 bytes, causing high register usage and memory bandwidth.

**Current Size**:
- `int board[64]` = 256 bytes
- `int side_to_move` = 4 bytes
- `bool castling_rights[4]` = 4 bytes
- `int en_passant` = 4 bytes
- `int halfmove_clock` = 4 bytes
- `int fullmove_number` = 4 bytes
- **Total**: ~280 bytes

**Implementation Approach**:
1. Use `int8_t` for board pieces (64 bytes vs 256)
2. Pack castling rights into a single byte
3. Use `int8_t` for en_passant and halfmove_clock

**Compact Structure**:
```cpp
struct CompactPosition {
    int8_t board[64];       // 64 bytes
    uint8_t flags;          // side_to_move (1 bit), castling (4 bits)
    int8_t en_passant;      // 1 byte (-1 or 0-63)
    uint8_t halfmove_clock; // 1 byte (max 100)
    // Total: 67 bytes (vs 280)
};
```

**Implementation Complexity**: Medium (requires updating all device functions)
**Expected Performance Impact**: 30-50% reduction in register pressure, faster memory access

---

## Medium-Priority Enhancements

Algorithmic improvements to MCTS that will improve playing strength.

---

### 9. Implement True MCTS with UCB Tree Policy

**Problem**:
Current implementation uses flat Monte Carlo simulations without a search tree. True MCTS builds a tree and uses UCB to balance exploration/exploitation.

**Implementation Approach**:

1. **Add Tree Node Structure** (in global memory):
```cpp
struct MCTSNode {
    int move_from, move_to;   // The move that led to this node
    float wins;               // Sum of backpropagated scores
    int visits;               // Number of times visited
    int children_start;       // Index in global node array
    int num_children;         // Number of child nodes
    int parent;               // Parent node index
};
```

2. **UCB Selection**:
```cpp
__device__ float ucb_score(MCTSNode* node, int parent_visits) {
    float exploitation = node->wins / (node->visits + 1);
    float exploration = EXPLORATION_CONSTANT *
                        sqrtf(logf(parent_visits + 1) / (node->visits + 1));
    return exploitation + exploration;
}
```

3. **Tree Phases**:
   - Selection: Walk tree using UCB until leaf
   - Expansion: Add new node for unexplored move
   - Simulation: Run playout from new node
   - Backpropagation: Update wins/visits up the tree

**Implementation Complexity**: High
**Expected Performance Impact**: Major strength improvement (proper exploration/exploitation balance)

---

### 10. Add Virtual Loss for Parallel MCTS

**Problem**:
Without virtual loss, parallel threads may all explore the same promising branch, reducing diversity.

**Implementation Approach**:
When a thread selects a node, atomically increment a "virtual loss" counter:
```cpp
atomicAdd(&node->virtual_losses, 1);
// ... run simulation ...
atomicSub(&node->virtual_losses, 1);
atomicAdd(&node->visits, 1);
atomicAdd(&node->wins, result);
```

Modify UCB to account for virtual losses:
```cpp
float adjusted_visits = node->visits + node->virtual_losses;
```

**Implementation Complexity**: Low (if MCTS tree is implemented)
**Expected Performance Impact**: Better parallelization efficiency for tree MCTS

---

### 11. Add Quiescence-Like Extension for Captures

**Problem**:
Playouts can terminate in tactically unstable positions (e.g., mid-capture sequence).

**Current Behavior**:
Playouts stop after MAX_PLAYOUT_MOVES (200) or game end, regardless of position stability.

**Implementation Approach**:
1. After reaching terminal evaluation, check if there are forcing captures
2. If yes, extend playout by a few more moves
3. Only consider captures and checks in extension

```cpp
// After normal playout ends...
while (has_winning_capture(pos) && extension_depth < 10) {
    Move captures[32];
    int num_caps = generate_captures(pos, captures);
    if (num_caps == 0) break;
    make_best_capture(pos, captures, num_caps);
    extension_depth++;
}
```

**Implementation Complexity**: Medium
**Expected Performance Impact**: 20-40% improvement in tactical positions

---

### 12. Implement Move History Learning

**Problem**:
The CPU implementation uses history heuristic to learn which quiet moves are good. The GPU has no learning.

**Implementation Approach**:
1. Maintain a global history table in device memory
2. After each simulation, update history for successful quiet moves
3. Use atomics for thread-safe updates

```cpp
__device__ int history_table[2][64][64];  // [color][from][to]

// After successful playout:
atomicAdd(&history_table[color][move.from][move.to], score_bonus);

// In score_moves:
score += history_table[pos.side_to_move][move.from][move.to] / 1000.0f;
```

**Implementation Complexity**: Medium
**Expected Performance Impact**: 15-25% improvement in quiet positions

---

## Low-Priority Refinements

Polish and minor optimizations.

---

### 13. Use Shared Memory for Move Arrays

**Problem**:
Each thread allocates moves array in registers, causing high register pressure.

**Implementation Approach**:
Use shared memory for per-block move storage:
```cpp
__shared__ Move shared_moves[THREADS_PER_BLOCK][MAX_MOVES];
Move* my_moves = shared_moves[threadIdx.x];
```

**Caveat**: Shared memory is limited (~48KB). With 256 threads and Move size ~24 bytes, need careful sizing.

**Implementation Complexity**: Low
**Expected Performance Impact**: 10-15% reduction in register pressure

---

### 14. Kernel Fusion for Move Generation

**Problem**:
Move generation happens in separate function calls, causing function call overhead.

**Implementation Approach**:
Inline all move generation into a single loop over all squares:
```cpp
for (int sq = 0; sq < 64; sq++) {
    int piece = pos.board[sq];
    if (piece == EMPTY || is_enemy(piece, pos.side_to_move)) continue;

    switch(piece_type(piece)) {
        case PAWN: generate_pawn_from(sq, ...); break;
        case KNIGHT: generate_knight_from(sq, ...); break;
        // ...
    }
}
```

**Implementation Complexity**: Low
**Expected Performance Impact**: 5-10% faster move generation

---

### 15. Use Lookup Tables for Move Generation

**Problem**:
Move offset calculations are done at runtime.

**Implementation Approach**:
Pre-compute attack bitboards/tables in constant memory:
```cpp
__constant__ int knight_attacks[64][8];  // Pre-computed knight destinations
__constant__ int king_attacks[64][8];
// etc.
```

**Implementation Complexity**: Low
**Expected Performance Impact**: 5-10% faster move generation

---

### 16. Add Checkmate Bonus Scaling by Depth

**Problem**:
Checkmates at different depths are valued similarly, making the engine not prefer faster mates.

**Location**: `monte_carlo_advanced_kernel.cu:593-596`

**Current Code**:
```cpp
int mate_bonus = MATE_SCORE - ply;
```

This is correct! But the MATE_SCORE constant (10000) may not be large enough relative to evaluation scores.

**Implementation Approach**:
Increase MATE_SCORE to avoid evaluation scores outweighing mates:
```cpp
const int MATE_SCORE = 100000;  // Was 10000
```

**Implementation Complexity**: Trivial
**Expected Performance Impact**: Better mate finding

---

## Implementation Priority Matrix

| Priority | Item | Complexity | Impact | Dependencies |
|----------|------|------------|--------|--------------|
| **P0** | Castling | Medium | Correctness | None |
| **P0** | En Passant | Low-Medium | Correctness | None |
| **P0** | King Legality Check | Medium | Correctness | None |
| **P1** | Parallel Move Eval | Medium-High | 5-10x speed | None |
| **P1** | Position Size Reduction | Medium | 30-50% speed | None |
| **P1** | Batched Reduction | Low-Medium | 20-50% speed | None |
| **P2** | True MCTS Tree | High | Major strength | P0 items |
| **P2** | Transposition Detection | Medium | 10-30% quality | None |
| **P2** | Quiescence Extension | Medium | 20-40% tactical | None |
| **P3** | History Learning | Medium | 15-25% positional | None |
| **P3** | Virtual Loss | Low | Parallelism | MCTS Tree |
| **P4** | Shared Memory Moves | Low | 10-15% speed | None |
| **P4** | Move Gen Fusion | Low | 5-10% speed | None |

---

## Recommended Implementation Order

### Phase 1: Correctness (Required)
1. Implement castling
2. Implement en passant
3. Add legal move filtering

### Phase 2: Core Performance
4. Parallel move evaluation with CUDA streams
5. Reduce Position struct size
6. Add batched result reduction

### Phase 3: Algorithmic Strength
7. Implement transposition detection
8. Add quiescence extension
9. Consider true MCTS tree (if strength still lacking)

### Phase 4: Polish
10. History table
11. Shared memory optimizations
12. Lookup table move generation

---

## Estimated Performance After Improvements

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Simulations/sec | ~50K | ~50K | ~250K-500K | ~500K-1M |
| Correctness | Broken | Fixed | Fixed | Fixed |
| Tactical Strength | Weak | Weak | Weak-Medium | Medium |
| vs CPU Depth 10 | Loses | Loses | Competitive | Competitive |

---

## Appendix: Code Snippets for Key Fixes

### Castling Generation (Pseudocode)

```cpp
__device__ void generate_castling_moves(const Position& pos, Move* moves, int& count) {
    int color = pos.side_to_move;
    int king_sq = (color == GPU_WHITE) ? 4 : 60;  // e1 or e8

    if (pos.board[king_sq] != ((color == GPU_WHITE) ? W_KING : B_KING)) return;
    if (is_in_check(pos, color)) return;  // Can't castle out of check

    // Kingside
    int ks_right = (color == GPU_WHITE) ? 0 : 2;
    int ks_rook_sq = king_sq + 3;
    int ks_f_sq = king_sq + 1;
    int ks_g_sq = king_sq + 2;

    if (pos.castling_rights[ks_right] &&
        pos.board[ks_f_sq] == EMPTY &&
        pos.board[ks_g_sq] == EMPTY &&
        !is_square_attacked(pos, ks_f_sq, 1 - color) &&
        !is_square_attacked(pos, ks_g_sq, 1 - color)) {
        moves[count++] = {king_sq, ks_g_sq, 0, EMPTY, pos.board[king_sq], 50.0f};
    }

    // Queenside (similar)
    // ...
}
```

### Parallel Move Evaluation with Streams

```cpp
void evaluate_all_moves_parallel(const chess::Board& board, int sims_per_move) {
    const int NUM_STREAMS = 8;
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    std::vector<float*> d_results(num_moves);
    std::vector<float*> h_results(num_moves);

    // Allocate memory for all moves
    for (int i = 0; i < num_moves; i++) {
        cudaMalloc(&d_results[i], total_threads * sizeof(float));
        h_results[i] = new float[total_threads];
    }

    // Launch all kernels asynchronously
    for (int i = 0; i < num_moves; i++) {
        int stream_idx = i % NUM_STREAMS;
        launch_kernel<<<blocks, threads, 0, streams[stream_idx]>>>(
            position, moves[i], d_results[i]
        );
    }

    // Copy results asynchronously
    for (int i = 0; i < num_moves; i++) {
        int stream_idx = i % NUM_STREAMS;
        cudaMemcpyAsync(h_results[i], d_results[i],
                        total_threads * sizeof(float),
                        cudaMemcpyDeviceToHost, streams[stream_idx]);
    }

    cudaDeviceSynchronize();

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}
```
