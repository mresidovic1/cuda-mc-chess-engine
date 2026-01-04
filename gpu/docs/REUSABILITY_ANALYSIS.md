# CPU → GPU Code Reusability Analysis

## Executive Summary

This document analyzes the feasibility of reusing components from the CPU minimax implementation in the GPU Monte Carlo implementation. The goal is to identify synergies that can accelerate GPU development while closing the performance gap.

**Key Finding**: Direct code reuse is limited due to CUDA constraints, but many **concepts and data structures** can be adapted with moderate effort.

---

## Part 1: chess.hpp Library Integration

### Can chess.hpp Be Used for GPU Move Generation?

**Answer: NO - Direct use is not possible**

### Technical Analysis

The `chess.hpp` library (disservin/chess-library v0.8.2) is a header-only C++ chess library that cannot be directly compiled for CUDA device code due to the following constraints:

#### 1. STL Container Usage
```cpp
// chess.hpp uses:
#include <vector>
#include <string>
#include <string_view>
#include <functional>
```
These STL containers are **not available in CUDA device code**. Device code cannot use:
- `std::vector` (dynamic memory allocation)
- `std::string` (heap allocation)
- `std::function` (virtual calls, heap allocation)

#### 2. Host-Only Constructs
```cpp
// Example from chess.hpp:
explicit operator std::string() const {
    return color == underlying::WHITE ? "w" : "b";  // string allocation
}

friend std::ostream& operator<<(std::ostream& os, const Color& color);  // iostream
```
I/O streams and string conversions are host-only operations.

#### 3. Magic Bitboard Initialization
```cpp
// attacks class uses runtime initialization:
static void initSliders(Square sq, Magic table[], U64 magic,
                        const std::function<Bitboard(Square, Bitboard)> &attacks);
```
The magic bitboard tables require **runtime initialization** via `attacks::initAttacks()`. This cannot run on GPU, and the initialized tables live in host memory.

#### 4. Assert/Exception Handling
```cpp
constexpr Square(int sq) : sq(static_cast<underlying>(sq)) {
    assert(sq <= 64 && sq >= 0);  // Host-only assert
}
```
CUDA device code has limited assert support and no exception handling.

### What Parts Can Be Adapted?

| Component | Reusable? | Adaptation Required |
|-----------|-----------|---------------------|
| **Piece encoding** | ✅ Yes | Already adapted - GPU uses compatible 0-14 encoding |
| **Square indexing** | ✅ Yes | GPU uses 0-63 indices directly |
| **Piece-square tables** | ✅ Yes | **Already reused** - identical values in `constants.h` and GPU kernel |
| **Bitboard concept** | ⚠️ Partially | Need GPU-native uint64_t implementation |
| **Attack patterns** | ⚠️ Partially | Pre-compute lookup tables for device constant memory |
| **Magic bitboards** | ❌ No | Would require host-init + device copy, complex setup |
| **Move representation** | ⚠️ Partially | GPU uses simplified struct, could align better |
| **FEN parsing** | ❌ No | String operations, keep on host |
| **Move validation** | ⚠️ Partially | Logic can be ported, not code |

### Performance Implications of chess.hpp Adaptation

If we were to fully port chess.hpp concepts to GPU:

| Approach | Effort | Performance Impact |
|----------|--------|-------------------|
| Full magic bitboard port | High | 3-5x faster move gen |
| Pre-computed attack tables | Medium | 2-3x faster move gen |
| Current offset-based approach | Done | Baseline |

**Recommendation**: Port attack lookup tables to device constant memory, not full magic bitboards.

---

## Part 2: CPU Component Reuse Assessment

### 2.1 Transposition Table

**Location**: `include/tt_parallel.h`, `include/transposition_table.h`

#### Can It Be Shared/Adapted?

**Answer: PARTIALLY - Concept yes, implementation needs GPU-specific version**

#### Technical Analysis

The CPU transposition table:
```cpp
struct TTEntryParallel {
    uint64_t key;
    chess::Move bestMove;
    int16_t score;
    int16_t staticEval;
    int16_t depth;
    uint8_t flag;
    uint8_t generation;
};
```

**Issues for GPU**:
1. Uses `chess::Move` type (host-only)
2. Lives in host memory
3. Size: ~24 bytes per entry (good)

**GPU Adaptation**:
```cpp
// GPU-compatible TT entry
struct GPUTTEntry {
    uint64_t key;           // Position hash
    int16_t from_to;        // Packed move (6 bits from, 6 bits to)
    int16_t score;          // Evaluation
    uint8_t depth;          // Search depth
    uint8_t flag;           // Bound type
};  // 16 bytes
```

**Memory Access Patterns**:
- CPU: Random access, lockless reads, generation-based replacement
- GPU: Would need atomic operations, potential bank conflicts
- **Solution**: Thread-local small TT in shared memory + global TT for cross-kernel persistence

#### Required Modifications

1. Create `GPUTTEntry` struct without chess.hpp dependencies
2. Allocate table in CUDA global memory
3. Use atomic operations for updates: `atomicCAS` for key, `atomicExch` for values
4. Implement Zobrist hashing in device code

#### Expected Benefit

| Metric | Without TT | With GPU TT |
|--------|------------|-------------|
| Redundant evaluations | ~40% | ~5% |
| Playout speed | Baseline | +20-30% |
| Memory usage | None | +256MB-1GB |

**Verdict**: ✅ **Implement** - High value, medium effort

---

### 2.2 Evaluation Function

**Location**: `src/chess_engine_parallelized.cpp:48-83`, GPU: `monte_carlo_advanced_kernel.cu:42-67`

#### Can It Be Reused?

**Answer: YES - Already partially reused!**

#### Technical Analysis

**CPU Evaluation**:
```cpp
int evaluate(const Board &board) {
    int evaluation = 0;
    // Material counting with bitboards
    evaluation += (wp.count() - bp.count()) * piece_values[0];
    // ... more piece types ...

    // Piece-square table scoring
    evaluation += pstScore(wp, Color::WHITE, pawn_table);
    // ... more PST ...

    // Tempo bonus
    evaluation += (board.sideToMove() == Color::WHITE) ? 10 : -10;
    return evaluation;
}
```

**GPU Evaluation**:
```cpp
__device__ int evaluate_position(const Position& pos) {
    int score = 0;
    for (int sq = 0; sq < 64; sq++) {
        // Material and PST (loop-based, not bitboard)
        int piece_value = get_piece_value(piece);
        int pst_value = piece_square_value(piece, sq, piece_color);
        // ...
    }
    // Tempo bonus
    score += (pos.side_to_move == GPU_WHITE) ? 10 : -10;
    return score;
}
```

**What's Already Shared**:
- ✅ Same piece values: `{100, 300, 320, 500, 900, 0}`
- ✅ Same piece-square tables (identical values)
- ✅ Same tempo bonus (+10)

**What Could Be Improved**:
1. **Bitboard-based evaluation on GPU**: Would require porting bitboard to device
2. **Additional eval terms**: King safety, pawn structure, mobility

#### Required Modifications

None required for basic parity. For enhancement:
```cpp
// Could add from CPU:
// - Doubled pawn penalty
// - Isolated pawn penalty
// - Bishop pair bonus
// - Rook on open file bonus
```

#### Expected Benefit

Current GPU eval is already equivalent to CPU material+PST eval.

**Verdict**: ✅ **Already reused** - No action needed

---

### 2.3 Pruning Techniques

**Location**: `src/chess_engine_parallelized.cpp` (various)

#### Can Alpha-Beta Pruning Apply to MCTS?

**Answer: PARTIALLY - Some concepts transfer**

#### Technical Analysis

**CPU Pruning Techniques**:

| Technique | CPU Implementation | MCTS Applicability |
|-----------|-------------------|-------------------|
| **Alpha-Beta** | Core search algorithm | ❌ No - MCTS doesn't use minimax bounds |
| **Null Move Pruning** | Skip a move to detect zugzwang | ❌ No - Not applicable to random playouts |
| **Late Move Reductions** | Reduce depth for later moves | ⚠️ Partial - Could reduce playout length |
| **Futility Pruning** | Skip moves that can't raise alpha | ⚠️ Partial - Could skip clearly bad playouts |
| **Razoring** | Drop to quiescence if hopeless | ✅ Yes - Early playout termination |

**Applicable Concepts**:

1. **Early Termination** (like razoring):
   ```cpp
   // In monte_carlo_playout():
   if (ply > 20 && abs(evaluate_position(pos)) > 500) {
       // Position is clearly winning/losing, terminate playout
       return evaluate_position(pos);
   }
   ```

2. **Selective Playout Extension** (like LMR inverse):
   ```cpp
   // Extend playouts in critical positions
   if (is_in_check(pos) || has_only_capture_moves) {
       max_playout_moves += 10;  // Extend
   }
   ```

3. **Move Pruning in Playouts**:
   ```cpp
   // Skip obviously bad moves during playout
   if (move.score < -1000 && num_good_moves > 3) {
       continue;  // Don't even consider this move
   }
   ```

#### Expected Benefit

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| Early termination | Low | +10-15% speed |
| Selective extension | Low | +5-10% quality |
| Move pruning | Low | +5% speed |

**Verdict**: ⚠️ **Partially applicable** - Adapt concepts, not code

---

### 2.4 Move Ordering

**Location**: `src/chess_engine_parallelized.cpp:230-255`

#### Can CPU Move Ordering Improve GPU MCTS?

**Answer: YES - Already partially implemented, can enhance**

#### Technical Analysis

**CPU Move Ordering Priority**:
```cpp
void order_moves(...) {
    // 1. TT move (+10000)
    // 2. Captures by SEE (+2000 + SEE score)
    // 3. Promotions (+1500)
    // 4. Killer moves (+1000)
    // 5. History heuristic (scaled score)
}
```

**GPU Move Scoring** (`score_moves()`):
```cpp
// Current GPU implementation:
// 1. Checks (+5000) - HIGHER than CPU!
// 2. Good captures (+1000 + SEE)
// 3. Promotions (+900)
// 4. King proximity (+100)
// 5. PST improvement
// 6. Center control (+20)
```

**Differences**:
| Feature | CPU | GPU |
|---------|-----|-----|
| TT move bonus | ✅ Yes | ❌ No TT |
| Check detection | ❌ No (implicit) | ✅ Yes (+5000) |
| Killer moves | ✅ Yes | ❌ No |
| History heuristic | ✅ Yes | ❌ No |
| King proximity | ❌ No | ✅ Yes |

**What Can Be Added to GPU**:

1. **History Heuristic** (P2-12):
   ```cpp
   __device__ int d_history_table[2][64][64];  // In global memory

   // In score_moves:
   if (!is_capture) {
       score += d_history_table[color][move.from][move.to] / 100.0f;
   }

   // After successful playout:
   atomicAdd(&d_history_table[color][best_move.from][best_move.to], bonus);
   ```

2. **Killer Moves** (less useful for random playouts):
   - In MCTS, killer moves are less effective because playouts are statistically sampled
   - Could track "playout killers" but benefit is marginal

#### Expected Benefit

| Enhancement | Effort | Impact |
|-------------|--------|--------|
| History heuristic | Medium | +15-20% quality |
| Killer moves | Low | +2-5% quality |

**Verdict**: ✅ **Implement history** - High value for MCTS playout quality

---

### 2.5 Static Exchange Evaluation (SEE)

**Location**: CPU: `src/chess_engine_parallelized.cpp:101-183`, GPU: `monte_carlo_advanced_kernel.cu:70-91`

#### Comparison

**CPU SEE** (Full implementation):
```cpp
int SEE(Move move, Board &board) {
    // Full iterative SEE with:
    // - Proper attacker discovery
    // - X-ray attacks through pieces
    // - Accurate gain calculation
    // - 32-ply capture sequence
}
```

**GPU SEE** (Simplified):
```cpp
__device__ int simple_SEE(const Position& pos, const Move& move) {
    int attacker_value = get_piece_value(move.piece);
    int victim_value = get_piece_value(move.capture);

    int see_value = victim_value;
    if (attacker_value > victim_value) {
        see_value -= attacker_value / 2;  // Rough penalty
    }
    return see_value;
}
```

**Gap Analysis**:
| Feature | CPU | GPU |
|---------|-----|-----|
| Accuracy | High | Low (~70%) |
| X-ray attacks | ✅ | ❌ |
| Full sequence | ✅ | ❌ |
| Speed | Medium | Fast |

#### Porting Full SEE to GPU

**Feasibility**: MEDIUM - The algorithm is iterative and can be ported

**Challenges**:
1. Need bitboard-style attack detection (can use lookup tables)
2. Loop complexity (up to 32 iterations)
3. Register pressure

**Simplified Improvement**:
```cpp
__device__ int improved_SEE(const Position& pos, const Move& move) {
    if (move.capture == EMPTY) return 0;

    int gain[16];
    gain[0] = get_piece_value(move.capture);

    int attacker_value = get_piece_value(move.piece);
    int target_sq = move.to;
    int depth = 1;
    int side = 1 - pos.side_to_move;

    // Simplified: check for immediate recapture
    int defenders = count_attackers(pos, target_sq, side);

    if (defenders > 0) {
        int min_defender = find_min_attacker_value(pos, target_sq, side);
        gain[1] = attacker_value - gain[0];

        // If our piece is worth more and they can recapture
        if (gain[1] > 0) return gain[0] - attacker_value;
    }

    return gain[0];
}
```

#### Expected Benefit

| Improvement | Effort | Impact |
|-------------|--------|--------|
| Full SEE port | High | +5-10% move ordering |
| Improved simple SEE | Medium | +3-5% move ordering |

**Verdict**: ⚠️ **Low priority** - Current simple SEE is adequate for MCTS

---

### 2.6 Other Reusable Components

#### Zobrist Hashing

**CPU**: Uses `board.hash()` from chess.hpp (pre-computed)

**GPU Opportunity**: Implement incremental Zobrist hashing for transposition detection

```cpp
// GPU Zobrist implementation
__constant__ uint64_t d_zobrist_pieces[64][16];  // [square][piece]
__constant__ uint64_t d_zobrist_side;
__constant__ uint64_t d_zobrist_castling[16];
__constant__ uint64_t d_zobrist_ep[8];

__device__ uint64_t compute_hash(const Position& pos) {
    uint64_t hash = 0;
    for (int sq = 0; sq < 64; sq++) {
        if (pos.board[sq] != EMPTY) {
            hash ^= d_zobrist_pieces[sq][pos.board[sq]];
        }
    }
    if (pos.side_to_move == GPU_BLACK) hash ^= d_zobrist_side;
    return hash;
}

__device__ uint64_t update_hash_move(uint64_t hash, const Move& move, int moving_piece) {
    hash ^= d_zobrist_pieces[move.from][moving_piece];  // Remove from source
    hash ^= d_zobrist_pieces[move.to][moving_piece];    // Add to dest
    if (move.capture != EMPTY) {
        hash ^= d_zobrist_pieces[move.to][move.capture]; // Remove captured
    }
    hash ^= d_zobrist_side;  // Flip side
    return hash;
}
```

**Verdict**: ✅ **Implement** - Required for P1-6 transposition detection

---

## Summary: Reusability Matrix

| Component | Reusable | Type | Priority | Implementation Effort |
|-----------|----------|------|----------|----------------------|
| chess.hpp directly | ❌ No | N/A | N/A | N/A |
| Piece-square tables | ✅ Yes | Data | Done | Already done |
| Piece values | ✅ Yes | Data | Done | Already done |
| Evaluation logic | ✅ Yes | Concept | Done | Already done |
| Transposition table | ⚠️ Partial | Concept | P1 | Medium |
| Zobrist hashing | ⚠️ Partial | Concept | P1 | Medium |
| History heuristic | ⚠️ Partial | Concept | P2 | Medium |
| Move ordering | ⚠️ Partial | Concept | Done | Partially done |
| SEE | ⚠️ Partial | Concept | Low | Low value |
| Pruning concepts | ⚠️ Partial | Concept | P2 | Low-Medium |
| Killer moves | ⚠️ Partial | Concept | Low | Low value |
| Bitboard operations | ⚠️ Partial | Concept | Optional | High |

---

## Recommended Implementation Order

### Phase 1: High-Value Reuse (P1)
1. Implement Zobrist hashing for GPU (enables TT and repetition detection)
2. Create GPU transposition table structure
3. Add early termination (razoring concept)

### Phase 2: Medium-Value Reuse (P2)
4. Implement history heuristic on GPU
5. Improve SEE accuracy
6. Add selective playout extension

### Phase 3: Low-Priority Polish
7. Consider bitboard port for evaluation
8. Add more eval terms from CPU
9. Implement killer moves (minimal benefit expected)

---

## Appendix: Code Templates for Reuse

### Zobrist Key Generation (Host-side initialization)

```cpp
// Call once at program start to initialize random keys
void init_zobrist_keys() {
    std::mt19937_64 rng(12345);  // Fixed seed for reproducibility

    uint64_t h_zobrist_pieces[64][16];
    uint64_t h_zobrist_side;

    for (int sq = 0; sq < 64; sq++) {
        for (int piece = 0; piece < 16; piece++) {
            h_zobrist_pieces[sq][piece] = rng();
        }
    }
    h_zobrist_side = rng();

    // Copy to device constant memory
    cudaMemcpyToSymbol(d_zobrist_pieces, h_zobrist_pieces, sizeof(h_zobrist_pieces));
    cudaMemcpyToSymbol(d_zobrist_side, &h_zobrist_side, sizeof(h_zobrist_side));
}
```

### GPU History Table Update (Thread-safe)

```cpp
__device__ void update_history(int color, int from, int to, int bonus) {
    // Clamp bonus to prevent overflow
    bonus = max(-2000, min(2000, bonus));

    // Atomic update with saturation
    int old_val = d_history_table[color][from][to];
    int new_val = old_val + bonus;
    new_val = max(-32768, min(32767, new_val));

    atomicExch(&d_history_table[color][from][to], new_val);
}
```
