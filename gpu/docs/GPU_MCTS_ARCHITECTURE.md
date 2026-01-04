# GPU Monte Carlo Chess Engine - Architecture Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Monte Carlo Search Algorithm](#monte-carlo-search-algorithm)
4. [GPU Parallelization Strategy](#gpu-parallelization-strategy)
5. [Data Structures](#data-structures)
6. [Key Device Functions](#key-device-functions)
7. [Performance Characteristics](#performance-characteristics)
8. [Integration Points](#integration-points)
9. [Comparison with CPU Implementation](#comparison-with-cpu-implementation)

---

## Overview

The GPU implementation uses a **Monte Carlo simulation approach** with heuristic-guided playouts rather than a traditional Monte Carlo Tree Search (MCTS) with UCB. The engine runs massively parallel simulations on NVIDIA GPUs using CUDA, evaluating positions through statistical sampling of game outcomes.

### Key Characteristics
- **Algorithm**: Heuristic-guided Monte Carlo simulations (not full MCTS with tree building)
- **Parallelization**: CUDA kernels with thousands of concurrent threads
- **Evaluation**: Combination of win/loss statistics and position evaluation
- **Move Selection**: Softmax-weighted random selection with strong heuristics

---

## Architecture

### File Structure
```
gpu/
├── monte_carlo_advanced_kernel.cu    # CUDA kernels and device functions
├── monte_carlo_advanced_kernel.cuh   # Header with constants, structs, declarations
├── monte_carlo_advanced.cpp          # Host-side C++ wrapper
├── monte_carlo_advanced.hpp          # C++ interface declarations
├── main_advanced.cpp                 # Test driver and CLI
├── Makefile                          # Build configuration
└── build.bat / build.sh              # Build scripts
```

### Component Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        Host (CPU) Side                          │
│  ┌──────────────────┐    ┌─────────────────────────────────┐   │
│  │ main_advanced.cpp │───>│ monte_carlo_advanced.cpp        │   │
│  │ (Test positions) │    │ - board_to_gpu_position()       │   │
│  └──────────────────┘    │ - chess_move_to_gpu_move()      │   │
│                          │ - evaluate_all_moves()          │   │
│                          │ - find_best_move()              │   │
│                          └───────────────┬─────────────────┘   │
└──────────────────────────────────────────│─────────────────────┘
                                           │ CUDA Launch
                                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Device (GPU) Side                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           monte_carlo_simulate_kernel()                  │   │
│  │  ┌────────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ Thread 0-255   │  │ Thread 256-511               │   │   │
│  │  │ - make root mv │  │ - make root mv               │   │   │
│  │  │ - run playout  │  │ - run playout                │   │   │
│  │  │ - score result │  │ - score result               │   │   │
│  │  └────────────────┘  └──────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Device Functions:                                              │
│  - generate_all_moves()   - monte_carlo_playout()              │
│  - evaluate_position()    - score_moves()                      │
│  - is_square_attacked()   - select_move_weighted()             │
│  - make_move()            - check_for_immediate_mate()         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Monte Carlo Search Algorithm

### Algorithm Type: Heuristic-Guided Monte Carlo Simulation

Unlike traditional MCTS with UCB (Upper Confidence Bound) tree policy, this implementation uses:

1. **No Persistent Tree**: Each move evaluation is independent; no tree is built or maintained
2. **Root Parallelism**: For each legal move, multiple simulations are run in parallel
3. **Heuristic-Guided Playouts**: Move selection during playouts uses evaluation heuristics, not random play

### Workflow
```
For each legal move at root:
  1. Launch CUDA kernel with N threads (e.g., 256 blocks × 256 threads)
  2. Each thread:
     a. Make the root move
     b. Run a playout (up to 200 ply)
     c. Score the final position
     d. Store result in global memory
  3. Average all thread results
  4. Return average as move score
```

### Playout Strategy (`monte_carlo_playout()`)

```cpp
for (ply = 0; ply < MAX_PLAYOUT_MOVES; ply++) {
    // 1. Generate all legal moves
    num_moves = generate_all_moves(pos, moves);

    // 2. Check for game over (checkmate, stalemate, 50-move)
    if (game_over) return game_result;

    // 3. Score all moves with heuristics
    score_moves(pos, moves, num_moves);  // Captures, checks, PST, etc.

    // 4. Check for immediate mate delivery
    if (best_move_gives_mate) return MATE_SCORE;

    // 5. Select move
    if (ply < 2) {
        selected = best_move;  // Greedy for first 2 moves
    } else {
        selected = random_from_top_3;  // Exploration after ply 2
    }

    // 6. Make move
    make_move(pos, selected);
}

// If no terminal, return position evaluation
return evaluate_position(pos);
```

### Move Scoring Heuristics (`score_moves()`)

Priority order (descending):
| Heuristic | Score Bonus | Description |
|-----------|-------------|-------------|
| Gives Check | +5000 | Critical for finding mates |
| Good Capture (SEE > 0) | +1000 + SEE | Material gain |
| Promotion | +900 | Queen promotion |
| Bad Capture (SEE < 0) | +200 + SEE | Might still be tactical |
| King Proximity | +100 | Moving closer to enemy king |
| PST Improvement | Variable | Piece-square table delta |
| Center Control | +20 | d4/d5/e4/e5 bonus |

---

## GPU Parallelization Strategy

### Thread Organization

```
Kernel Launch Configuration:
- Blocks: ceil(simulations_per_move / threads_per_move)
- Threads per block: 256 (default)
- Total threads: blocks × 256
- Simulations per thread: ceil(total_sims / total_threads)
```

### Memory Hierarchy

| Memory Type | Usage | Data |
|-------------|-------|------|
| **Constant Memory** | Read-only | Piece values, PST tables |
| **Global Memory** | Read/write | Results array, random seeds |
| **Registers** | Per-thread | Position state, local variables |
| **Shared Memory** | Not used | (Potential optimization target) |

### Constant Memory Layout (`__constant__`)

```cpp
__constant__ int d_piece_values[6] = {100, 300, 320, 500, 900, 0};
__constant__ int d_pawn_table[64] = {...};
__constant__ int d_knight_table[64] = {...};
__constant__ int d_bishop_table[64] = {...};
__constant__ int d_rook_table[64] = {...};
__constant__ int d_queen_table[64] = {...};
__constant__ int d_king_table[64] = {...};
```

### Random Number Generation

Uses cuRAND for per-thread random state:
```cpp
curandState rand_state;
curand_init(seed, thread_idx, 0, &rand_state);
float r = curand_uniform(&rand_state);  // For move selection
```

---

## Data Structures

### Position (GPU representation)

```cpp
struct Position {
    int board[64];           // Piece per square (0=empty, 1-6=white, 9-14=black)
    int side_to_move;        // 0=WHITE, 1=BLACK
    bool castling_rights[4]; // WK, WQ, BK, BQ (NOT IMPLEMENTED in kernel)
    int en_passant;          // -1 or square index (NOT IMPLEMENTED)
    int halfmove_clock;      // For 50-move rule
    int fullmove_number;     // Game move counter
};
```

### Move (GPU representation)

```cpp
struct Move {
    int from;       // Source square (0-63)
    int to;         // Target square (0-63)
    int promotion;  // 0=none, 2=knight, 3=bishop, 4=rook, 5=queen
    int capture;    // Captured piece type
    int piece;      // Moving piece type
    float score;    // Heuristic score for ordering
};
```

### Piece Encoding

```
White: W_PAWN=1, W_KNIGHT=2, W_BISHOP=3, W_ROOK=4, W_QUEEN=5, W_KING=6
Black: B_PAWN=9, B_KNIGHT=10, B_BISHOP=11, B_ROOK=12, B_QUEEN=13, B_KING=14
Empty: 0
```

---

## Key Device Functions

### Move Generation

| Function | Purpose |
|----------|---------|
| `generate_pawn_moves()` | Pawn pushes, doubles, captures, promotions |
| `generate_knight_moves()` | L-shaped knight jumps |
| `generate_sliding_moves()` | Bishop/rook/queen rays |
| `generate_king_moves()` | Single-step king moves |
| `generate_all_moves()` | Combines all piece generators |

**Note**: Castling and en passant are NOT implemented in the GPU move generator.

### Attack Detection

```cpp
__device__ bool is_square_attacked(const Position& pos, int square, int attacking_color);
```
Checks all piece attack patterns (pawns, knights, sliders, king) for a given square.

### Position Evaluation

```cpp
__device__ int evaluate_position(const Position& pos);
```
- Material counting with standard values
- Piece-square table bonuses
- Tempo bonus (+10 for side to move)
- Returns score from side-to-move perspective

### SEE (Static Exchange Evaluation)

```cpp
__device__ int simple_SEE(const Position& pos, const Move& move);
```
Simplified SEE that estimates capture value minus risk of losing attacker.

---

## Performance Characteristics

### Throughput

From README benchmarks:
- **~50K simulations/second** (varies by GPU)
- Compared to CPU: **~1M positions/second** (minimax with pruning)

### Bottlenecks

1. **Sequential Move Evaluation**: Each root move is evaluated serially; kernel launches are sequential
2. **Memory Bandwidth**: Large Position structs (256+ bytes) passed by value
3. **Register Pressure**: Each thread maintains full game state in registers
4. **Warp Divergence**: Different threads take different playout paths
5. **No Transposition Table**: Repeated positions are re-evaluated

### Strengths

1. **Massive Parallelism**: Thousands of simulations run concurrently
2. **Tactical Awareness**: Heuristic-guided playouts find checks and captures
3. **No Search Explosion**: Constant depth regardless of game complexity
4. **GPU Utilization**: Keeps GPU cores busy with independent work

---

## Integration Points

### Host-Device Interface

```cpp
// Convert chess.hpp types to GPU types
Position board_to_gpu_position(const chess::Board& board);
Move chess_move_to_gpu_move(const chess::Move& move, const chess::Board& board);

// Launch kernel wrapper
extern "C" void launch_monte_carlo_simulate_kernel(
    const Position* root_position,
    const Move* root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed,
    int blocks,
    int threads_per_block
);
```

### Chess Library Dependency

Uses `chess.hpp` (disservin/chess-library v0.8.2) for:
- FEN parsing
- Legal move generation (host side)
- Board display
- UCI move conversion

---

## Comparison with CPU Implementation

| Aspect | CPU (Lazy SMP Minimax) | GPU (Monte Carlo) |
|--------|------------------------|-------------------|
| **Algorithm** | Negamax + Alpha-Beta | Heuristic Monte Carlo |
| **Parallelism** | OpenMP threads (8-32) | CUDA threads (10,000+) |
| **Tree Structure** | Explicit game tree with pruning | No tree, flat simulations |
| **Transposition Table** | Yes (lock-free TT) | No |
| **Killer Moves** | Yes (per-thread) | No |
| **History Heuristic** | Yes | No |
| **Null Move Pruning** | Yes | No |
| **LMR** | Yes | No (fixed exploration) |
| **Quiescence Search** | Yes | Implicit in playouts |
| **Evaluation** | PST + material | PST + material + simulation stats |
| **Move Ordering** | SEE + History + Killers | SEE + Check detection |
| **Castling** | Yes | No |
| **En Passant** | Yes | No |
| **Depth** | Variable (iterative deepening) | Fixed playout length (200) |

### Performance Gap Analysis

The CPU implementation is currently **faster and stronger** because:

1. **Alpha-Beta Pruning**: CPU prunes ~99% of the search tree; GPU evaluates all moves fully
2. **Transposition Table**: CPU avoids redundant work; GPU recomputes everything
3. **Depth**: CPU reaches depth 10-16 with full evaluation; GPU does 200-ply random playouts
4. **Move Ordering**: CPU's history heuristic learns during search; GPU has no learning

---

## Future Development Notes

See `IMPROVEMENT_ROADMAP.md` for detailed enhancement plans covering:
- Critical bug fixes (castling, en passant)
- High-priority optimizations (parallel move evaluation, batched simulations)
- Algorithmic improvements (tree MCTS, virtual loss)
- Low-priority refinements (shared memory, kernel fusion)
