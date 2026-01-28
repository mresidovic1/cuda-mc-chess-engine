# CUDA MCTS Chess Engine - Improvements Documentation

This document describes the improvements implemented in the CUDA-based Monte Carlo Tree Search chess engine, as listed in `improvements.txt`.

---

## Bug Fixes

### 1. Playout Perspective Bug Fix
**Purpose:** Fixed bug that caused the engine to sometimes maximize the opponent's score instead of its own.

**Description:** The playout evaluation was incorrectly scoring positions from the wrong perspective, leading to suboptimal move selection.

**Implementation:**
- **File:** `gpu/src/kernels/playouts.cu`
- **Lines:** 99-111 (EvalPlayout - perspective correction)
- **Lines:** 395-405 (QuiescencePlayout - perspective correction)
- **File:** `gpu/include/evaluation.h`
- **Lines:** 846-853 (`score_to_winprob()` - proper side-to-move handling)

---

### 2. Tactical Solver Perspective Bug Fix
**Purpose:** Fixed bug causing all hard tactical tests to fail with very negative scores.

**Description:** The tactical solver was evaluating positions from the wrong perspective, resulting in incorrect move recommendations.

**Implementation:**
- **File:** `gpu/src/kernels/tactical_search.cu`
- **File:** `gpu/include/kernel_launchers.h`
- **Lines:** 41-48 (tactical solver kernel declaration)

---

## Methodology Fixes

### 3. Increased Number of Simulations
**Purpose:** Increase default simulation counts for better search quality.

**Description:** All simulation presets were significantly increased:

| Preset | Before | After |
|--------|--------|-------|
| Default | 10,000 | 350,000 |
| Quick | 1,000 | 50,000 |
| Normal | 10,000 | 350,000 |
| Strong | 50,000 | 1,000,000 |

**Implementation:**
- **File:** `gpu/include/puct_mcts.h`
- **Lines:**
  - 504: `num_simulations = 350000` (default)
  - 578: `Quick()` preset - 50,000 simulations
  - 586: `Normal()` preset - 350,000 simulations
  - 596-607: `Strong()` preset - 1,000,000 simulations
  - 611-628: `Adaptive()` preset - 350,000 simulations
  - 631-664: `Advanced()` preset - 350,000 simulations

---

## Core Engine Improvements

### 4. Continuation History
**Purpose:** Track move quality by assigning higher priority to better moves.

**Description:** Implements Stockfish-style continuation history that tracks move quality based on the piece type and destination square, allowing for more accurate move ordering in subsequent searches.

**Implementation:**
- **File:** `gpu/include/puct_mcts.h`
- **Lines:** 134-173 (`ContinuationHistory` class definition)
- **Lines:** 541-542 (Configuration: `use_continuation_history`, `continuation_weight`)
- **File:** `gpu/src/puct_mcts.cpp`
- **Lines:** 702-708 (Usage in move prior computation)
- **Lines:** 935-943 (`update_history_tables()` method)

---

### 5. Stockfish 10's Best Move Change Tracking
**Purpose:** Track how often the best move changes during search to enable early stopping.

**Description:** Monitors the stability of the best move throughout the search. If the best move remains stable, the search can terminate early. If unstable, the search may be extended.

**Implementation:**
- **File:** `gpu/include/puct_mcts.h`
- **Lines:** 177-216 (`BestMoveTracker` struct)
- **File:** `gpu/src/puct_mcts.cpp`
- **Lines:** 330 (Best move tracking during search)
- **Lines:** 361-367 (Early stopping when stable)
- **Lines:** 372-377 (Search extension when unstable)

---

### 6. Adaptive Exploration
**Purpose:** Dynamically adjust exploration behavior based on node statistics.

**Description:** Two components:
- **Dynamic c_puct:** Exploration constant adapts based on node visit counts using AlphaGo Zero formula
- **Move number scaling:** Stockfish-style exploration reduction for later moves in the ordering

**Implementation:**
- **File:** `gpu/include/puct_mcts.h`
- **Lines:** 544-551 (Adaptive exploration configuration)
- **Lines:** 670-717 (`adaptive_puct_score()` method)
- **File:** `gpu/src/puct_mcts.cpp`
- **Lines:** 528-536 (Adaptive PUCT selection with move number scaling)
- **Lines:** 557-561 (Dynamic c_puct calculation)
- **Lines:** 1012-1025 (`compute_move_number_factor()` - Stockfish-style reduction)

---

## Root Enhancements

### 7. Aspiration Windows
**Purpose:** Speed up search by focusing on a narrow window around the expected best value.

**Description:** Maintains a dynamic window of expected Q-values. If the best move falls within the window, the window narrows. If it falls outside, the window widens (fail-high/fail-low handling).

**Implementation:**
- **File:** `gpu/include/puct_mcts.h`
- **Lines:** 219-266 (`AspirationWindowState` struct)
- **Lines:** 554-557 (Configuration parameters)
- **File:** `gpu/src/puct_mcts.cpp`
- **Lines:** 332-343 (Window usage during search)
- **Lines:** 1114-1127 (Aspiration window helper methods)
- **Lines:** 1129-1155 (Child selection within window)

---

### 8. Temperature Decay Schedule
**Purpose:** Control exploration vs exploitation during final move selection.

**Description:** Uses AlphaZero-inspired temperature schedule that starts high (more exploration) and decays to a low value (more exploitation) as the search progresses.

**Implementation:**
- **File:** `gpu/include/puct_mcts.h`
- **Lines:** 269-296 (`TemperatureSchedule` struct)
- **Lines:** 559-562 (Temperature decay configuration)
- **File:** `gpu/src/puct_mcts.cpp`
- **Lines:** 176-179 (Temperature schedule initialization)
- **Lines:** 451-453 (Temperature usage in final move selection)
- **Lines:** 1157-1162 (`get_current_temperature()` method)

---

## Evaluation Improvements

### 9. Tapered Evaluation System
**Purpose:** Smoothly interpolate between middlegame and endgame evaluations.

**Description:** Uses a game phase value (0-256) to blend middlegame and endgame scores. As pieces are exchanged, the evaluation smoothly transitions from middlegame-oriented to endgame-oriented values.

**Implementation:**
- **File:** `gpu/include/evaluation.h`
- **Lines:** 64-74 (`calculate_phase()` - game phase calculation)
- **Lines:** 77-80 (`tapered_eval()` - interpolation function)
- **Lines:** 362, 763 (Usage in king evaluation and full evaluation)
- **File:** `gpu/src/kernels/evaluation.cu`
- **Lines:** 177-187 (`calculate_phase()` - GPU version)
- **Lines:** 191-194 (`tapered_eval()` - GPU version)
- **Lines:** 829-832 (Usage in GPU evaluation)

---

### 10. Material Imbalance Evaluation
**Purpose:** Account for strategic advantages of piece combinations beyond raw material count.

**Description:** Evaluates:
- **Three minors vs two minors:** Bonus for having three minor pieces when opponent also has three
- **Rook with no minors:** Penalty for having rooks without minor piece support

**Implementation:**
- **File:** `gpu/include/evaluation.h`
- **Lines:** 50-53 (Imbalance constants)
- **Lines:** 268-290 (Imbalance evaluation in `evaluate_material()`)
- **File:** `gpu/src/kernels/evaluation.cu`
- **Lines:** 49-61 (Imbalance constants)
- **Lines:** 261-283 (Imbalance evaluation in GPU)

---

### 11. Enhanced Bishop Pair with Wing Pawn Bonuses
**Purpose:** Additional bonus for bishop pair when pawns are on both sides of the board.

**Description:** The bishop pair is more valuable when pawns are present on both wings (queenside and kingside), as the bishops can attack weaknesses on both sides of the board.

**Implementation:**
- **File:** `gpu/include/evaluation.h`
- **Lines:** 30-32 (Bishop pair constants including `EVAL_BISHOP_PAIR_WINGS`)
- **Lines:** 234-250, 252-266 (Wing pawn enhancement logic)
- **File:** `gpu/src/kernels/evaluation.cu`
- **Lines:** 22-24 (Bishop pair constants)
- **Lines:** 228-243, 245-259 (Wing pawn enhancement in GPU)

**Additional evaluation improvements** (Stockfish 10 reference):
- Center control bonuses (knights, bishops)
- Extended center control
- Rook on 7th rank bonus
- Rook on open/semi-open files
- Doubled rooks bonus
- King pawn shield evaluation
- Pawn structure (passed, isolated, doubled, backward pawns)
- Piece-square tables for all pieces

---

## Test Updates

### 12. Added 28 New Tests (BS2830 Suite)
**Purpose:** Expand test coverage with additional tactical positions.

**Description:** Added 28 new test positions from the Amundsen BS2830 test suite for comprehensive engine testing.

**Implementation:**
- **File:** `tests/include/test_positions.h`
- **Lines:** 20-47 (`get_bratko_kopec_suite()` - 24 Bratko-Kopec positions)
- **Lines:** 50-58 (`get_wac_suite()` - Win at Chess positions)
- **Reference:** https://github.com/johnbergbom/Amundsen/blob/master/testsuites/bs2830.epd

---

## Benchmark Implementation

### 13. CPU vs GPU Comparison Benchmarks
**Purpose:** Enable performance comparison between CPU and GPU engine implementations.

**Description:** Implemented comparison benchmarks that measure:
- Nodes per second (CPU)
- Playouts per second (GPU)
- Head-to-head match play with Elo calculation
- Fixed-time throughput benchmarks

**Implementation:**
- **File:** `tests/src/cpu_engine_wrapper.cpp`
  - Adapter for CPU Negamax engine
- **File:** `tests/src/gpu_engine_wrapper.cpp`
  - Adapter for GPU PUCT/MCTS engine
- **File:** `tests/src/benchmark_throughput.cpp`
  - Measures nodes/sec (CPU) and playouts/sec (GPU)
- **File:** `tests/src/benchmark_matches.cpp`
  - Plays games between CPU and GPU engines, computes Elo difference
- **File:** `tests/src/benchmark_fixed_time.cpp`
  - Fixed-time comparison benchmarks

---

## Summary of Key Files

| File | Purpose |
|------|---------|
| `gpu/improvements.txt` | Original improvement specification |
| `gpu/include/puct_mcts.h` | PUCT engine configuration and structures |
| `gpu/src/puct_mcts.cpp` | Core MCTS implementation |
| `gpu/include/evaluation.h` | CPU-side evaluation functions |
| `gpu/src/kernels/evaluation.cu` | GPU evaluation kernels |
| `gpu/src/kernels/playouts.cu` | GPU playout kernels |
| `gpu/src/kernels/tactical_search.cu` | Tactical solver kernel |
| `tests/include/test_positions.h` | Test position definitions |
| `tests/src/benchmark_*.cpp` | Benchmark implementations |
