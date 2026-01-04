# GPU Chess Engine - Changelog

## Purpose
This changelog tracks the development progress of the GPU-accelerated Monte Carlo Chess Engine.
Use this as a memory of what has been implemented, tested, and verified.

---

## v2.0 - Fresh Start (2026-01-04)

### Current State
- **v1 Engine**: Working Monte Carlo engine with array-based board representation
- **Goal**: Properly implement bitboard-based chess logic for GPU, then parallelize

### Cleanup (Completed)
- [x] Moved v1 files to `v1_reference/` directory
- [x] Removed incomplete v2 files

---

## Implementation Log

### 2026-01-04 - Project Cleanup
- Moved 6 v1 files to `v1_reference/` for safekeeping
- Deleted 4 incomplete v2 bitboard files
- Clean slate established for proper GPU chess logic implementation

### 2026-01-04 - Phase 1: Core Chess Logic (Step 1-2)
- Created `gpu_chess_types.cuh`: Position struct (12 bitboards + state), Move encoding (16-bit), constants
- Created `gpu_chess_bitops.cuh`: CUDA intrinsics (pop_lsb, count_bits), shift helpers, Kogge-Stone sliding attacks
- Architecture: sm_75 (GTX 1660 Super), shared memory strategy for warp-level simulations

### 2026-01-04 - Phase 1: Move Generation & Position (Step 3-4)
- Created `gpu_chess_movegen.cuh`: Full legal move generation with pins, checks, castling, en passant, promotions
- Created `gpu_chess_position.cuh`: make_move (handles all special moves), flip_position (Jaglavak technique), game state detection
- Created `gpu_chess_test.cu`: Perft kernel and move generation test kernel
- Created `gpu_chess_test.cpp`: Host test runner with perft validation
- Created `build_chess_test.bat`: Build script for testing

