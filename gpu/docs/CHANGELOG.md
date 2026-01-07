# Changelog

All notable changes to the GPU MCTS Chess Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [0.5.0] - 2026-01-06

### Added - Phase 5: Tactical Position Testing & MCTS Enhancements

- **Check Extension in MCTS Tree** (Phase 5.1):
  - Added `is_check` and `gives_check` flags to MCTSNode
  - Modified `expand()` to detect and prioritize checking moves
  - UCB1 formula now boosts checking moves by 50% (`base_value *= 1.5`)
  - Checking moves are preferentially selected during expansion

- **Mate Distance Heuristic** (Phase 5.2):
  - Added `mate_distance` field to MCTSNode (>0 = we mate in N, <0 = they mate in N)
  - Immediate checkmate detection during node expansion
  - Mate distance propagation during backpropagation
  - UCB1 bonus for forcing mate: `+100.0 / mate_distance`
  - UCB1 penalty for getting mated: `-100.0 / (-mate_distance)`

- **Documentation**:
  - `docs/TACTICAL_ANALYSIS_PHASE5.md` - Comprehensive tactical failure analysis
  - Detailed root cause analysis of MCTS tactical limitations
  - Progressive improvement strategy (Phase 5.1, 5.2, 5.3)
  - Expected results and success criteria

### Changed

- **MCTSNode structure** (`include/mcts.h`):
  - Extended with tactical information fields
  - Enhanced UCB1 calculation for tactical awareness

- **Tree Expansion** (`src/mcts.cpp`):
  - Prioritizes checking moves among untried moves
  - Detects checkmate in child positions
  - Improved terminal position handling

- **Backpropagation** (`src/mcts.cpp`):
  - Propagates mate distance up the tree
  - Tracks shortest mate path for each node
  - Handles forced mate situations

### Technical Details

**Check Detection:**
```cpp
// In expand():
BoardState test_state = node->state;
cpu_movegen::make_move_cpu(&test_state, move);
if (cpu_movegen::in_check_cpu(&test_state)) {
    selected_move = move;  // Prioritize checking move
}
```

**Mate Distance:**
```cpp
// In expand():
if (num_child_moves == 0 && child_ptr->is_check) {
    child_ptr->mate_distance = 1;  // Checkmate!
}

// In UCB1:
if (mate_distance > 0) {
    base_value += 100.0f / mate_distance;
}
```

### Known Limitations

**MCTS vs Negamax Solver:**
- The test suite includes both MCTS engine and GPU negamax solver
- Tactical tests use negamax solver (depth-2 alpha-beta search)
- MCTS improvements are for tree search, not the negamax solver

**Expected Tactical Performance:**
- Mate-in-1: Should improve significantly with check extension
- Mate-in-2+: Still challenging due to MCTS exploration breadth
- Strategic positions: MCTS strength unchanged (already good)

### Future Work (Deferred)

**Phase 5.3: Progressive Widening**
- Limit initial children to top K moves (K=3)
- Expand more moves based on visit count
- Further reduce exploration breadth

**Phase 5.4: Deeper Quiescence**
- 2-3 ply recursive quiescence (GPU stack risk)
- Experimental - may cause instability

## [0.4.0] - 2026-01-05

### Added
- **Quiescence Search**: Tactical extension for captures and promotions
  - Non-recursive 1-ply design for GPU safety
  - `generate_tactical_moves()` filters for tactical moves only
  - `quiescence_search_simple()` with stand-pat evaluation
  - `QuiescencePlayout` kernel (5 random moves + quiescence)

- **MVV-LVA Move Ordering**: Most Valuable Victim - Least Valuable Attacker
  - Prioritizes good captures (QxP over PxQ)
  - Promotion captures scored highest (10,000+)
  - Enables beta cutoffs in quiescence

- **Tactical Move Detection**:
  - `is_tactical_move()` identifies captures and promotions
  - Filters moves for quiescence search

- **New Playout Mode**: `PlayoutMode::QUIESCENCE`
  - Short random playout + quiescence search
  - Configurable depth via `search_config.quiescence_depth`
  - Default depth: 3 ply

- **Documentation**: `docs/TACTICAL.md` - Comprehensive tactical features guide

### Changed
- Default quiescence depth: 3 ply (conservative for GPU stack)
- Test runner now uses QUIESCENCE mode for engine tests

### Performance
- Quiescence mode: ~180-200k sims/sec (20-25% slower than EVAL_HYBRID)
- Memory safe: Non-recursive design avoids GPU stack overflow

### Known Limitations

**Tactical Test Results**: 0/5 passing (unchanged from v0.3.0)

This is an inherent MCTS limitation, not an implementation bug:
- MCTS explores broadly (UCB1), not deeply on forcing lines
- Tactical puzzles require deep, narrow search (alpha-beta strength)
- Mate-in-1 requires finding ONE move among ~20, MCTS tries all equally
- Quiescence improves evaluation but doesn't change tree policy

**MCTS vs Alpha-Beta**:
- MCTS: Strategic, broad, positional
- Alpha-Beta: Tactical, deep, forcing
- This engine: Correct MCTS with quiescence, but MCTS isn't designed for tactics

See `docs/TACTICAL.md` for detailed analysis and recommendations.

## [0.3.0] - 2026-01-05

### Added
- **Static Evaluation Function**: Material counting with piece-square tables
  - Material values: P=100, N=320, B=330, R=500, Q=900
  - PST for all pieces including separate king tables for middlegame/endgame
  - Game phase detection (interpolates between MG and EG king PST)

- **Playout Modes**: Three modes for MCTS leaf evaluation
  - `RANDOM`: Pure random playouts (original behavior)
  - `EVAL_HYBRID`: Short random + static evaluation (new default)
  - `STATIC_EVAL`: Immediate static evaluation (fastest)

- **GPU Evaluation Kernel**: Fully parallel evaluation on GPU
  - PST tables in constant memory
  - Sigmoid function to convert centipawn to win probability
  - `EvalPlayout` kernel for hybrid mode
  - `StaticEval` kernel for immediate evaluation

- **Documentation**: Comprehensive evaluation documentation (`docs/EVALUATION.md`)

### Changed
- Default playout mode changed from RANDOM to EVAL_HYBRID
- `search()` now uses `searchWithConfig()` internally for consistency
- Improved score differentiation (no longer all ~0.5)

### Technical Details
- Sigmoid scale factor: 400 centipawns
- Hybrid playout depth: 10 random moves before evaluation
- Benchmarked at ~250k simulations/sec

### Known Limitations
- MCTS with evaluation still struggles with sharp tactics
- Engine tests (mate-in-1, forks) remain challenging
- Quiescence search not implemented

## [0.2.1] - 2026-01-05

### Fixed
- **En Passant File Wrap-Around Bug**: Fixed critical bug where pawns on h-file could generate
  spurious en passant captures to a-file due to arithmetic wrap-around (e.g., after 1.h4 a5,
  the h4 pawn would incorrectly generate h4xa6). Added file boundary checks.

- **Promotion Capture Underpromotion**: Fixed bug where promotion captures only generated
  queen promotions. Now generates all 4 promotion types (Q, R, B, N) for capture promotions,
  matching the push promotion behavior.

### Added
- Expanded perft test suite from 7 to 16 tests
- Deeper perft tests: Starting position depth 4 (197,281 nodes)
- Position 4 and 5 tests for castling edge cases and promotions
- Move generation documentation (`docs/MOVEGEN.md`)
- Debug tools for perft analysis (`tests/debug_perft.cpp`, `tests/debug_perft2.cpp`)

### Changed
- Updated ROADMAP to focused version (removed out-of-scope features)

### Test Results
- All 16 perft tests pass (100%)
- Move generation fully validated

## [0.2.0] - 2026-01-05

### Added
- **FEN Support**: Full FEN string parsing and generation
  - `FENParser::parse()` - Parse FEN to BoardState
  - `FENParser::toFEN()` - Generate FEN from BoardState
  - `FENParser::validate()` - Validate FEN string correctness
  - Support for castling rights, en passant, move counters

- **Configurable Search**: New `SearchConfig` struct for flexible search control
  - `max_depth` - Maximum tree depth limit
  - `max_iterations` - Maximum MCTS iterations
  - `time_limit_ms` - Time-based search termination
  - `exploration_constant` - Tunable UCB1 exploration
  - `simulations_per_batch` - Configurable GPU batch size
  - Preset configurations: `quick()`, `standard()`, `deep()`, `tournament()`

- **Search Result Tracking**: New `SearchResult` struct
  - Principal variation extraction
  - Node count and depth statistics
  - Elapsed time tracking

- **Test Infrastructure**: Comprehensive testing framework
  - `test_positions.h` - Curated test positions by difficulty
  - `test_runner.cpp` - Full test harness with CLI options
  - FEN validation tests
  - Perft move generation tests
  - Engine accuracy tests (easy/medium/hard)

- **Documentation**: Project documentation suite
  - `CHANGELOG.md` - Version history
  - `ARCHITECTURE.md` - System design documentation
  - `ROADMAP.md` - Future development priorities

### Changed
- `MCTSEngine::search()` now uses internal `SearchConfig`
- Added `searchWithConfig()` for full configuration control
- Added `searchToDepth()` for depth-limited search
- Dynamic GPU buffer resizing via `ensure_batch_capacity()`

## [0.1.0] - 2026-01-05

### Added
- **Core GPU MCTS Engine**
  - CPU-side MCTS tree with UCB1 selection
  - GPU-accelerated random playouts
  - Batched GPU operations (configurable batch size)

- **Bitboard Chess Representation**
  - 64-bit bitboards for each piece type and color
  - Magic bitboard lookup tables for sliding pieces
  - Pre-computed attack tables for knights and kings

- **Move Generation**
  - Legal move generation on GPU
  - Support for all piece types including promotions
  - Castling and en passant handling
  - Check detection and evasion

- **GPU Kernels**
  - `random_playout_kernel` - Monte Carlo playouts
  - `generate_legal_moves_gpu` - GPU move generation
  - cuRAND integration for random move selection

- **Performance Optimizations**
  - Pinned host memory for fast CPU-GPU transfers
  - Constant memory for attack lookup tables
  - Coalesced memory access patterns

### Fixed
- MSVC compatibility for bit manipulation intrinsics
- CUDA symbol linkage across translation units
- BoardState size assertion (104 bytes actual)

### Technical Details
- Target: NVIDIA GTX 1660 Super (Compute Capability 7.5)
- Architecture: sm_75
- BoardState: 104 bytes (12 bitboards + flags)
- Achieved: ~11,900 simulations/second

## Known Issues

### Move Generation
- **FIXED in v0.2.1**: En passant file wrap-around bug
- **FIXED in v0.2.1**: Promotion capture underpromotion bug
- All 16 perft tests now pass (100%)

### Engine Strength
1. **Random Playouts**: Pure random playouts cannot find tactical moves
   - All easy/medium/hard engine tests fail (0/12)
   - Scores hover around 0.5 (no meaningful evaluation)
   - **Fix**: Implement basic evaluation function (see ROADMAP.md)

### Missing Features (Out of Scope)
2. **Endgame Evaluation**: No tablebase support (out of scope)
3. **Opening Book**: No opening book integration (out of scope)
4. **Time Management**: Basic time control, no increment handling
5. **UCI Protocol**: Not implemented yet

### Test Results (v0.2.1)
```
FEN Parser Tests            8/8 (100.0%)
Perft Tests                16/16 (100.0%)
Easy Engine Tests           0/5 (0.0%)  <- Requires evaluation function
Medium Engine Tests         0/5 (0.0%)  <- Requires evaluation function
Hard Engine Tests           0/2 (0.0%)  <- Requires evaluation function
--------------------------------------------------
OVERALL                    24/35 (68.6%)
```

## Migration Guide

### From v0.1.0 to v0.2.0

The basic `search()` interface remains unchanged for backward compatibility:

```cpp
// Old way (still works)
Move best = engine.search(board, 10000);

// New way with configuration
SearchConfig config = SearchConfig::standard()
    .withDepth(10)
    .withTimeLimit(5000);
SearchResult result = engine.searchWithConfig(board, config);
```

FEN parsing is now available:

```cpp
BoardState board;
FENError err = FENParser::parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", board);
if (err == FEN_OK) {
    // Use board
}
```
