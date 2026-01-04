//N - v1: Shared type definitions (no CUDA dependencies)
#pragma once
#include <stdint.h>

// ============================================================================
// GPU Constants and Defines
// ============================================================================
#define EMPTY 0
#define W_PAWN 1
#define W_KNIGHT 2
#define W_BISHOP 3
#define W_ROOK 4
#define W_QUEEN 5
#define W_KING 6
#define B_PAWN 9
#define B_KNIGHT 10
#define B_BISHOP 11
#define B_ROOK 12
#define B_QUEEN 13
#define B_KING 14

#define GPU_WHITE 0
#define GPU_BLACK 1

#define MAX_MOVES 256
#define MAX_PLAYOUT_MOVES 150

//N - v1: MCTS configuration constants
#define MCTS_MAX_NODES 65536          // Maximum tree nodes per kernel
#define MCTS_EXPLORATION_CONSTANT 1.414f  // UCB exploration parameter (sqrt(2))
#define MCTS_VIRTUAL_LOSS_VALUE 1     // Virtual loss increment
#define MCTS_QUIESCENCE_EXTENSION 8   // Extra moves for tactical positions

//N - v1: Transposition table configuration
#define GPU_TT_SIZE 1048576           // 1M entries (~16MB with 16-byte entries)
#define GPU_TT_MASK (GPU_TT_SIZE - 1)

//N - v1: History table configuration
#define HISTORY_MAX 32767
#define HISTORY_MIN -32768

// ============================================================================
//N - v1: Compact Position structure (was 280 bytes, now 72 bytes)
// ============================================================================
struct CompactPosition {
    int8_t board[64];       // 64 bytes - piece per square
    uint8_t flags;          // 1 byte - side_to_move (bit 0), castling (bits 1-4)
    int8_t en_passant;      // 1 byte - -1 or square index
    uint8_t halfmove_clock; // 1 byte
    uint8_t fullmove_low;   // 1 byte - low byte of fullmove
    uint32_t padding;       // 4 bytes for alignment
};  // Total: 72 bytes (was 280)

//N - v1: Accessor macros for compact position
#define POS_SIDE_TO_MOVE(p) ((p).flags & 0x01)
#define POS_SET_SIDE(p, s) ((p).flags = ((p).flags & 0xFE) | ((s) & 0x01))
#define POS_CASTLING(p) (((p).flags >> 1) & 0x0F)
#define POS_SET_CASTLING(p, c) ((p).flags = ((p).flags & 0xE1) | (((c) & 0x0F) << 1))

// Legacy Position for compatibility (still used in host interface)
struct Position {
    int board[64];
    int side_to_move;
    bool castling_rights[4];
    int en_passant;
    int halfmove_clock;
    int fullmove_number;
};

// ============================================================================
//N - v1: Compact Move structure (was 24 bytes, now 8 bytes)
// ============================================================================
struct CompactMove {
    uint8_t from;           // 1 byte - source square
    uint8_t to;             // 1 byte - target square
    uint8_t piece;          // 1 byte - moving piece
    uint8_t capture;        // 1 byte - captured piece
    uint8_t promotion;      // 1 byte - promotion piece (0 = none)
    uint8_t flags;          // 1 byte - reserved for castling, ep flags
    int16_t score;          // 2 bytes - heuristic score (was float)
};  // Total: 8 bytes (was 24)

// Legacy Move for host compatibility
struct Move {
    int from;
    int to;
    int promotion;
    int capture;
    int piece;
    float score;
};

// ============================================================================
//N - v1: GPU Transposition Table Entry (16 bytes)
// ============================================================================
struct GPUTTEntry {
    uint64_t key;           // 8 bytes - Zobrist hash
    int16_t score;          // 2 bytes - evaluation
    uint8_t depth;          // 1 byte - search depth / visit count
    uint8_t flag;           // 1 byte - bound type (0=exact, 1=lower, 2=upper)
    uint8_t best_from;      // 1 byte - best move source
    uint8_t best_to;        // 1 byte - best move target
    uint16_t generation;    // 2 bytes - age for replacement
};  // Total: 16 bytes

// ============================================================================
//N - v1: MCTS Tree Node (32 bytes)
// ============================================================================
struct MCTSNode {
    int32_t parent;              // 4 bytes - parent node index (-1 for root)
    int32_t first_child;         // 4 bytes - first child index (-1 if none)
    int32_t num_children;        // 4 bytes - number of children
    int32_t visits;              // 4 bytes - visit count (N)
    float total_value;           // 4 bytes - sum of backpropagated values (W)
    int32_t virtual_loss;        // 4 bytes - N - v1: virtual loss counter
    uint8_t move_from;           // 1 byte - move that led here
    uint8_t move_to;             // 1 byte
    uint8_t move_piece;          // 1 byte
    uint8_t move_promotion;      // 1 byte
    uint32_t reserved;           // 4 bytes - padding
};  // Total: 32 bytes

// ============================================================================
//N - v1: Simulation result for reduction
// ============================================================================
struct SimulationResult {
    float score;
    int move_idx;
};
