
#pragma once
#include <cuda_runtime.h>

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
#define MAX_PLAYOUT_MOVES 200

// ============================================================================
// GPU Constant Memory Declarations
// ============================================================================
extern __constant__ int d_piece_values[6];
extern __constant__ int d_pawn_table[64];
extern __constant__ int d_knight_table[64];
extern __constant__ int d_bishop_table[64];
extern __constant__ int d_rook_table[64];
extern __constant__ int d_queen_table[64];
extern __constant__ int d_king_table[64];

// ============================================================================
// GPU Data Structures
// ============================================================================
struct Move {
    int from;
    int to;
    int promotion; // 0 = none, 2=knight, 3=bishop, 4=rook, 5=queen
    int capture;   // Captured piece
    int piece;     // Moving piece
    float score;   // Heuristic score for move ordering
};

struct Position {
    int board[64];
    int side_to_move; // GPU_WHITE or GPU_BLACK
    bool castling_rights[4]; // WK, WQ, BK, BQ
    int en_passant; // -1 or square index
    int halfmove_clock;
    int fullmove_number;
};
