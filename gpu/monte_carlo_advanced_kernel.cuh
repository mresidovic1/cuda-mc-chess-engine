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
// GPU Constant Memory Declarations/Definitions
// ============================================================================
#ifdef GPU_CONST_DEF
__constant__ int d_piece_values[6] = {100, 300, 320, 500, 900, 0};
__constant__ int d_pawn_table[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_knight_table[64] = {
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50
};
__constant__ int d_bishop_table[64] = {
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20
};
__constant__ int d_rook_table[64] = {
     0,  0,  5, 10, 10,  5,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};
__constant__ int d_queen_table[64] = {
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20
};
__constant__ int d_king_table[64] = {
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20
};
#else
extern __constant__ int d_piece_values[6];
extern __constant__ int d_pawn_table[64];
extern __constant__ int d_knight_table[64];
extern __constant__ int d_bishop_table[64];
extern __constant__ int d_rook_table[64];
extern __constant__ int d_queen_table[64];
extern __constant__ int d_king_table[64];
#endif

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

// ============================================================================
// Kernel Function Declaration
// ============================================================================
__global__ void monte_carlo_simulate_kernel(
    const Position root_position,
    const Move root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed
);

// ============================================================================
// Launch Function Declaration
// ============================================================================
extern "C" void launch_monte_carlo_simulate_kernel(
    const Position* root_position,
    const Move* root_move,
    int num_simulations_per_thread,
    float* results,
    unsigned long long seed,
    int blocks,
    int threads_per_block
);
