#ifndef KERNELS_COMMON_CUH
#define KERNELS_COMMON_CUH

#include "../chess_types.cuh"
#include <curand_kernel.h>

// Constants for MCTS
#define MAX_PLAYOUT_MOVES 500  // Maximum moves per playout
#define BLOCK_SIZE 256         // Threads per block

// Magic bitboard constants
#define ROOK_MAGIC_BITS   12
#define BISHOP_MAGIC_BITS 9

// Mate scores
#define MATE_SCORE 30000
#define INF_SCORE 32000

// Attack tables in constant memory
extern __constant__ Bitboard g_KNIGHT_ATTACKS[64];
extern __constant__ Bitboard g_KING_ATTACKS[64];
extern __constant__ Bitboard g_PAWN_ATTACKS[2][64];

extern __constant__ Bitboard g_ROOK_MAGICS[64];
extern __constant__ Bitboard g_BISHOP_MAGICS[64];
extern __constant__ Bitboard g_ROOK_MASKS[64];
extern __constant__ Bitboard g_BISHOP_MASKS[64];

// Large attack tables in global memory
extern __device__ Bitboard g_ROOK_ATTACKS[64][1 << ROOK_MAGIC_BITS];
extern __device__ Bitboard g_BISHOP_ATTACKS[64][1 << BISHOP_MAGIC_BITS];

#endif
