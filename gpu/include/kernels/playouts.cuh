#ifndef KERNELS_PLAYOUTS_CUH
#define KERNELS_PLAYOUTS_CUH

#include "common.cuh"
#include "movegen.cuh"

// Helper functions defined inline in header
__device__ __forceinline__ bool is_tactical_move(const BoardState* pos, Move m) {
    int move_type = (m >> 12) & 0xF;
    if (move_type >= MOVE_PROMO_N && move_type <= MOVE_PROMO_CAP_Q) return true;
    if (move_type == MOVE_CAPTURE || move_type == MOVE_EP_CAPTURE) return true;
    return false;
}

__device__ __forceinline__ int generate_tactical_moves(const BoardState* pos, Move* moves) {
    Move all_moves[MAX_MOVES];
    int total = generate_legal_moves(pos, all_moves);
    int count = 0;
    for (int i = 0; i < total; i++) {
        if (is_tactical_move(pos, all_moves[i])) {
            moves[count++] = all_moves[i];
        }
    }
    return count;
}

// Non-inline helper functions
__device__ int mvv_lva_score(const BoardState* pos, Move m);
__device__ int see_capture(const BoardState* pos, Move m);

// Quiescence search - defined in playouts.cu
__device__ int quiescence_search_simple(const BoardState* pos, int max_depth);

// Kernel declarations
__global__ void RandomPlayout(const BoardState* __restrict__ starting_boards, float* __restrict__ results, int numBoards, unsigned int seed);
__global__ void EvalPlayout(const BoardState* __restrict__ starting_boards, float* __restrict__ results, int numBoards, unsigned int seed);
__global__ void StaticEval(const BoardState* __restrict__ boards, float* __restrict__ results, int numBoards);
__global__ void QuiescencePlayout(const BoardState* __restrict__ starting_boards, float* __restrict__ results, int numBoards, unsigned int seed, int max_q_depth);

#endif
